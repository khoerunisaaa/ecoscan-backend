import io
import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


APP_NAME = os.getenv("APP_NAME", "EcoScan API")
API_PREFIX = "/api/v1"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models/best_model.keras"))
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "300"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "8"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MODEL_CLASSES = [
    item.strip()
    for item in os.getenv(
        "MODEL_CLASSES",
        "Anorganik,B3,Kertas,Organik,Residu",
    ).split(",")
    if item.strip()
]
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_HISTORY_TABLE = os.getenv("SUPABASE_HISTORY_TABLE", "scan_history")
SUPABASE_USERS_TABLE = os.getenv("SUPABASE_USERS_TABLE", "app_users")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app")


CATEGORY_MAP = {
    "biological": "Organik",
    "paper": "Organik",
    "cardboard": "Organik",
    "organik": "Organik",
    "kertas": "Organik",
    "plastic": "Anorganik",
    "glass": "Anorganik",
    "metal": "Anorganik",
    "clothes": "Anorganik",
    "shoes": "Anorganik",
    "anorganik": "Anorganik",
    "battery": "B3",
    "trash": "B3",
    "b3": "B3",
    "residu": "B3",
}

HANDLING_ADVICE = {
    "Organik": "Pisahkan dari sampah kering, lalu olah menjadi kompos atau serahkan ke pengelola organik.",
    "Anorganik": "Bersihkan, keringkan, lalu setorkan ke bank sampah atau pusat daur ulang.",
    "B3": "Jangan dicampur dengan sampah lain. Simpan tertutup dan serahkan ke titik pengumpulan limbah B3.",
}


app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_origin_regex=CORS_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model: Any | None = None
model_error: str | None = None
memory_history: list[dict[str, Any]] = []
memory_users: dict[str, dict[str, Any]] = {}


class AuthRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=6, max_length=128)
    name: str | None = Field(default=None, max_length=80)


def load_model_once() -> Any | None:
    global model, model_error

    if model is not None or model_error is not None:
        return model

    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as exc:  # pragma: no cover - depends on TensorFlow/runtime
        model_error = str(exc)
        return None


@app.on_event("startup")
def startup() -> None:
    load_model_once()


def map_category(label: str) -> str:
    normalized = label.strip().lower()
    return CATEGORY_MAP.get(normalized, "Anorganik")


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", normalized):
        raise HTTPException(status_code=422, detail="Format email tidak valid.")
    return normalized


def sanitize_name(name: str | None, email: str) -> str:
    clean_name = (name or "").strip()
    if clean_name:
        return clean_name[:80]
    return email.split("@", 1)[0].replace(".", " ").replace("_", " ").title()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"pbkdf2_sha256$120000${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt, digest = password_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        int(iterations),
    ).hex()
    return hmac.compare_digest(candidate, digest)


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "created_at": user["created_at"],
    }


def supabase_headers(prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


async def fetch_user_by_email(email: str) -> dict[str, Any] | None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return memory_users.get(email)

    params = {
        "select": "*",
        "email": f"eq.{email}",
        "limit": "1",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_USERS_TABLE}",
            params=params,
            headers=supabase_headers(),
        )
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else None


async def create_user(email: str, password: str, name: str) -> dict[str, Any]:
    user = {
        "id": str(uuid4()),
        "email": email,
        "name": name,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        memory_users[email] = user
        return user

    headers = supabase_headers("return=representation")
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_USERS_TABLE}",
            json=user,
            headers=headers,
        )
    response.raise_for_status()
    return response.json()[0]


async def register_user(payload: AuthRequest) -> dict[str, Any]:
    email = normalize_email(payload.email)
    name = sanitize_name(payload.name, email)

    existing_user = await fetch_user_by_email(email)
    if existing_user:
        raise HTTPException(status_code=409, detail="Email sudah terdaftar.")

    try:
        user = await create_user(email, payload.password, name)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal membuat akun di Supabase: {exc}") from exc

    return {"user": public_user(user)}


async def login_user(payload: AuthRequest) -> dict[str, Any]:
    email = normalize_email(payload.email)

    try:
        user = await fetch_user_by_email(email)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal membaca akun dari Supabase: {exc}") from exc

    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Email atau password salah.")

    return {"user": public_user(user)}


def build_scan_response(
    *,
    filename: str,
    predicted_label: str,
    confidence: float,
    raw_predictions: list[float],
) -> dict[str, Any]:
    category = map_category(predicted_label)
    return {
        "id": str(uuid4()),
        "filename": filename,
        "predicted_class": category,
        "specific_class": predicted_label,
        "category": category,
        "confidence": confidence,
        "handling_advice": HANDLING_ADVICE[category],
        "raw_predictions": raw_predictions,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def save_history(record: dict[str, Any]) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        memory_history.insert(0, record)
        del memory_history[50:]
        return

    payload = {
        "id": record["id"],
        "filename": record["filename"],
        "predicted_class": record["specific_class"],
        "category": record["category"],
        "confidence": record["confidence"],
        "handling_advice": record["handling_advice"],
        "raw_predictions": record["raw_predictions"],
        "created_at": record["created_at"],
    }
    headers = {
        **supabase_headers("return=minimal"),
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}",
            json=payload,
            headers=headers,
        )
    response.raise_for_status()


async def fetch_history(limit: int) -> list[dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return memory_history[:limit]

    headers = supabase_headers()
    params = {
        "select": "*",
        "order": "created_at.desc",
        "limit": str(limit),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}",
            params=params,
            headers=headers,
        )
    response.raise_for_status()
    rows = response.json()
    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "predicted_class": row["category"],
            "specific_class": row["predicted_class"],
            "category": row["category"],
            "confidence": float(row["confidence"]),
            "handling_advice": row.get("handling_advice") or HANDLING_ADVICE[row["category"]],
            "raw_predictions": row.get("raw_predictions") or [],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


async def classify_upload(file: UploadFile) -> dict[str, Any]:
    current_model = load_model_once()
    if current_model is None:
        detail = "Model belum dimuat."
        if model_error:
            detail = f"{detail} Detail: {model_error}"
        raise HTTPException(status_code=503, detail=detail)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar JPG, PNG, atau WebP.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"Ukuran gambar maksimal {MAX_UPLOAD_SIZE_MB} MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="File gambar tidak valid atau rusak.") from exc

    image = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    batch = np.expand_dims(image_array, axis=0)

    prediction = current_model.predict(batch, verbose=0)
    scores = prediction.tolist()[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = (
        MODEL_CLASSES[predicted_index]
        if predicted_index < len(MODEL_CLASSES)
        else f"class_{predicted_index}"
    )

    record = build_scan_response(
        filename=file.filename or "upload.jpg",
        predicted_label=predicted_label,
        confidence=float(np.max(prediction)),
        raw_predictions=[float(score) for score in scores],
    )

    try:
        await save_history(record)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal menyimpan riwayat ke Supabase: {exc}") from exc

    return record


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "EcoScan API aktif. Gunakan /api/v1/classify untuk klasifikasi gambar."}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model_error is None else "degraded",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "database": "supabase" if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else "memory",
        "cors_origins": CORS_ORIGINS,
        "cors_origin_regex": CORS_ORIGIN_REGEX,
        "auth": "supabase" if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else "memory",
        "error": model_error,
    }


@app.post(f"{API_PREFIX}/auth/register")
async def auth_register(payload: AuthRequest) -> dict[str, Any]:
    return await register_user(payload)


@app.post(f"{API_PREFIX}/auth/login")
async def auth_login(payload: AuthRequest) -> dict[str, Any]:
    return await login_user(payload)


@app.post(f"{API_PREFIX}/classify")
async def classify(file: UploadFile = File(...)) -> dict[str, Any]:
    return await classify_upload(file)


@app.post("/predict")
async def predict_compat(file: UploadFile = File(...)) -> dict[str, Any]:
    return await classify_upload(file)


@app.get(f"{API_PREFIX}/history")
async def history(limit: int = 20) -> dict[str, Any]:
    safe_limit = min(max(limit, 1), 100)
    try:
        items = await fetch_history(safe_limit)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil riwayat dari Supabase: {exc}") from exc
    return {"items": items}
