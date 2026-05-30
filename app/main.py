import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


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
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
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

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
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
        "error": model_error,
    }


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
