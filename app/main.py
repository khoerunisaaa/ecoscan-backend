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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
SUPABASE_COMMUNITY_POSTS_TABLE = os.getenv("SUPABASE_COMMUNITY_POSTS_TABLE", "community_posts")
SUPABASE_COMMUNITY_COMMENTS_TABLE = os.getenv("SUPABASE_COMMUNITY_COMMENTS_TABLE", "community_comments")
SUPABASE_TRIVIA_TABLE = os.getenv("SUPABASE_TRIVIA_TABLE", "eco_trivia")
SUPABASE_CHALLENGES_TABLE = os.getenv("SUPABASE_CHALLENGES_TABLE", "weekly_challenges")
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
    "paper": "Kertas",
    "cardboard": "Kertas",
    "organik": "Organik",
    "kertas": "Kertas",
    "plastic": "Anorganik",
    "glass": "Anorganik",
    "metal": "Anorganik",
    "clothes": "Anorganik",
    "shoes": "Anorganik",
    "anorganik": "Anorganik",
    "battery": "B3",
    "trash": "Residu",
    "b3": "B3",
    "residu": "Residu",
}

HANDLING_ADVICE = {
    "Organik": "Pisahkan dari sampah kering, lalu olah menjadi kompos atau serahkan ke pengelola organik.",
    "Anorganik": "Bersihkan, keringkan, lalu setorkan ke bank sampah atau pusat daur ulang.",
    "B3": "Jangan dicampur dengan sampah lain. Simpan tertutup dan serahkan ke titik pengumpulan limbah B3.",
    "Kertas": "Pastikan kertas kering, tidak berminyak, lalu kumpulkan terpisah untuk didaur ulang.",
    "Residu": "Masukkan ke wadah residu karena sulit didaur ulang atau dikomposkan secara mandiri.",
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
memory_community_posts: list[dict[str, Any]] = [
    {
        "id": "6a58f05e-6815-46f4-89c3-70ad589db1f6",
        "author": "Nadia",
        "badge": "Eco Mentor",
        "title": "Tips memilah sampah dapur",
        "body": "Pisahkan kulit buah dan sisa sayur sejak awal. Wadah kecil di dekat meja masak bikin kebiasaan ini lebih gampang.",
        "type": "post",
        "tag": "",
        "likes": 128,
        "comments": [
            {
                "id": "b3b8e585-dbdc-41d1-9477-4c37623682fb",
                "author": "Sari",
                "body": "Aku pakai wadah bekas es krim, ternyata praktis banget.",
                "created_at": "Hari ini",
                "replies": [],
            }
        ],
        "created_at": "Hari ini",
    },
    {
        "id": "65055bff-4a69-42c8-b163-cb59d1d2a900",
        "author": "EcoScan",
        "badge": "Panduan",
        "title": "Simpan limbah B3 terpisah",
        "body": "Baterai, lampu, dan obat kedaluwarsa jangan dicampur dengan sampah rumah tangga.",
        "type": "tip",
        "tag": "Keamanan",
        "likes": 51,
        "comments": [],
        "created_at": "Minggu ini",
    },
]
memory_challenge: dict[str, Any] = {
    "id": "weekly-plastic-10",
    "title": "Scan 10 sampah plastik",
    "description": "Kumpulkan scan plastik bersih minggu ini dan bagikan tips pemilahanmu.",
    "current": 6,
    "target": 10,
    "reward": 80,
    "ends_at": "Minggu ini",
}
memory_trivia: list[dict[str, Any]] = [
    {
        "id": "trivia-plastic",
        "title": "Fakta Daur Ulang",
        "text": "Botol plastik PET sebaiknya dicuci, dikeringkan, lalu disetor ke bank sampah.",
        "details": "Botol PET yang bersih lebih mudah diterima bank sampah karena tidak mencemari material lain.",
        "type": "plastic",
    },
    {
        "id": "trivia-organic",
        "title": "Sampah Organik",
        "text": "Sisa sayur dan buah bisa diolah menjadi kompos untuk mengurangi sampah rumah.",
        "details": "Sampah organik seperti kulit buah, sisa sayur, ampas kopi, dan daun kering bisa masuk komposter.",
        "type": "organic",
    },
]


class AuthRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=6, max_length=128)
    name: str | None = Field(default=None, max_length=80)


class CommunityPostCreate(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=3, max_length=1000)
    author: str = Field(default="Eco Warrior", max_length=80)
    badge: str = Field(default="Anggota", max_length=40)
    user_id: str | None = None
    tag: str | None = Field(default=None, max_length=40)
    type: str = Field(default="post", pattern="^(post|tip)$")


class CommunityCommentCreate(BaseModel):
    body: str = Field(min_length=1, max_length=600)
    author: str = Field(default="Eco Warrior", max_length=80)
    user_id: str | None = None
    parent_id: str | None = None


class CommunityLikeUpdate(BaseModel):
    liked: bool = True


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    email: str | None = Field(default=None, min_length=5, max_length=254)
    avatar_url: str | None = Field(default=None, max_length=3_000_000)


class PasswordUpdate(BaseModel):
    current_password: str = Field(min_length=6, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


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
        "avatar_url": user.get("avatar_url") or "",
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


async def fetch_user_by_id(user_id: str) -> dict[str, Any] | None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return next((item for item in memory_users.values() if item["id"] == user_id), None)

    params = {
        "select": "*",
        "id": f"eq.{user_id}",
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
        "avatar_url": "",
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
    user_id: str | None = None,
) -> dict[str, Any]:
    category = map_category(predicted_label)
    return {
        "id": str(uuid4()),
        "user_id": user_id,
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
        "user_id": record.get("user_id"),
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


async def fetch_history(limit: int, user_id: str | None = None) -> list[dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        rows = [item for item in memory_history if not user_id or item.get("user_id") == user_id]
        return rows[:limit]

    headers = supabase_headers()
    params = {
        "select": "*",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    if user_id:
        params["user_id"] = f"eq.{user_id}"

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
            "user_id": row.get("user_id"),
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


def calculate_streak(created_at_values: list[str]) -> int:
    scan_dates: set[Any] = set()
    for value in created_at_values:
        try:
            scan_dates.add(datetime.fromisoformat(value.replace("Z", "+00:00")).date())
        except (TypeError, ValueError):
            continue

    if not scan_dates:
        return 0

    today = datetime.now(timezone.utc).date()
    current = today if today in scan_dates else max(scan_dates)
    streak = 0

    while current in scan_dates:
        streak += 1
        current = current.fromordinal(current.toordinal() - 1)

    return streak


def build_stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    total_scans = len(rows)
    return {
        "total_scans": total_scans,
        "points": total_scans * 10,
        "streak": calculate_streak([row.get("created_at", "") for row in rows]),
    }


async def fetch_user_stats(user_id: str) -> dict[str, int]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        rows = [item for item in memory_history if item.get("user_id") == user_id]
        return build_stats_from_rows(rows)

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}",
            params={
                "select": "id,created_at",
                "user_id": f"eq.{user_id}",
                "order": "created_at.desc",
                "limit": "10000",
            },
            headers=supabase_headers(),
        )
    response.raise_for_status()
    return build_stats_from_rows(response.json())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_post(post: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": post["id"],
        "author": post.get("author") or "Eco Warrior",
        "badge": post.get("badge") or "Anggota",
        "title": post["title"],
        "body": post["body"],
        "type": post.get("type") or "post",
        "tag": post.get("tag") or "",
        "likes": int(post.get("likes") or 0),
        "comments": post.get("comments") or [],
        "created_at": post.get("created_at") or utc_now(),
    }


def nest_comments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comments_by_id: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []

    for row in rows:
        comment = {
            "id": row["id"],
            "author": row.get("author") or "Eco Warrior",
            "body": row["body"],
            "created_at": row.get("created_at") or utc_now(),
            "replies": [],
        }
        comments_by_id[comment["id"]] = comment

    for row in rows:
        comment = comments_by_id[row["id"]]
        parent_id = row.get("parent_id")
        if parent_id and parent_id in comments_by_id:
            comments_by_id[parent_id]["replies"].append(comment)
        else:
            roots.append(comment)

    return roots


def filter_posts(items: list[dict[str, Any]], search: str | None) -> list[dict[str, Any]]:
    keyword = (search or "").strip().lower()
    if not keyword:
        return items

    return [
        item
        for item in items
        if any(
            keyword in str(item.get(field, "")).lower()
            for field in ("title", "body", "author", "badge", "tag")
        )
    ]


async def fetch_community_posts(search: str | None = None) -> list[dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return filter_posts([public_post(post) for post in memory_community_posts], search)

    async with httpx.AsyncClient(timeout=10) as client:
        posts_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_COMMUNITY_POSTS_TABLE}",
            params={"select": "*", "order": "created_at.desc"},
            headers=supabase_headers(),
        )
        posts_response.raise_for_status()
        post_rows = posts_response.json()

        comments_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_COMMUNITY_COMMENTS_TABLE}",
            params={"select": "*", "order": "created_at.asc", "limit": "500"},
            headers=supabase_headers(),
        )
        comments_response.raise_for_status()
        comment_rows = comments_response.json()

    comments_by_post: dict[str, list[dict[str, Any]]] = {}
    for row in comment_rows:
        comments_by_post.setdefault(row["post_id"], []).append(row)

    items = []
    for row in post_rows:
        row["comments"] = nest_comments(comments_by_post.get(row["id"], []))
        items.append(public_post(row))

    return filter_posts(items, search)


async def create_community_post(payload: CommunityPostCreate) -> dict[str, Any]:
    post = {
        "id": str(uuid4()),
        "user_id": payload.user_id,
        "author": payload.author.strip() or "Eco Warrior",
        "badge": payload.badge.strip() or "Anggota",
        "title": payload.title.strip(),
        "body": payload.body.strip(),
        "type": payload.type,
        "tag": (payload.tag or "").strip(),
        "likes": 0,
        "created_at": utc_now(),
    }

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        memory_community_posts.insert(0, {**post, "comments": []})
        return public_post(memory_community_posts[0])

    headers = {**supabase_headers("return=representation"), "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_COMMUNITY_POSTS_TABLE}",
            json=post,
            headers=headers,
        )
    response.raise_for_status()
    saved = response.json()[0]
    saved["comments"] = []
    return public_post(saved)


async def add_community_comment(post_id: str, payload: CommunityCommentCreate) -> dict[str, Any]:
    comment = {
        "id": str(uuid4()),
        "post_id": post_id,
        "parent_id": payload.parent_id,
        "user_id": payload.user_id,
        "author": payload.author.strip() or "Eco Warrior",
        "body": payload.body.strip(),
        "created_at": utc_now(),
    }

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        target = next((post for post in memory_community_posts if post["id"] == post_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Postingan tidak ditemukan.")
        public_comment = {k: comment[k] for k in ("id", "author", "body", "created_at")}
        public_comment["replies"] = []
        if payload.parent_id:
            parent = next((item for item in target["comments"] if item["id"] == payload.parent_id), None)
            if parent:
                parent.setdefault("replies", []).append(public_comment)
            else:
                target["comments"].append(public_comment)
        else:
            target["comments"].append(public_comment)
        return public_comment

    headers = {**supabase_headers("return=representation"), "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_COMMUNITY_COMMENTS_TABLE}",
            json=comment,
            headers=headers,
        )
    response.raise_for_status()
    row = response.json()[0]
    return {
        "id": row["id"],
        "author": row.get("author") or "Eco Warrior",
        "body": row["body"],
        "created_at": row["created_at"],
        "replies": [],
    }


async def toggle_community_like(post_id: str, liked: bool = True) -> dict[str, Any]:
    delta = 1 if liked else -1

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        target = next((post for post in memory_community_posts if post["id"] == post_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Postingan tidak ditemukan.")
        target["likes"] = max(0, int(target.get("likes") or 0) + delta)
        return {"likes": target["likes"]}

    async with httpx.AsyncClient(timeout=10) as client:
        get_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_COMMUNITY_POSTS_TABLE}",
            params={"select": "likes", "id": f"eq.{post_id}", "limit": "1"},
            headers=supabase_headers(),
        )
        get_response.raise_for_status()
        rows = get_response.json()
        if not rows:
            raise HTTPException(status_code=404, detail="Postingan tidak ditemukan.")

        likes = max(0, int(rows[0].get("likes") or 0) + delta)
        patch_response = await client.patch(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_COMMUNITY_POSTS_TABLE}",
            params={"id": f"eq.{post_id}"},
            json={"likes": likes},
            headers={**supabase_headers("return=minimal"), "Content-Type": "application/json"},
        )
    patch_response.raise_for_status()
    return {"likes": likes}


async def fetch_leaderboard() -> list[dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        users_by_id = {user["id"]: user for user in memory_users.values()}
        scans_by_user: dict[str, list[dict[str, Any]]] = {}
        for item in memory_history:
            user_id = item.get("user_id")
            if user_id:
                scans_by_user.setdefault(user_id, []).append(item)

        rows = []
        for user_id, scans in scans_by_user.items():
            user = users_by_id.get(user_id, {})
            stats = build_stats_from_rows(scans)
            rows.append({
                "name": user.get("name") or "Eco Warrior",
                "scans": stats["total_scans"],
                "points": stats["points"],
            })

        rows.sort(key=lambda item: item["points"], reverse=True)
        return [{"rank": index + 1, **item} for index, item in enumerate(rows)]

    async with httpx.AsyncClient(timeout=10) as client:
        history_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_HISTORY_TABLE}",
            params={
                "select": "user_id,created_at",
                "user_id": "not.is.null",
                "order": "created_at.desc",
                "limit": "10000",
            },
            headers=supabase_headers(),
        )
        history_response.raise_for_status()
        history_rows = history_response.json()

        users_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_USERS_TABLE}",
            params={"select": "id,name"},
            headers=supabase_headers(),
        )
        users_response.raise_for_status()
        users_by_id = {row["id"]: row for row in users_response.json()}

    scans_by_user: dict[str, list[dict[str, Any]]] = {}
    for row in history_rows:
        user_id = row.get("user_id")
        if user_id:
            scans_by_user.setdefault(user_id, []).append(row)

    items = []
    for user_id, scans in scans_by_user.items():
        stats = build_stats_from_rows(scans)
        items.append({
            "name": users_by_id.get(user_id, {}).get("name") or "Eco Warrior",
            "scans": stats["total_scans"],
            "points": stats["points"],
        })

    items.sort(key=lambda item: item["points"], reverse=True)
    return [{"rank": index + 1, **item} for index, item in enumerate(items[:20])]


async def fetch_weekly_challenge() -> dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return memory_challenge

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_CHALLENGES_TABLE}",
            params={"select": "*", "order": "created_at.desc", "limit": "1"},
            headers=supabase_headers(),
        )
    response.raise_for_status()
    rows = response.json()
    return rows[0] if rows else memory_challenge


async def fetch_trivia_items() -> list[dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return memory_trivia

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TRIVIA_TABLE}",
            params={"select": "*", "order": "created_at.desc", "limit": "20"},
            headers=supabase_headers(),
        )
    response.raise_for_status()
    return response.json() or memory_trivia


async def update_profile(user_id: str, payload: ProfileUpdate) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if payload.name is not None:
        updates["name"] = payload.name.strip()
    if payload.email is not None:
        updates["email"] = normalize_email(payload.email)
    if payload.avatar_url is not None:
        updates["avatar_url"] = payload.avatar_url.strip()

    if not updates:
        raise HTTPException(status_code=422, detail="Tidak ada data profil yang diperbarui.")

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        user = next((item for item in memory_users.values() if item["id"] == user_id), None)
        if not user:
            return {"id": user_id, **updates}
        old_email = user["email"]
        user.update(updates)
        if user["email"] != old_email:
            memory_users.pop(old_email, None)
            memory_users[user["email"]] = user
        return public_user(user)

    headers = {**supabase_headers("return=representation"), "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.patch(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_USERS_TABLE}",
            params={"id": f"eq.{user_id}"},
            json=updates,
            headers=headers,
        )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan.")
    return public_user(rows[0])


async def update_password(user_id: str, payload: PasswordUpdate) -> None:
    user = await fetch_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan.")

    if not verify_password(payload.current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Password lama tidak sesuai.")

    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=422, detail="Password baru harus berbeda dari password lama.")

    password_hash = hash_password(payload.new_password)
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        user["password_hash"] = password_hash
        return

    headers = {**supabase_headers("return=minimal"), "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.patch(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_USERS_TABLE}",
            params={"id": f"eq.{user_id}"},
            json={"password_hash": password_hash},
            headers=headers,
        )
    response.raise_for_status()


async def classify_upload(file: UploadFile, user_id: str | None = None) -> dict[str, Any]:
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
        user_id=user_id,
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
async def classify(file: UploadFile = File(...), user_id: str | None = Form(default=None)) -> dict[str, Any]:
    return await classify_upload(file, user_id=user_id)


@app.post("/predict")
async def predict_compat(file: UploadFile = File(...), user_id: str | None = Form(default=None)) -> dict[str, Any]:
    return await classify_upload(file, user_id=user_id)


@app.get(f"{API_PREFIX}/history")
async def history(limit: int = 20, user_id: str | None = None) -> dict[str, Any]:
    safe_limit = min(max(limit, 1), 100)
    try:
        items = await fetch_history(safe_limit, user_id=user_id)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil riwayat dari Supabase: {exc}") from exc
    return {"items": items}


@app.get(f"{API_PREFIX}/users/{{user_id}}/stats")
async def user_stats(user_id: str) -> dict[str, Any]:
    try:
        stats = await fetch_user_stats(user_id)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil statistik pengguna: {exc}") from exc
    return {"stats": stats}


@app.get(f"{API_PREFIX}/community/challenge")
async def community_challenge() -> dict[str, Any]:
    try:
        challenge = await fetch_weekly_challenge()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil tantangan mingguan: {exc}") from exc
    return {"challenge": challenge}


@app.get(f"{API_PREFIX}/community/posts")
async def community_posts(search: str | None = None) -> dict[str, Any]:
    try:
        items = await fetch_community_posts(search)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil feed komunitas: {exc}") from exc
    return {"items": items}


@app.post(f"{API_PREFIX}/community/posts")
async def community_post_create(payload: CommunityPostCreate) -> dict[str, Any]:
    try:
        item = await create_community_post(payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal menyimpan postingan: {exc}") from exc
    return {"item": item}


@app.post(f"{API_PREFIX}/community/posts/{{post_id}}/comments")
async def community_comment_create(post_id: str, payload: CommunityCommentCreate) -> dict[str, Any]:
    try:
        item = await add_community_comment(post_id, payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal menyimpan komentar: {exc}") from exc
    return {"item": item}


@app.post(f"{API_PREFIX}/community/posts/{{post_id}}/like")
async def community_like(post_id: str, payload: CommunityLikeUpdate) -> dict[str, Any]:
    try:
        return await toggle_community_like(post_id, payload.liked)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal memperbarui like: {exc}") from exc


@app.get(f"{API_PREFIX}/community/leaderboard")
async def community_leaderboard() -> dict[str, Any]:
    try:
        items = await fetch_leaderboard()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil leaderboard: {exc}") from exc
    return {"items": items}


@app.get(f"{API_PREFIX}/trivia")
async def trivia() -> dict[str, Any]:
    try:
        items = await fetch_trivia_items()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil eco trivia: {exc}") from exc
    return {"items": items}


@app.put(f"{API_PREFIX}/users/{{user_id}}")
async def user_update(user_id: str, payload: ProfileUpdate) -> dict[str, Any]:
    try:
        user = await update_profile(user_id, payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal memperbarui profil: {exc}") from exc
    return {"user": user}


@app.put(f"{API_PREFIX}/users/{{user_id}}/password")
async def user_password_update(user_id: str, payload: PasswordUpdate) -> dict[str, str]:
    try:
        await update_password(user_id, payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Gagal memperbarui password: {exc}") from exc
    return {"message": "Password berhasil diperbarui."}
