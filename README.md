# EcoScan Backend

FastAPI backend untuk klasifikasi gambar sampah EcoScan, siap local development dan deploy Render.

## Local

```bash
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Jika Python lokal bukan 3.11, jalankan dengan Docker:

```bash
cd backend
docker build -t ecoscan-api .
docker run --env-file .env -p 8000:8000 ecoscan-api
```

Endpoint utama:

- `POST /api/v1/classify` dengan multipart field `file`
- `GET /api/v1/history`
- `GET /health`
- `POST /predict` kompatibel dengan frontend lama

Jika `SUPABASE_URL` dan `SUPABASE_SERVICE_ROLE_KEY` kosong, riwayat disimpan sementara di memori agar local tetap bisa dicoba.

Model yang tersedia di repo saat ini memakai 5 output (`Anorganik,B3,Kertas,Organik,Residu`). Jika model tim AI diganti ke 10 kelas Garbage Classification V2, ubah `MODEL_CLASSES` menjadi:

```env
MODEL_CLASSES=clothes,glass,plastic,shoes,cardboard,paper,metal,battery,biological,trash
```

## Supabase

Jalankan `sql/supabase_schema.sql` di SQL Editor Supabase. Di Render, isi:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `CORS_ORIGINS` dengan URL frontend production

## Render

Deploy dari `render.yaml`, atau buat Web Service manual:

- Root Directory: `backend`
- Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
