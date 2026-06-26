"""Minimal FastAPI wrapper for rule/model inference."""

from __future__ import annotations

import shutil
import uuid
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .infer import infer_font
from .server.api import router as generation_router

APP_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = APP_ROOT / "uploads"
OUTPUT_DIR = APP_ROOT / "outputs" / "api"

app = FastAPI(title="EcoFont AI Lab", version="0.1.0")


def _cors_origins() -> list[str]:
    raw = os.environ.get("ECOFONT_API_CORS_ORIGINS", "*")
    return [item.strip() for item in raw.split(",") if item.strip()]


_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(generation_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/fonts/convert")
def convert_font(
    ttf: UploadFile = File(...),
    language: str = Form("ko"),
    target_saving: float = Form(0.25),
    preview_text: str = Form(""),
    method: str = Form("rules"),
    checkpoint: str = Form(""),
) -> JSONResponse:
    """Synchronous MVP conversion endpoint."""
    job_id = uuid.uuid4().hex
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(ttf.filename or "font.ttf").suffix or ".ttf"
    font_path = job_upload_dir / f"input{suffix}"
    with font_path.open("wb") as handle:
        shutil.copyfileobj(ttf.file, handle)

    summary = infer_font(
        font_path=font_path,
        output=job_output_dir,
        checkpoint=checkpoint or None,
        method=method,
        language=language,
        text=preview_text or None,
        target_saving=target_saving,
    )
    return JSONResponse({"jobId": job_id, "status": "completed", "result": summary})
