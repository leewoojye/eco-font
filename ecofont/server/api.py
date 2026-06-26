from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .font_generation_jobs import FontGenerationRequest, font_generation_job_store
from .font_generation_runner import SUPPORTED_FONT_GENERATION_METHODS
from .jobs import job_store
from .profiles import list_profile_infos
from .schemas import (
    FontGenerationEvaluation,
    FontGenerationJobAccepted,
    FontGenerationJobStatus,
    FontGenerationMethodInfo,
    FontGenerationSpec,
    GenerateRequest,
    JobAccepted,
    JobStatus,
    ProfileInfo,
)
from .script_sets import detect_font_script, font_script_counts


router = APIRouter(prefix="/v1", tags=["generation"])
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_CANDIDATE_COUNT = 20
DEFAULT_OCR_SAMPLE_SIZE = 32


def _validate_request(request: GenerateRequest) -> None:
    try:
        job_store.runner.prepare_request(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


async def _read_uploaded_font(font: UploadFile) -> bytes:
    data = await font.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="uploaded font is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="uploaded font is too large")
    return data


def _auto_cherokee_spec(input_path: str | Path) -> FontGenerationSpec:
    detected_script = detect_font_script(input_path)
    counts = font_script_counts(input_path)
    if detected_script != "cherokee":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "uploaded font is not a supported Cherokee font",
                "detected_script": detected_script,
                "script_counts": counts,
                "supported_scripts": ["cherokee"],
            },
        )
    spec = FontGenerationSpec(
        script="cherokee",
        method="eco_research_guided",
        candidate_count=DEFAULT_CANDIDATE_COUNT,
        codepoint_set="cherokee_full",
        return_format="zip",
        evaluation=FontGenerationEvaluation(
            ocr=True,
            ink=True,
            ocr_eval_mode="sample",
            eval_sample_size=DEFAULT_OCR_SAMPLE_SIZE,
            ocr_lang="chr",
            ocr_psm=[8, 6, 10],
        ),
    )
    try:
        font_generation_job_store.runner.prepare_spec(spec)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return spec


def _save_upload_and_build_request(job_id: str, font: UploadFile, data: bytes) -> FontGenerationRequest:
    try:
        input_path = font_generation_job_store.runner.save_upload(job_id, font.filename or "input.ttf", data)
        parsed = _auto_cherokee_spec(input_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return FontGenerationRequest(
        spec=parsed,
        input_font=input_path,
        original_filename=font.filename or "input.ttf",
    )


async def _run_auto_font_generation_sync(font: UploadFile) -> dict:
    data = await _read_uploaded_font(font)
    job_id = font_generation_job_store.new_job_id()
    request = _save_upload_and_build_request(job_id, font, data)
    record = font_generation_job_store.run_sync(job_id, request)
    if record.status == "failed":
        raise HTTPException(status_code=500, detail=record.error)
    if record.result is None:
        raise HTTPException(status_code=500, detail="font generation did not produce a result")
    return record.result


@router.get("/health")
def health_v1() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/profiles", response_model=list[ProfileInfo])
def profiles() -> list[ProfileInfo]:
    return list_profile_infos()


@router.get("/font-generation/methods", response_model=list[FontGenerationMethodInfo])
def font_generation_methods() -> list[dict]:
    return SUPPORTED_FONT_GENERATION_METHODS


@router.post("/font-generation/ttf")
async def create_font_generation_from_ttf(
    font: UploadFile = File(...),
) -> dict:
    return await _run_auto_font_generation_sync(font)


@router.post(
    "/font-generation/jobs",
    response_model=FontGenerationJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_font_generation_job(
    font: UploadFile = File(...),
) -> FontGenerationJobAccepted:
    data = await _read_uploaded_font(font)
    job_id = font_generation_job_store.new_job_id()
    request = _save_upload_and_build_request(job_id, font, data)
    record = font_generation_job_store.submit(job_id, request)
    return FontGenerationJobAccepted(
        job_id=record.job_id,
        status=record.status,
        status_url=f"/v1/font-generation/jobs/{record.job_id}",
        result_url=f"/v1/font-generation/jobs/{record.job_id}/result",
    )


@router.post("/font-generation/jobs-sync")
async def create_font_generation_job_sync(
    font: UploadFile = File(...),
) -> dict:
    return await _run_auto_font_generation_sync(font)


@router.get("/font-generation/jobs/{job_id}", response_model=FontGenerationJobStatus)
def font_generation_job_status(job_id: str) -> FontGenerationJobStatus:
    try:
        return font_generation_job_store.get(job_id).to_status()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown font generation job_id: {job_id}") from exc


@router.get("/font-generation/jobs/{job_id}/result")
def font_generation_job_result(job_id: str) -> dict:
    try:
        return font_generation_job_store.result(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown font generation job_id: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/generate", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def generate(request: GenerateRequest) -> JobAccepted:
    _validate_request(request)
    record = job_store.submit(request)
    return JobAccepted(
        job_id=record.job_id,
        status=record.status,
        status_url=f"/v1/jobs/{record.job_id}",
        result_url=f"/v1/jobs/{record.job_id}/result",
    )


@router.post("/generate-sync")
def generate_sync(request: GenerateRequest) -> dict:
    _validate_request(request)
    record = job_store.run_sync(request)
    if record.status == "failed":
        raise HTTPException(status_code=500, detail=record.error)
    if record.result is None:
        raise HTTPException(status_code=500, detail="generation did not produce a result")
    return record.result


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    try:
        return job_store.get(job_id).to_status()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}") from exc


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> dict:
    try:
        return job_store.result(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/assets/{job_id}/{asset_path:path}")
def asset(job_id: str, asset_path: str) -> FileResponse:
    try:
        path = job_store.runner.resolve_asset(job_id, asset_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_path}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path)
