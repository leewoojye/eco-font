from __future__ import annotations

import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .font_generation_runner import UploadedFontGenerationRunner
from .schemas import FontGenerationJobStatus, FontGenerationSpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_spec(spec: FontGenerationSpec) -> dict[str, Any]:
    if hasattr(spec, "model_dump"):
        return spec.model_dump()
    return spec.dict()


@dataclass
class FontGenerationRequest:
    spec: FontGenerationSpec
    input_font: Path
    original_filename: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "spec": _dump_spec(self.spec),
            "input_filename": self.original_filename,
        }


@dataclass
class FontGenerationJobRecord:
    job_id: str
    request: FontGenerationRequest
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    future: Future | None = None

    def to_status(self) -> FontGenerationJobStatus:
        return FontGenerationJobStatus(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error=self.error,
            request=self.request.to_public_dict(),
            result_url=f"/v1/font-generation/jobs/{self.job_id}/result" if self.status == "completed" else None,
        )


class FontGenerationJobStore:
    def __init__(self, runner: UploadedFontGenerationRunner | None = None, max_workers: int = 1) -> None:
        self.runner = runner or UploadedFontGenerationRunner()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eco-font-ttf-job")
        self.records: dict[str, FontGenerationJobRecord] = {}
        self.lock = Lock()

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def submit(self, job_id: str, request: FontGenerationRequest) -> FontGenerationJobRecord:
        record = FontGenerationJobRecord(job_id=job_id, request=request)
        with self.lock:
            self.records[job_id] = record
        record.future = self.executor.submit(self._run_record, job_id)
        return record

    def run_sync(self, job_id: str, request: FontGenerationRequest) -> FontGenerationJobRecord:
        record = FontGenerationJobRecord(job_id=job_id, request=request)
        with self.lock:
            self.records[job_id] = record
        self._run_record(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> FontGenerationJobRecord:
        with self.lock:
            record = self.records.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def result(self, job_id: str) -> dict[str, Any]:
        record = self.get(job_id)
        if record.status != "completed" or record.result is None:
            raise RuntimeError(f"font generation job '{job_id}' is {record.status}")
        return record.result

    def _run_record(self, job_id: str) -> None:
        record = self.get(job_id)
        with self.lock:
            record.status = "running"
            record.started_at = _now()
            record.error = None
        try:
            result = self.runner.run(
                job_id=job_id,
                spec=record.request.spec,
                input_font=record.request.input_font,
                original_filename=record.request.original_filename,
            )
        except Exception as exc:
            with self.lock:
                record.status = "failed"
                record.completed_at = _now()
                record.error = f"{exc}\n{traceback.format_exc()}"
            return
        with self.lock:
            record.status = "completed"
            record.completed_at = _now()
            record.result = result


font_generation_job_store = FontGenerationJobStore()

