from __future__ import annotations

import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from .research_runner import EcoResearchRunner
from .schemas import GenerateRequest, JobStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_request(request: GenerateRequest) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return request.dict()


@dataclass
class JobRecord:
    job_id: str
    request: GenerateRequest
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    future: Future | None = None

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error=self.error,
            request=_dump_request(self.request),
            result_url=f"/v1/jobs/{self.job_id}/result" if self.status == "completed" else None,
        )


class JobStore:
    def __init__(self, runner: EcoResearchRunner | None = None, max_workers: int = 1) -> None:
        self.runner = runner or EcoResearchRunner()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eco-font-job")
        self.records: dict[str, JobRecord] = {}
        self.lock = Lock()

    def submit(self, request: GenerateRequest) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id, request=request)
        with self.lock:
            self.records[job_id] = record
        record.future = self.executor.submit(self._run_record, job_id)
        return record

    def run_sync(self, request: GenerateRequest) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id, request=request)
        with self.lock:
            self.records[job_id] = record
        self._run_record(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> JobRecord:
        with self.lock:
            record = self.records.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def result(self, job_id: str) -> dict[str, Any]:
        record = self.get(job_id)
        if record.status != "completed" or record.result is None:
            raise RuntimeError(f"job '{job_id}' is {record.status}")
        return record.result

    def _run_record(self, job_id: str) -> None:
        record = self.get(job_id)
        with self.lock:
            record.status = "running"
            record.started_at = _now()
            record.error = None
        try:
            result = self.runner.run(job_id, record.request)
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


job_store = JobStore()

