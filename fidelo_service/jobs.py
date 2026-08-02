from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Callable
from uuid import uuid4

from .config import Settings
from .schemas import GenerationJob, JobStatus, utc_now
from .tasks import run_generation


class JobBackend(ABC):
    @abstractmethod
    def enqueue(self, payload: dict[str, Any]) -> GenerationJob:
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str) -> GenerationJob | None:
        raise NotImplementedError

    @abstractmethod
    def list(self, limit: int = 30) -> list[GenerationJob]:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError


@dataclass
class _LocalRecord:
    job: GenerationJob
    future: Future[dict[str, Any]] | None = None
    cancel_event: Event | None = None


class LocalJobBackend(JobBackend):
    def __init__(self, settings: Settings, on_update: Callable[[str], None] | None = None):
        self.settings = settings
        self.on_update = on_update
        self.executor = ThreadPoolExecutor(
            max_workers=settings.local_workers,
            thread_name_prefix="fidelo-gpu",
        )
        self.records: dict[str, _LocalRecord] = {}
        self.lock = Lock()

    def enqueue(self, payload: dict[str, Any]) -> GenerationJob:
        job_id = uuid4().hex
        job = GenerationJob(
            id=job_id,
            status=JobStatus.queued,
            prompt=payload["prompt"],
            duration_seconds=payload["duration_seconds"],
            seed=payload["seed"],
            created_at=utc_now(),
        )
        with self.lock:
            self.records[job_id] = _LocalRecord(job=job)
            cancel_event = Event()
            self.records[job_id].cancel_event = cancel_event
        future = self.executor.submit(self._execute, job_id, payload, cancel_event)
        with self.lock:
            self.records[job_id].future = future
        return job.model_copy(deep=True)

    def _execute(self, job_id: str, payload: dict[str, Any], cancel_event: Event) -> dict[str, Any]:
        self._update(job_id, status=JobStatus.running, started_at=utc_now())
        try:
            result = run_generation(job_id, payload, self.settings.as_task_dict(), cancel_event)
        except Exception as exc:
            self._update(
                job_id,
                status=JobStatus.failed,
                finished_at=utc_now(),
                error=str(exc)[-4000:],
            )
            raise
        self._update(
            job_id,
            status=JobStatus.completed,
            finished_at=utc_now(),
            logs=result.get("logs"),
        )
        return result

    def _update(self, job_id: str, **changes: Any) -> None:
        with self.lock:
            record = self.records[job_id]
            record.job = record.job.model_copy(update=changes)
        if self.on_update is not None:
            self.on_update(job_id)

    def get(self, job_id: str) -> GenerationJob | None:
        with self.lock:
            record = self.records.get(job_id)
            if record is None:
                return None
            job = record.job.model_copy(deep=True)
            future = record.future
        if job.status == JobStatus.completed and future is not None:
            result = future.result()
            job.audio_url = f"/api/files/{result['object_key']}"
            job.download_url = f"/api/files/{result['object_key']}?download=1"
        return job

    def list(self, limit: int = 30) -> list[GenerationJob]:
        with self.lock:
            ids = list(reversed(self.records.keys()))[:limit]
        return [job for job_id in ids if (job := self.get(job_id)) is not None]

    def shutdown(self) -> None:
        with self.lock:
            records = tuple(self.records.values())
        for record in records:
            if record.cancel_event is not None:
                record.cancel_event.set()
            if record.future is not None:
                record.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)


class RedisJobBackend(JobBackend):
    def __init__(self, settings: Settings):
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:
            raise RuntimeError("Install the production dependencies to use Redis/RQ") from exc
        self.settings = settings
        self.connection = Redis.from_url(settings.redis_url)
        self.queue = Queue(settings.queue_name, connection=self.connection)
        self.index_key = f"fidelo:jobs:{settings.queue_name}"

    def enqueue(self, payload: dict[str, Any]) -> GenerationJob:
        job_id = uuid4().hex
        metadata = {
            "prompt": payload["prompt"],
            "duration_seconds": payload["duration_seconds"],
            "seed": payload["seed"],
        }
        self.queue.enqueue(
            run_generation,
            job_id,
            payload,
            self.settings.as_task_dict(),
            job_id=job_id,
            job_timeout=self.settings.job_timeout_seconds,
            result_ttl=86400 * 7,
            failure_ttl=86400 * 7,
            meta=metadata,
        )
        self.connection.lpush(self.index_key, job_id)
        self.connection.ltrim(self.index_key, 0, 499)
        job = self.queue.fetch_job(job_id)
        return self._serialize(job)

    def _serialize(self, rq_job: Any) -> GenerationJob:
        rq_status = rq_job.get_status(refresh=True)
        status_key = getattr(rq_status, "value", str(rq_status))
        status_map = {
            "created": JobStatus.queued,
            "queued": JobStatus.queued,
            "deferred": JobStatus.queued,
            "scheduled": JobStatus.queued,
            "started": JobStatus.running,
            "finished": JobStatus.completed,
            "failed": JobStatus.failed,
            "stopped": JobStatus.failed,
            "canceled": JobStatus.failed,
        }
        status = status_map.get(status_key, JobStatus.queued)
        result = rq_job.result if status == JobStatus.completed else None
        object_key = result.get("object_key") if isinstance(result, dict) else None
        url = f"/api/files/{object_key}" if object_key else None
        error = rq_job.exc_info[-4000:] if rq_job.exc_info else None
        return GenerationJob(
            id=rq_job.id,
            status=status,
            prompt=rq_job.meta.get("prompt", "Unknown prompt"),
            duration_seconds=rq_job.meta.get("duration_seconds", 0),
            seed=rq_job.meta.get("seed", 0),
            created_at=rq_job.created_at or utc_now(),
            started_at=rq_job.started_at,
            finished_at=rq_job.ended_at,
            audio_url=url,
            download_url=f"{url}?download=1" if url else None,
            error=error,
            logs=result.get("logs") if isinstance(result, dict) else None,
        )

    def get(self, job_id: str) -> GenerationJob | None:
        job = self.queue.fetch_job(job_id)
        return self._serialize(job) if job is not None else None

    def list(self, limit: int = 30) -> list[GenerationJob]:
        job_ids = [value.decode() if isinstance(value, bytes) else value for value in self.connection.lrange(self.index_key, 0, limit - 1)]
        jobs = (self.queue.fetch_job(job_id) for job_id in job_ids)
        return [self._serialize(job) for job in jobs if job is not None]

    def shutdown(self) -> None:
        self.connection.close()


def get_job_backend(settings: Settings) -> JobBackend:
    return LocalJobBackend(settings) if settings.is_local else RedisJobBackend(settings)
