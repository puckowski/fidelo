from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=240)
    duration_seconds: int = Field(default=30, ge=5, le=180)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    source_strength: float = Field(default=0.90, ge=0.0, le=1.0)
    temperature: float = Field(default=0.60, ge=0.05, le=2.0)
    top_k: int = Field(default=4, ge=1, le=64)
    top_p: float = Field(default=0.85, ge=0.05, le=1.0)
    creativity: float = Field(default=0.04, ge=0.0, le=0.5)
    theme_seconds: float = Field(default=4.0, ge=1.0, le=20.0)
    transition_seconds: float = Field(default=5.0, ge=0.0, le=20.0)

    def with_seed(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["seed"] = self.seed if self.seed is not None else uuid4().int % 2_147_483_647
        return payload


class GenerationJob(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    duration_seconds: int
    seed: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    audio_url: str | None = None
    download_url: str | None = None
    error: str | None = None
    logs: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
