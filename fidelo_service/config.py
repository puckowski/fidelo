from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    mode: str = "local"
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "fidelo-gpu"
    output_dir: str = str(PROJECT_ROOT / "generated")
    generator_script: str = str(
        PROJECT_ROOT
        / "generate_latent_audio_cuda_source_creative_ranked_energy_gate_thematic_sticky_intro_body_latent_guided_tokens.py"
    )
    tokenizer_dir: str = str(PROJECT_ROOT / "3.33smodel" / "9smodel")
    prior_dir: str = str(PROJECT_ROOT / "3.33smodel" / "9smodel" / "prior_transition_finetuned")
    local_workers: int = 1
    max_duration_seconds: int = 180
    job_timeout_seconds: int = 3600
    s3_bucket: str = ""
    s3_prefix: str = "generations"
    s3_region: str = ""
    s3_endpoint_url: str = ""
    keep_local_outputs: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            mode=os.getenv("FIDELO_MODE", "local").strip().lower(),
            redis_url=os.getenv("FIDELO_REDIS_URL", "redis://localhost:6379/0"),
            queue_name=os.getenv("FIDELO_QUEUE_NAME", "fidelo-gpu"),
            output_dir=os.getenv("FIDELO_OUTPUT_DIR", str(PROJECT_ROOT / "generated")),
            generator_script=os.getenv("FIDELO_GENERATOR_SCRIPT", cls.generator_script),
            tokenizer_dir=os.getenv("FIDELO_TOKENIZER_DIR", cls.tokenizer_dir),
            prior_dir=os.getenv("FIDELO_PRIOR_DIR", cls.prior_dir),
            local_workers=max(1, int(os.getenv("FIDELO_LOCAL_WORKERS", "1"))),
            max_duration_seconds=max(1, int(os.getenv("FIDELO_MAX_DURATION_SECONDS", "180"))),
            job_timeout_seconds=max(60, int(os.getenv("FIDELO_JOB_TIMEOUT_SECONDS", "3600"))),
            s3_bucket=os.getenv("FIDELO_S3_BUCKET", ""),
            s3_prefix=os.getenv("FIDELO_S3_PREFIX", "generations").strip("/"),
            s3_region=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "")),
            s3_endpoint_url=os.getenv("FIDELO_S3_ENDPOINT_URL", ""),
            keep_local_outputs=_env_bool("FIDELO_KEEP_LOCAL_OUTPUTS", True),
        )

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    def as_task_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_settings(settings: Settings) -> None:
    if settings.mode not in {"local", "production"}:
        raise ValueError("FIDELO_MODE must be 'local' or 'production'")
    if not settings.is_local and not settings.s3_bucket:
        raise ValueError("FIDELO_S3_BUCKET is required in production mode")
