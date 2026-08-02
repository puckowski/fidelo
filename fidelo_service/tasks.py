from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from typing import Any

from .config import Settings
from .events import publish_redis_job_event
from .storage import get_object_store


def build_generator_command(payload: dict[str, Any], settings: Settings, output_path: Path) -> list[str]:
    return [
        sys.executable,
        settings.generator_script,
        "--tokenizer-dir",
        settings.tokenizer_dir,
        "--prior-dir",
        settings.prior_dir,
        "--prompt",
        payload["prompt"],
        "--duration-seconds",
        str(payload["duration_seconds"]),
        "--seed",
        str(payload["seed"]),
        "--source-strength",
        str(payload["source_strength"]),
        "--temperature",
        str(payload["temperature"]),
        "--top-k",
        str(payload["top_k"]),
        "--top-p",
        str(payload["top_p"]),
        "--creative-token-mix",
        str(payload["creativity"]),
        "--theme-seconds",
        str(payload["theme_seconds"]),
        "--intro-theme-seconds",
        str(payload["theme_seconds"]),
        "--latent-transition-seconds",
        str(payload["transition_seconds"]),
        "--output",
        str(output_path),
    ]


def run_generation(
    job_id: str,
    payload: dict[str, Any],
    settings_data: dict[str, Any],
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    settings = Settings(**settings_data)
    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_id}.wav"
    output_path = output_dir / filename
    command = build_generator_command(payload, settings, output_path)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")

    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(settings.generator_script).resolve().parent),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            remaining_seconds = settings.job_timeout_seconds
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    raise RuntimeError("Generation cancelled during server shutdown")
                try:
                    stdout, stderr = process.communicate(timeout=min(0.25, remaining_seconds))
                    break
                except subprocess.TimeoutExpired:
                    remaining_seconds -= 0.25
                    if remaining_seconds <= 0:
                        process.kill()
                        process.communicate()
                        raise RuntimeError(f"Generation exceeded {settings.job_timeout_seconds} seconds")
        except OSError as exc:
            raise RuntimeError(f"Could not start generator: {exc}") from exc

        logs = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
        if process.returncode != 0:
            tail = logs[-4000:] if logs else "No model output was captured"
            raise RuntimeError(f"Generator exited with code {process.returncode}:\n{tail}")
        if not output_path.is_file():
            raise RuntimeError("Generator completed without creating an output file")

        object_key = f"{settings.s3_prefix}/{filename}" if not settings.is_local else filename
        published = get_object_store(settings).publish(output_path, object_key)
        if not settings.is_local and not settings.keep_local_outputs:
            output_path.unlink(missing_ok=True)
        result = {**published, "logs": logs[-12000:]}
    except Exception as exc:
        if not settings.is_local:
            publish_redis_job_event(
                settings.redis_url,
                settings.queue_name,
                {"id": job_id, "status": "failed", "error": str(exc)[-4000:]},
            )
        raise

    if not settings.is_local:
        url = f"/api/files/{result['object_key']}"
        publish_redis_job_event(
            settings.redis_url,
            settings.queue_name,
            {
                "id": job_id,
                "status": "completed",
                "audio_url": url,
                "download_url": f"{url}?download=1",
                "logs": result["logs"],
            },
        )
    return result
