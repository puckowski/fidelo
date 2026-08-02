from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path
import sys
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings, validate_settings
from .events import LocalJobEventHub, job_event_channel
from .jobs import JobBackend, LocalJobBackend, get_job_backend
from .schemas import GenerationJob, GenerationRequest


def _is_expected_windows_disconnect(context: dict[str, object]) -> bool:
    exception = context.get("exception")
    error_code = getattr(exception, "winerror", None) or getattr(exception, "errno", None)
    return (
        sys.platform == "win32"
        and isinstance(exception, ConnectionResetError)
        and error_code == 10054
        and "_ProactorBasePipeTransport._call_connection_lost" in repr(context.get("handle"))
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    validate_settings(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        Path(active_settings.output_dir).mkdir(parents=True, exist_ok=True)
        app.state.settings = active_settings
        app.state.jobs = get_job_backend(active_settings)
        app.state.job_events = LocalJobEventHub()
        loop = asyncio.get_running_loop()
        previous_exception_handler = loop.get_exception_handler()

        def handle_loop_exception(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
            if _is_expected_windows_disconnect(context):
                return
            if previous_exception_handler is not None:
                previous_exception_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(handle_loop_exception)
        if isinstance(app.state.jobs, LocalJobBackend):
            app.state.jobs.on_update = app.state.job_events.publish
        try:
            yield
        finally:
            try:
                await asyncio.to_thread(app.state.jobs.shutdown)
            finally:
                loop.set_exception_handler(previous_exception_handler)

    app = FastAPI(
        title="Fidelo Music API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "mode": active_settings.mode,
            "queue": "in-process" if active_settings.is_local else active_settings.queue_name,
            "storage": "filesystem" if active_settings.is_local else "s3",
        }

    @app.get("/api/config")
    def public_config() -> dict[str, object]:
        return {
            "mode": active_settings.mode,
            "max_duration_seconds": active_settings.max_duration_seconds,
        }

    @app.post("/api/jobs", response_model=GenerationJob, status_code=202)
    def create_job(request: GenerationRequest) -> GenerationJob:
        if request.duration_seconds > active_settings.max_duration_seconds:
            raise HTTPException(
                status_code=422,
                detail=f"Duration cannot exceed {active_settings.max_duration_seconds} seconds",
            )
        backend: JobBackend = app.state.jobs
        return backend.enqueue(request.with_seed())

    @app.get("/api/jobs", response_model=list[GenerationJob])
    def list_jobs(limit: int = Query(default=30, ge=1, le=100)) -> list[GenerationJob]:
        return app.state.jobs.list(limit)

    @app.get("/api/jobs/{job_id}", response_model=GenerationJob)
    def get_job(job_id: str) -> GenerationJob:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.websocket("/api/jobs/{job_id}/events")
    async def watch_job(job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        backend: JobBackend = app.state.jobs
        try:
            if active_settings.is_local:
                async with app.state.job_events.subscribe(job_id) as updates:
                    job = backend.get(job_id)
                    if job is None:
                        await websocket.close(code=1008)
                        return
                    await websocket.send_json(job.model_dump(mode="json"))
                    while job.status.value not in {"completed", "failed"}:
                        await updates.get()
                        job = backend.get(job_id)
                        if job is None:
                            break
                        await websocket.send_json(job.model_dump(mode="json"))
            else:
                pubsub = backend.connection.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(job_event_channel(active_settings.queue_name))
                try:
                    job = backend.get(job_id)
                    if job is None:
                        await websocket.close(code=1008)
                        return
                    await websocket.send_json(job.model_dump(mode="json"))
                    while job.status.value not in {"completed", "failed"}:
                        message = await asyncio.to_thread(pubsub.get_message, timeout=30.0)
                        if message is None or message["type"] != "message":
                            continue
                        event = json.loads(message["data"])
                        if event.get("id") != job_id:
                            continue
                        await websocket.send_json(event)
                        job = job.model_copy(update=event)
                finally:
                    pubsub.close()
        except WebSocketDisconnect:
            return
        finally:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close()

    @app.get("/api/files/{object_key:path}")
    def get_file(object_key: str, download: bool = False):
        if active_settings.is_local:
            safe_name = Path(object_key).name
            path = Path(active_settings.output_dir).resolve() / safe_name
            if not path.is_file():
                raise HTTPException(status_code=404, detail="Audio file not found")
            return FileResponse(
                path,
                media_type="audio/wav",
                filename=safe_name if download else None,
                content_disposition_type="attachment" if download else "inline",
            )

        try:
            import boto3
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="S3 support is not installed") from exc
        options = {"region_name": active_settings.s3_region} if active_settings.s3_region else {}
        if active_settings.s3_endpoint_url:
            options["endpoint_url"] = active_settings.s3_endpoint_url
        client = boto3.client("s3", **options)
        params = {"Bucket": active_settings.s3_bucket, "Key": object_key}
        if download:
            params["ResponseContentDisposition"] = f'attachment; filename="{Path(object_key).name}"'
        url = client.generate_presigned_url("get_object", Params=params, ExpiresIn=3600)
        return RedirectResponse(url)

    static_dir = PROJECT_ROOT / "web"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


app = create_app()
