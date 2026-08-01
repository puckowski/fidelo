from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import Settings


class ObjectStore(ABC):
    @abstractmethod
    def publish(self, source: Path, object_key: str) -> dict[str, str]:
        raise NotImplementedError


class LocalObjectStore(ObjectStore):
    def publish(self, source: Path, object_key: str) -> dict[str, str]:
        return {"object_key": object_key, "storage": "local"}


class S3ObjectStore(ObjectStore):
    def __init__(self, settings: Settings):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install the production dependencies to use S3 storage") from exc

        options: dict[str, Any] = {}
        if settings.s3_region:
            options["region_name"] = settings.s3_region
        if settings.s3_endpoint_url:
            options["endpoint_url"] = settings.s3_endpoint_url
        self.client = boto3.client("s3", **options)
        self.bucket = settings.s3_bucket

    def publish(self, source: Path, object_key: str) -> dict[str, str]:
        self.client.upload_file(
            str(source),
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": "audio/wav"},
        )
        return {"object_key": object_key, "storage": "s3"}


def get_object_store(settings: Settings) -> ObjectStore:
    return LocalObjectStore() if settings.is_local else S3ObjectStore(settings)
