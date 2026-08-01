from __future__ import annotations

from redis import Redis
from rq import Worker

from .config import Settings, validate_settings


def main() -> None:
    settings = Settings.from_env()
    validate_settings(settings)
    if settings.is_local:
        raise RuntimeError("The standalone worker is only used with FIDELO_MODE=production")
    connection = Redis.from_url(settings.redis_url)
    worker = Worker([settings.queue_name], connection=connection)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()