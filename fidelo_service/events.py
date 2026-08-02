from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import json
from threading import Lock
from typing import AsyncIterator


def job_event_channel(queue_name: str) -> str:
    return f"fidelo:job-events:{queue_name}"


class LocalJobEventHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[None]]]] = defaultdict(set)
        self._lock = Lock()

    @asynccontextmanager
    async def subscribe(self, job_id: str) -> AsyncIterator[asyncio.Queue[None]]:
        subscriber = (asyncio.get_running_loop(), asyncio.Queue())
        with self._lock:
            self._subscribers[job_id].add(subscriber)
        try:
            yield subscriber[1]
        finally:
            with self._lock:
                subscribers = self._subscribers.get(job_id)
                if subscribers is not None:
                    subscribers.discard(subscriber)
                    if not subscribers:
                        del self._subscribers[job_id]

    def publish(self, job_id: str) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers.get(job_id, ()))
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, None)


def publish_redis_job_event(redis_url: str, queue_name: str, job: dict[str, object]) -> None:
    try:
        from redis import Redis

        Redis.from_url(redis_url).publish(job_event_channel(queue_name), json.dumps(job))
    except Exception:
        # Notification delivery must not turn a successfully generated track into a failed job.
        pass