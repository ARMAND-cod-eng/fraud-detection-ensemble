"""In-process pub/sub for server-sent events.

Each subscriber gets a bounded queue. A slow consumer never blocks scoring: when its queue is
full the oldest event is dropped and a drop counter is incremented (reported to the client).
Multi-instance deployments would replace this with Redis pub/sub (see ENGINEERING_TRADEOFFS).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any


class Subscriber:
    def __init__(self, maxsize: int) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0


class Broadcaster:
    def __init__(self, maxsize: int = 500) -> None:
        self._subs: set[Subscriber] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.maxsize = maxsize

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> Subscriber:
        sub = Subscriber(self.maxsize)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def _put(self, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if sub.queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    sub.queue.get_nowait()
                sub.dropped += 1
            sub.queue.put_nowait(event)

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe: may be called from worker threads (scoring runs in a thread pool)."""
        if not self._subs or self._loop is None or self._loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._put(event)
        else:
            self._loop.call_soon_threadsafe(self._put, event)
