"""In-memory cache backend with TTL eviction on read."""

from __future__ import annotations

import time
from typing import Any

from policygpt.cache.base import CacheBackend


class InMemoryBackend(CacheBackend):
    """Process-local cache with TTL eviction on read.

    Reads and writes to Python dicts are GIL-protected for single-key
    operations — sufficient for CPython under typical web workloads.
    For shared state across multiple app servers use RedisBackend.
    """

    def __init__(self) -> None:
        # { key: (value, expires_at) }  — expires_at is a monotonic float
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear_prefix(self, prefix: str) -> None:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(expired)

    def size(self) -> int:
        return len(self._store)
