"""Redis-backed cache backend.

Requires: pip install redis

Values are pickled before storage so any Python object (numpy arrays,
dataclasses, lists) round-trips correctly — matching InMemoryBackend
behaviour exactly.

clear_prefix uses cursor-based SCAN (not KEYS) so it is safe on large keyspaces.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

from policygpt.cache.base import CacheBackend

logger = logging.getLogger(__name__)


class RedisBackend(CacheBackend):
    """Redis-backed cache. Drop-in replacement for InMemoryBackend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        ssl: bool = False,
        ssl_ca_certs: str | None = None,
        key_prefix: str = "pgpt:",
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
    ) -> None:
        try:
            import redis as _redis
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisBackend. "
                "Install it with: pip install redis"
            ) from exc

        self._key_prefix = key_prefix
        self._client = _redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password or None,
            ssl=ssl,
            ssl_ca_certs=ssl_ca_certs or None,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False,  # binary — we handle pickle ourselves
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _k(self, key: str) -> str:
        return self._key_prefix + key

    @staticmethod
    def _serialize(value: Any) -> bytes:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize(raw: bytes) -> Any:
        return pickle.loads(raw)  # noqa: S301 — values we wrote ourselves

    # ── CacheBackend interface ────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(self._k(key))
            if raw is None:
                return None
            return self._deserialize(raw)
        except Exception as exc:
            logger.warning("RedisBackend.get [%s] failed: %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            self._client.setex(self._k(key), ttl, self._serialize(value))
        except Exception as exc:
            logger.warning("RedisBackend.set [%s] failed: %s", key, exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._k(key))
        except Exception as exc:
            logger.warning("RedisBackend.delete [%s] failed: %s", key, exc)

    def clear_prefix(self, prefix: str) -> None:
        pattern = self._k(prefix) + "*"
        cursor = 0
        try:
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("RedisBackend.clear_prefix [%s] failed: %s", prefix, exc)

    def size(self) -> int | str:
        try:
            return self._client.dbsize()
        except Exception:
            return "n/a"

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False
