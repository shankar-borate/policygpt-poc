"""Factory that instantiates the correct cache backend from config."""

from __future__ import annotations

import logging

from policygpt.cache.base import CacheBackend
from policygpt.cache.backends.inmemory import InMemoryBackend
from policygpt.cache.backends.redis import RedisBackend

logger = logging.getLogger(__name__)


def build_cache_backend(
    provider: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str = "",
    redis_ssl: bool = False,
    redis_ssl_ca_certs: str = "",
    redis_key_prefix: str = "pgpt:",
) -> CacheBackend:
    """Instantiate the correct backend from config.

    provider: "inmemory" | "redis"
    """
    if provider == "redis":
        logger.info("Cache backend: Redis at %s:%s db=%s", redis_host, redis_port, redis_db)
        return RedisBackend(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password or None,
            ssl=redis_ssl,
            ssl_ca_certs=redis_ssl_ca_certs or None,
            key_prefix=redis_key_prefix,
        )
    if provider == "inmemory":
        logger.info("Cache backend: InMemory")
        return InMemoryBackend()
    raise ValueError(
        f"Unknown cache_provider '{provider}'. Valid values: 'inmemory', 'redis'."
    )
