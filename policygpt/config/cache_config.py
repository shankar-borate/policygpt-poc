"""Cache backend configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheConfig:
    # "inmemory" — process-local, no external dependency (default).
    # "redis"    — shared across all app server instances.
    cache_provider: str = "inmemory"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_ssl: bool = False
    redis_ssl_ca_certs: str = ""
    redis_key_prefix: str = "pgpt:"
