"""Cache backend configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheConfig:
    # "inmemory" — process-local, no external dependency (default).
    # "redis"    — shared across all app server instances.
    cache_provider: str = "inmemory"

    # Redis connection — only used when cache_provider = "redis".
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Redis auth password. None = no authentication.
    redis_password: str | None = None

    redis_ssl: bool = False

    # Path to CA certificate bundle for Redis TLS. None = use system default.
    redis_ssl_ca_certs: str | None = None

    redis_key_prefix: str = "pgpt:"
