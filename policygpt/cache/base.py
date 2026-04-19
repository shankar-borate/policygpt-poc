"""Abstract cache backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """Provider-agnostic key/value cache interface.

    All backends must implement get / set / delete / clear_prefix.
    TTL is always expressed in seconds.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing / expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store value with a TTL in seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a single key (no-op if absent)."""

    @abstractmethod
    def clear_prefix(self, prefix: str) -> None:
        """Remove all keys that start with prefix."""

    def size(self) -> int | str:
        """Return approximate number of keys. Override in concrete backends."""
        return "n/a"

    def ping(self) -> bool:
        """Health check. Returns True if the backend is reachable."""
        return True
