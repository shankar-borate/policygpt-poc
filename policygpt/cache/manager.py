"""CacheManager — typed cache interface for PolicyGPT.

Wraps a CacheBackend with strongly-typed methods per cache bucket.
Switching to Redis later = pass RedisBackend() to the constructor.

Cache buckets
─────────────
answer          (normalized_question, active_doc_ids) → (answer_text, sources)
embedding       text → numpy vector
acl             user_id → list[doc_id]  (None = admin / unrestricted)
profile         user_id → UserProfile
doc_meta        doc_id  → {title, audiences, document_type}
entity_expansion hash(focus_terms) → list[expanded_term]

Default TTLs are conservative — override per-call when needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from policygpt.cache.base import CacheBackend
from policygpt.cache.backends.inmemory import InMemoryBackend

if TYPE_CHECKING:
    from policygpt.config.user_profiles import UserProfile
    from policygpt.models import SourceReference

logger = logging.getLogger(__name__)

# ── Default TTLs (seconds) ────────────────────────────────────────────────────
_TTL_ANSWER     = 3_600       # 1 hour
_TTL_EMBEDDING  = 86_400      # 24 hours
_TTL_ACL        = 900         # 15 minutes
_TTL_PROFILE    = 1_800       # 30 minutes
_TTL_DOC_META   = 3_600       # 1 hour
_TTL_ENTITY_EXP = 21_600      # 6 hours

# ── Namespace prefixes ────────────────────────────────────────────────────────
_NS_ANSWER     = "ans:"
_NS_EMBEDDING  = "emb:"
_NS_ACL        = "acl:"
_NS_PROFILE    = "prof:"
_NS_DOC_META   = "doc:"
_NS_ENTITY_EXP = "ent:"


def _hash(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class CacheManager:
    """Typed cache layer. Backend is swappable (InMemory → Redis)."""

    def __init__(self, backend: CacheBackend | None = None) -> None:
        self._b: CacheBackend = backend or InMemoryBackend()

    # ── Answer cache ──────────────────────────────────────────────────────────

    def get_answer(
        self,
        normalized_question: str,
        active_doc_ids: frozenset[str],
    ) -> tuple[str, list["SourceReference"]] | None:
        key = _NS_ANSWER + _hash(normalized_question, sorted(active_doc_ids))
        return self._b.get(key)

    def set_answer(
        self,
        normalized_question: str,
        active_doc_ids: frozenset[str],
        answer: str,
        sources: list["SourceReference"],
        ttl: int = _TTL_ANSWER,
    ) -> None:
        key = _NS_ANSWER + _hash(normalized_question, sorted(active_doc_ids))
        self._b.set(key, (answer, sources), ttl)

    # ── Embedding cache ───────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> np.ndarray | None:
        key = _NS_EMBEDDING + _hash(text)
        value = self._b.get(key)
        if value is None:
            return None
        # Stored as list for JSON-safe serialization; convert back to ndarray.
        return np.array(value, dtype=np.float32)

    def set_embedding(
        self,
        text: str,
        vector: np.ndarray,
        ttl: int = _TTL_EMBEDDING,
    ) -> None:
        key = _NS_EMBEDDING + _hash(text)
        # Store as plain list so the value is Redis-serializable later.
        self._b.set(key, vector.tolist(), ttl)

    # ── ACL cache ─────────────────────────────────────────────────────────────

    def get_acl(self, user_id: str | int) -> list[str] | None:
        """Return cached doc_id list, or None if not cached.

        Note: a cached value of [] means the user has no access (distinct from
        a cache miss which returns None).  A cached value of the sentinel
        '__admin__' means the user is an unrestricted admin.
        """
        key = _NS_ACL + str(user_id)
        return self._b.get(key)

    def set_acl(
        self,
        user_id: str | int,
        doc_ids: list[str] | None,
        ttl: int = _TTL_ACL,
    ) -> None:
        """Cache ACL result.  Pass None for admin (unrestricted) users."""
        key = _NS_ACL + str(user_id)
        # Encode None (admin) as a sentinel so we can distinguish cache-miss
        # from "user is admin" when reading back.
        self._b.set(key, "__admin__" if doc_ids is None else doc_ids, ttl)

    def get_acl_resolved(self, user_id: str | int) -> tuple[bool, list[str] | None]:
        """Return (cache_hit, doc_ids).

        doc_ids=None  → admin / unrestricted
        doc_ids=[]    → no access
        cache_hit=False → not cached, must query OS
        """
        value = self.get_acl(user_id)
        if value is None:
            return False, None
        if value == "__admin__":
            return True, None
        return True, value

    def invalidate_acl(self, user_id: str | int) -> None:
        self._b.delete(_NS_ACL + str(user_id))

    # ── User profile cache ────────────────────────────────────────────────────

    def get_profile(self, user_id: str | int) -> "UserProfile | None":
        key = _NS_PROFILE + str(user_id)
        return self._b.get(key)

    def set_profile(
        self,
        user_id: str | int,
        profile: "UserProfile",
        ttl: int = _TTL_PROFILE,
    ) -> None:
        key = _NS_PROFILE + str(user_id)
        self._b.set(key, profile, ttl)

    def invalidate_profile(self, user_id: str | int) -> None:
        self._b.delete(_NS_PROFILE + str(user_id))

    # ── Document metadata cache ───────────────────────────────────────────────

    def get_doc_meta(self, doc_id: str) -> dict | None:
        key = _NS_DOC_META + doc_id
        return self._b.get(key)

    def set_doc_meta(
        self,
        doc_id: str,
        meta: dict,
        ttl: int = _TTL_DOC_META,
    ) -> None:
        """meta should contain: title, audiences, document_type."""
        key = _NS_DOC_META + doc_id
        self._b.set(key, meta, ttl)

    def invalidate_doc_meta(self, doc_id: str) -> None:
        self._b.delete(_NS_DOC_META + doc_id)

    # ── Entity expansion cache ────────────────────────────────────────────────

    def get_entity_expansion(self, focus_terms: list[str]) -> list[str] | None:
        key = _NS_ENTITY_EXP + _hash(sorted(focus_terms))
        return self._b.get(key)

    def set_entity_expansion(
        self,
        focus_terms: list[str],
        expanded: list[str],
        ttl: int = _TTL_ENTITY_EXP,
    ) -> None:
        key = _NS_ENTITY_EXP + _hash(sorted(focus_terms))
        self._b.set(key, expanded, ttl)

    # ── Bulk invalidation ─────────────────────────────────────────────────────

    def clear_answers(self) -> None:
        self._b.clear_prefix(_NS_ANSWER)

    def clear_all(self) -> None:
        for prefix in (_NS_ANSWER, _NS_EMBEDDING, _NS_ACL,
                       _NS_PROFILE, _NS_DOC_META, _NS_ENTITY_EXP):
            self._b.clear_prefix(prefix)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        backend = self._b
        return {
            "backend": type(backend).__name__,
            "size": backend.size() if hasattr(backend, "size") else "n/a",
        }
