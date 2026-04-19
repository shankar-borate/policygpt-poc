"""ThreadRepository — persists ThreadState to an OpenSearch index.

Design
------
- One index: {prefix}_threads, one document per thread (_id = thread_id).
- display_messages and last_answer_sources are stored but NOT indexed
  (enabled: false) — we never search message content.
- list_for_user() fetches only summary fields (no messages) to keep list
  calls lightweight.
- save() is called synchronously after bot.chat() completes; the LLM call
  already dominates latency so one OS write per response is negligible.
- ConversationManager clears display_messages from in-memory ThreadState
  after save() — only recent_messages (LLM context window) stay in memory.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from policygpt.config import Config
from policygpt.models.conversation import Message, ThreadState
from policygpt.models.retrieval import SourceReference
from policygpt.search.providers.opensearch.client import create_client

logger = logging.getLogger(__name__)

_THREADS_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "thread_id":            {"type": "keyword"},
            "user_id":              {"type": "keyword"},
            "title":                {"type": "keyword"},
            "created_at":           {"type": "date"},
            "updated_at":           {"type": "date"},
            "current_topic":        {"type": "keyword"},
            "active_doc_ids":       {"type": "keyword"},
            "active_section_ids":   {"type": "keyword"},
            "conversation_summary": {"type": "text"},
            # Messages and sources are stored as opaque objects — not indexed.
            "display_messages":     {"type": "object", "enabled": False},
            "recent_messages":      {"type": "object", "enabled": False},
            "last_answer_sources":  {"type": "object", "enabled": False},
        }
    },
}


class ThreadRepository:
    """OpenSearch-backed persistence for ThreadState objects."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._index = f"{config.search.opensearch_index_prefix}_threads"
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            self._client = create_client(
                host=self._config.search.opensearch_host,
                port=self._config.search.opensearch_port,
                username=self._config.search.opensearch_username,
                password=self._config.search.opensearch_password,
                use_ssl=self._config.search.opensearch_use_ssl,
                verify_certs=self._config.search.opensearch_verify_certs,
            )
        return self._client

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def ensure_index(self) -> None:
        """Create the threads index if it does not already exist."""
        if not self.client.indices.exists(index=self._index):
            self.client.indices.create(index=self._index, body=_THREADS_MAPPING)
            logger.info("Created OpenSearch index: %s", self._index)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, thread: ThreadState) -> None:
        """Upsert full thread state. Silently logs on failure — never raises."""
        try:
            body = {
                "thread_id":            thread.thread_id,
                "user_id":              thread.user_id,
                "title":                thread.title,
                "created_at":           thread.created_at,
                "updated_at":           thread.updated_at,
                "current_topic":        thread.current_topic,
                "active_doc_ids":       thread.active_doc_ids,
                "active_section_ids":   thread.active_section_ids,
                "conversation_summary": thread.conversation_summary,
                "display_messages": [
                    {"role": m.role, "content": m.content}
                    for m in thread.display_messages
                ],
                "recent_messages": [
                    {"role": m.role, "content": m.content}
                    for m in thread.recent_messages
                ],
                "last_answer_sources": [
                    _serialize_source(s) for s in thread.last_answer_sources
                ],
            }
            self.client.index(index=self._index, id=thread.thread_id, body=body)
        except Exception as exc:
            logger.warning("ThreadRepository.save failed [%s]: %s", thread.thread_id, exc)

    def append_and_save(self, thread: ThreadState) -> None:
        """Atomically append new display_messages and update all other fields.

        Uses a single Painless script update so the append is atomic — no
        read-modify-write race condition between concurrent app servers.
        The 'upsert' block handles the first save for a brand-new thread.
        """
        new_display = [{"role": m.role, "content": m.content} for m in thread.display_messages]
        recent = [{"role": m.role, "content": m.content} for m in thread.recent_messages]
        sources = [_serialize_source(s) for s in thread.last_answer_sources]
        try:
            self.client.update(
                index=self._index,
                id=thread.thread_id,
                body={
                    "script": {
                        "lang": "painless",
                        "source": """
                            if (ctx._source.display_messages == null) {
                                ctx._source.display_messages = params.new_display;
                            } else {
                                ctx._source.display_messages.addAll(params.new_display);
                            }
                            ctx._source.recent_messages      = params.recent_messages;
                            ctx._source.active_doc_ids       = params.active_doc_ids;
                            ctx._source.active_section_ids   = params.active_section_ids;
                            ctx._source.last_answer_sources  = params.last_answer_sources;
                            ctx._source.conversation_summary = params.conversation_summary;
                            ctx._source.current_topic        = params.current_topic;
                            ctx._source.title                = params.title;
                            ctx._source.updated_at           = params.updated_at;
                        """,
                        "params": {
                            "new_display":          new_display,
                            "recent_messages":      recent,
                            "active_doc_ids":       thread.active_doc_ids,
                            "active_section_ids":   thread.active_section_ids,
                            "last_answer_sources":  sources,
                            "conversation_summary": thread.conversation_summary,
                            "current_topic":        thread.current_topic,
                            "title":                thread.title,
                            "updated_at":           thread.updated_at,
                        },
                    },
                    # Full document for the case where the thread doesn't exist yet.
                    "upsert": {
                        "thread_id":            thread.thread_id,
                        "user_id":              thread.user_id,
                        "title":                thread.title,
                        "created_at":           thread.created_at,
                        "updated_at":           thread.updated_at,
                        "current_topic":        thread.current_topic,
                        "active_doc_ids":       thread.active_doc_ids,
                        "active_section_ids":   thread.active_section_ids,
                        "conversation_summary": thread.conversation_summary,
                        "display_messages":     new_display,
                        "recent_messages":      recent,
                        "last_answer_sources":  sources,
                    },
                },
            )
        except Exception as exc:
            logger.warning("ThreadRepository.append_and_save failed [%s]: %s", thread.thread_id, exc)

    def delete(self, thread_id: str) -> None:
        try:
            self.client.delete(index=self._index, id=thread_id, ignore=[404])
        except Exception as exc:
            logger.warning("ThreadRepository.delete failed [%s]: %s", thread_id, exc)

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, thread_id: str) -> ThreadState | None:
        """Load full thread (including messages) by ID. Returns None if missing."""
        try:
            resp = self.client.get(index=self._index, id=thread_id, ignore=[404])
            if not resp.get("found"):
                return None
            return _deserialize(resp["_source"])
        except Exception as exc:
            logger.warning("ThreadRepository.load failed [%s]: %s", thread_id, exc)
            return None

    def list_for_user(self, user_id: str, size: int = 20) -> list[dict]:
        """Return lightweight thread summaries (no messages) sorted by updated_at."""
        try:
            body = {
                "query": {"term": {"user_id": user_id}},
                "sort":  [{"updated_at": {"order": "desc"}}],
                "size":  size,
                "_source": [
                    "thread_id", "user_id", "title",
                    "created_at", "updated_at",
                ],
            }
            resp = self.client.search(index=self._index, body=body)
            return [hit["_source"] for hit in resp.get("hits", {}).get("hits", [])]
        except Exception as exc:
            logger.warning("ThreadRepository.list_for_user failed [%s]: %s", user_id, exc)
            return []


# ── Serialization helpers ──────────────────────────────────────────────────


def _serialize_source(source: Any) -> dict:
    try:
        return dataclasses.asdict(source)
    except TypeError:
        return source if isinstance(source, dict) else {}


def _deserialize(src: dict) -> ThreadState:
    thread = ThreadState(
        thread_id=src["thread_id"],
        user_id=src.get("user_id", ""),
    )
    thread.title                = src.get("title", "New chat")
    thread.created_at           = src.get("created_at", thread.created_at)
    thread.updated_at           = src.get("updated_at", thread.updated_at)
    thread.current_topic        = src.get("current_topic", "")
    thread.active_doc_ids       = src.get("active_doc_ids", [])
    thread.active_section_ids   = src.get("active_section_ids", [])
    thread.conversation_summary = src.get("conversation_summary", "")
    thread.display_messages = [
        Message(role=m["role"], content=m["content"])
        for m in src.get("display_messages", [])
    ]
    thread.recent_messages = [
        Message(role=m["role"], content=m["content"])
        for m in src.get("recent_messages", [])
    ]
    thread.last_answer_sources = [
        SourceReference(**s) for s in src.get("last_answer_sources", [])
        if isinstance(s, dict)
    ]
    return thread
