"""ConversationManager — thread lifecycle and optional OS-backed persistence.

Two modes
---------
In-memory only (no repo):
    Existing behaviour unchanged.  All thread state (including
    display_messages) lives in self.threads for the process lifetime.

OS-backed (repo provided):
    - new_thread / reset_thread  → save to OS immediately.
    - get_thread                 → check in-memory first (hot cache),
                                   then lazy-load from OS.
    - get_thread_for_display     → always loads from OS so the caller gets
                                   the full message history even if it was
                                   cleared from memory.
    - list_threads               → OS query (summaries only, no messages).
    - After bot.chat() the route calls save_thread(thread) which:
        1. Persists to OS (full state including display_messages).
        2. Clears display_messages from the in-memory object so only
           recent_messages (LLM context window) stays resident.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from policygpt.models import ThreadState

if TYPE_CHECKING:
    from policygpt.storage.threads import ThreadRepository


class ConversationManager:
    def __init__(self, repo: "ThreadRepository | None" = None) -> None:
        self.threads: dict[str, ThreadState] = {}
        self._repo = repo

    # ── Thread lifecycle ──────────────────────────────────────────────────────

    def new_thread(self, user_id: str = "") -> str:
        thread_id = str(uuid.uuid4())
        thread = ThreadState(thread_id=thread_id, user_id=user_id)
        self.threads[thread_id] = thread
        if self._repo is not None:
            self._repo.save(thread)
        return thread_id

    def reset_thread(self, thread_id: str) -> None:
        existing = self.threads.get(thread_id)
        user_id = existing.user_id if existing else ""
        thread = ThreadState(thread_id=thread_id, user_id=user_id)
        self.threads[thread_id] = thread
        if self._repo is not None:
            self._repo.save(thread)

    def get_thread(self, thread_id: str) -> ThreadState:
        """Return the thread for LLM/bot use (recent_messages always in memory).

        When OS-backed, lazy-loads from OS on cache miss — but display_messages
        are NOT re-populated into in-memory state (kept lean).
        """
        if thread_id in self.threads:
            return self.threads[thread_id]

        if self._repo is not None:
            loaded = self._repo.load(thread_id)
            if loaded is not None:
                # Keep recent_messages in memory; drop display_messages to stay lean.
                loaded.display_messages = []
                self.threads[thread_id] = loaded
                return loaded

        # Fall-through: create a fresh thread (handles missing / first-access).
        thread = ThreadState(thread_id=thread_id)
        self.threads[thread_id] = thread
        return thread

    def get_thread_for_display(self, thread_id: str) -> ThreadState | None:
        """Return thread with full display_messages for API serialization.

        Returns None when the thread does not exist anywhere (OS or memory),
        so callers can distinguish "found but empty" from "not found".
        OS-backed mode:  loads fresh from OS (authoritative message history).
        In-memory mode:  returns from self.threads.
        """
        if self._repo is not None:
            loaded = self._repo.load(thread_id)
            if loaded is not None:
                return loaded
            # Not in OS yet — fall back to in-memory (e.g. just created,
            # not yet persisted via save_thread).
            return self.threads.get(thread_id)

        return self.threads.get(thread_id)

    def save_thread(self, thread: ThreadState) -> None:
        """Persist to OS then clear display_messages from in-memory state.

        No-op when OS is not configured.

        When OS-backed, loads the existing message history from OS and
        prepends it so that display_messages accumulates across turns rather
        than being overwritten with only the latest Q&A pair.
        """
        if self._repo is None:
            return
        # Accumulate display_messages: prepend OS history to the new messages
        # that chat() just appended to the (previously-cleared) in-memory list.
        if thread.display_messages:
            existing = self._repo.load(thread.thread_id)
            if existing and existing.display_messages:
                thread.display_messages = existing.display_messages + thread.display_messages
        self._repo.save(thread)
        # Keep memory lean — full message history now lives in OS only.
        thread.display_messages = []

    def list_threads(self, user_id: str = "") -> list[ThreadState]:
        """List threads.

        OS-backed + user_id provided → query OS for lightweight summaries.
        Otherwise → return sorted in-memory threads.
        """
        if self._repo is not None and user_id:
            summaries = self._repo.list_for_user(user_id)
            return [
                ThreadState(
                    thread_id=s["thread_id"],
                    user_id=s.get("user_id", ""),
                    title=s.get("title", "New chat"),
                    created_at=s.get("created_at", ""),
                    updated_at=s.get("updated_at", ""),
                )
                for s in summaries
            ]

        return sorted(
            self.threads.values(),
            key=lambda t: t.updated_at,
            reverse=True,
        )
