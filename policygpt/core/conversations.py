"""ConversationManager — thread lifecycle and optional OS-backed persistence.

Two modes
---------
In-memory only (no repo):
    All thread state lives in self.threads for the process lifetime.

OS-backed (repo provided):
    Designed for stateless, multi-server deployments.  OpenSearch is the
    single source of truth — no in-memory caching of thread state.

    - new_thread         → create ThreadState, save to OS immediately.
    - reset_thread       → overwrite thread in OS with blank state.
    - get_thread         → ALWAYS loads from OS (never serves stale in-memory
                           state). Ensures any app server sees the latest
                           recent_messages regardless of which server handled
                           the previous turn.
    - get_thread_for_display → loads from OS with full display_messages.
    - list_threads       → OS query (summaries only, no messages).
    - save_thread        → atomically appends new display_messages to the
                           existing OS document and updates all other fields
                           via a Painless script update. No read-modify-write
                           race condition.
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
        # Prefer OS for user_id so any server can reset any thread.
        if self._repo is not None:
            existing = self._repo.load(thread_id)
        else:
            existing = self.threads.get(thread_id)
        user_id = existing.user_id if existing else ""
        thread = ThreadState(thread_id=thread_id, user_id=user_id)
        self.threads[thread_id] = thread
        if self._repo is not None:
            self._repo.save(thread)

    def get_thread(self, thread_id: str) -> ThreadState:
        """Return the thread for LLM/bot use.

        OS-backed: always loads from OS so every app server sees the latest
        recent_messages written by whichever server handled the previous turn.
        display_messages are dropped (kept in OS only — not needed for LLM).

        In-memory: returns from self.threads, creating a blank thread on miss.
        """
        if self._repo is not None:
            loaded = self._repo.load(thread_id)
            if loaded is not None:
                loaded.display_messages = []
                return loaded
            # Thread exists in memory but not yet in OS (just created this request).
            return self.threads.get(thread_id) or ThreadState(thread_id=thread_id)

        # In-memory only mode.
        if thread_id in self.threads:
            return self.threads[thread_id]
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
        """Persist thread state to OS atomically.

        No-op when OS is not configured.

        Uses append_and_save() which issues a single Painless script update:
        new display_messages are appended to the existing OS array (no
        read-modify-write), and all other fields are updated in the same
        atomic operation. This is safe under concurrent requests from
        multiple app servers.
        """
        if self._repo is None:
            return
        self._repo.append_and_save(thread)
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
