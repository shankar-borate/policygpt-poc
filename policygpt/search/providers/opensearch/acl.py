"""OpenSearch ACL (Access Control Layer).

Manages the ``{prefix}_recipient`` index which stores flat
``(user_id, doc_id)`` pairs — one record per user-document assignment.

Responsibilities
----------------
* grant_access / grant_admin_access — write user→doc assignments
* revoke_access / revoke_all_access_for_doc — remove assignments
* get_accessible_doc_ids — resolve which docs a user can see
* _build_doc_id_filter — translate ACL result into an OS filter clause

The admin wildcard (``doc_id="*"``) signals unrestricted access;
``get_accessible_doc_ids`` returns ``None`` for those users so the
caller knows to skip the filter entirely.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ADMIN_WILDCARD = "*"
_NO_ACCESS_SENTINEL = "__no_access__"


class OpenSearchACL:
    """Manages user → document access assignments in OpenSearch."""

    def __init__(self, client, recipient_index: str) -> None:
        self._client = client
        self._index  = recipient_index

    # ── Write ──────────────────────────────────────────────────────────────────

    def grant_access(self, user_ids: list[str | int], doc_id: str) -> None:
        """Bulk-upsert one access record per (user, doc) pair.

        The composite key ``{user_id}__{doc_id}`` makes re-granting idempotent.
        """
        if not user_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        actions: list[dict] = []
        for uid in user_ids:
            record_id = f"{uid}__{doc_id}"
            actions.append({"index": {"_index": self._index, "_id": record_id}})
            actions.append({"user_id": str(uid), "doc_id": doc_id, "granted_at": now})
        try:
            self._client.bulk(body=actions)
            logger.debug("Granted access: %d users → doc %s", len(user_ids), doc_id)
        except Exception as exc:
            logger.warning("grant_access failed for doc %s: %s", doc_id, exc)

    def grant_admin_access(self, user_id: str | int) -> None:
        """Grant unrestricted (admin) access — bypasses all doc_id filters."""
        now = datetime.now(timezone.utc).isoformat()
        record_id = f"{user_id}__{ADMIN_WILDCARD}"
        try:
            self._client.index(
                index=self._index,
                id=record_id,
                body={"user_id": str(user_id), "doc_id": ADMIN_WILDCARD, "granted_at": now},
            )
        except Exception as exc:
            logger.warning("grant_admin_access failed for user %s: %s", user_id, exc)

    def revoke_access(self, user_id: str | int, doc_id: str) -> None:
        """Remove a single user → document assignment."""
        record_id = f"{user_id}__{doc_id}"
        try:
            self._client.delete(index=self._index, id=record_id, ignore=[404])
        except Exception as exc:
            logger.warning("revoke_access failed (user=%s, doc=%s): %s", user_id, doc_id, exc)

    def revoke_all_access_for_doc(self, doc_id: str) -> None:
        """Remove all user assignments for a document (called on document deletion)."""
        try:
            self._client.delete_by_query(
                index=self._index,
                body={"query": {"term": {"doc_id": doc_id}}},
            )
        except Exception as exc:
            logger.warning("revoke_all_access_for_doc failed (doc=%s): %s", doc_id, exc)

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_accessible_doc_ids(self, user_id: str | int) -> list[str] | None:
        """Return the doc_ids this user may access.

        Returns
        -------
        None
            Admin (wildcard) user — caller should apply no filter.
        list[str]
            Specific doc_ids assigned to the user.  An empty list means no
            access at all.
        """
        try:
            resp = self._client.search(
                index=self._index,
                body={
                    "query": {"term": {"user_id": str(user_id)}},
                    "aggs": {
                        "doc_ids": {
                            "terms": {"field": "doc_id", "size": 10_000}
                        }
                    },
                    "size": 0,
                },
            )
            buckets = resp.get("aggregations", {}).get("doc_ids", {}).get("buckets", [])
            doc_ids = [b["key"] for b in buckets]
            if ADMIN_WILDCARD in doc_ids:
                return None   # unrestricted admin
            return doc_ids
        except Exception as exc:
            logger.warning("get_accessible_doc_ids failed for user %s: %s", user_id, exc)
            return []

    def build_doc_id_filter(self, user_id: str | int) -> dict | None:
        """Translate the ACL result into an OpenSearch filter clause.

        Returns
        -------
        None
            Admin user — no filter needed.
        dict
            ``terms`` filter on ``doc_id``.  Uses the no-access sentinel when
            the list is empty so the query returns zero results.
        """
        doc_ids = self.get_accessible_doc_ids(user_id)
        if doc_ids is None:
            return None
        return {"terms": {"doc_id": doc_ids or [_NO_ACCESS_SENTINEL]}}
