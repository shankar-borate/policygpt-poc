"""Tests for the OpenSearch ACL layer.

Uses a mock OpenSearch client so no real cluster is needed.
Tests the grant_access / revoke_access / get_accessible_doc_ids /
grant_admin_access flow with 10 users (IDs 100–109).
"""

import unittest
from unittest.mock import MagicMock, call, patch
from datetime import datetime, timezone

from policygpt.config import Config
from policygpt.search.providers.opensearch.store import OpenSearchVectorStore, _ADMIN_WILDCARD


def _make_store() -> tuple[OpenSearchVectorStore, MagicMock]:
    """Return a store wired to a mock client."""
    config = Config(
        opensearch_index_prefix="test",
        opensearch_host="localhost",
    )
    store = OpenSearchVectorStore(config)
    mock_client = MagicMock()
    store._client = mock_client
    return store, mock_client


USER_IDS = [str(uid) for uid in range(100, 110)]   # ["100", "101", ..., "109"]
DOC_A = "doc_alpha"
DOC_B = "doc_beta"


class TestGrantAccess(unittest.TestCase):
    """grant_access bulk-indexes one record per (user, doc) pair."""

    def test_bulk_called_with_correct_record_count(self):
        store, client = _make_store()
        store.grant_access(USER_IDS, DOC_A)

        client.bulk.assert_called_once()
        actions = client.bulk.call_args[1]["body"]
        # Each user → 2 items: action header + body
        self.assertEqual(len(actions), len(USER_IDS) * 2)

    def test_record_ids_are_composite_keys(self):
        store, client = _make_store()
        store.grant_access(USER_IDS, DOC_A)

        actions = client.bulk.call_args[1]["body"]
        headers = actions[::2]   # every other item starting at 0 is the action header
        ids = [h["index"]["_id"] for h in headers]

        for uid in USER_IDS:
            self.assertIn(f"{uid}__{DOC_A}", ids)

    def test_correct_user_id_and_doc_id_in_body(self):
        store, client = _make_store()
        store.grant_access(["100"], DOC_A)

        actions = client.bulk.call_args[1]["body"]
        body = actions[1]   # second item is the document body
        self.assertEqual(body["user_id"], "100")
        self.assertEqual(body["doc_id"], DOC_A)

    def test_empty_user_list_does_not_call_bulk(self):
        store, client = _make_store()
        store.grant_access([], DOC_A)
        client.bulk.assert_not_called()

    def test_integer_user_ids_are_coerced_to_strings(self):
        store, client = _make_store()
        store.grant_access([100, 101, 102], DOC_A)

        actions = client.bulk.call_args[1]["body"]
        bodies = actions[1::2]
        user_ids_stored = [b["user_id"] for b in bodies]
        self.assertEqual(user_ids_stored, ["100", "101", "102"])


class TestRevokeAccess(unittest.TestCase):
    """revoke_access deletes the composite-key record."""

    def test_delete_called_with_correct_id(self):
        store, client = _make_store()
        store.revoke_access("105", DOC_A)
        client.delete.assert_called_once_with(
            index="test_recipient",
            id="105__doc_alpha",
            ignore=[404],
        )

    def test_revoke_all_access_for_doc(self):
        store, client = _make_store()
        store.revoke_all_access_for_doc(DOC_A)
        client.delete_by_query.assert_called_once_with(
            index="test_recipient",
            body={"query": {"term": {"doc_id": DOC_A}}},
        )


class TestGetAccessibleDocIds(unittest.TestCase):
    """get_accessible_doc_ids returns the correct doc list from an aggregation."""

    def _mock_agg_response(self, doc_ids: list[str]) -> dict:
        return {
            "aggregations": {
                "doc_ids": {
                    "buckets": [{"key": doc_id} for doc_id in doc_ids]
                }
            }
        }

    def test_returns_assigned_docs_for_regular_user(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([DOC_A, DOC_B])

        result = store.get_accessible_doc_ids("100")
        self.assertEqual(result, [DOC_A, DOC_B])

    def test_returns_empty_list_for_user_with_no_assignments(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([])

        result = store.get_accessible_doc_ids("100")
        self.assertEqual(result, [])

    def test_returns_none_for_admin_wildcard(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([DOC_A, _ADMIN_WILDCARD])

        result = store.get_accessible_doc_ids("100")
        self.assertIsNone(result)

    def test_returns_empty_list_on_exception(self):
        store, client = _make_store()
        client.search.side_effect = Exception("connection refused")

        result = store.get_accessible_doc_ids("100")
        self.assertEqual(result, [])

    def test_query_targets_access_index(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([DOC_A])

        store.get_accessible_doc_ids("103")

        call_kwargs = client.search.call_args[1]
        self.assertEqual(call_kwargs["index"], "test_recipient")
        self.assertEqual(
            call_kwargs["body"]["query"],
            {"term": {"user_id": "103"}},
        )


class TestGrantAdminAccess(unittest.TestCase):

    def test_admin_record_uses_wildcard_doc_id(self):
        store, client = _make_store()
        store.grant_admin_access("100")

        client.index.assert_called_once()
        call_kwargs = client.index.call_args[1]
        self.assertEqual(call_kwargs["body"]["doc_id"], _ADMIN_WILDCARD)
        self.assertEqual(call_kwargs["body"]["user_id"], "100")
        self.assertEqual(call_kwargs["id"], f"100__{_ADMIN_WILDCARD}")


class TestBuildDocIdFilter(unittest.TestCase):
    """_build_doc_id_filter translates ACL results into OS filter clauses."""

    def _mock_agg_response(self, doc_ids: list[str]) -> dict:
        return {
            "aggregations": {
                "doc_ids": {"buckets": [{"key": d} for d in doc_ids]}
            }
        }

    def test_returns_none_for_admin(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([_ADMIN_WILDCARD])

        f = store._build_doc_id_filter("100")
        self.assertIsNone(f)

    def test_returns_terms_filter_for_regular_user(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([DOC_A, DOC_B])

        f = store._build_doc_id_filter("100")
        self.assertEqual(f, {"terms": {"doc_id": [DOC_A, DOC_B]}})

    def test_no_access_returns_impossible_filter(self):
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([])

        f = store._build_doc_id_filter("100")
        self.assertEqual(f, {"terms": {"doc_id": ["__no_access__"]}})


class TestEndToEndWithTenUsers(unittest.TestCase):
    """Scenario: 10 users (100–109), two documents, selective access."""

    def _mock_agg_response(self, doc_ids: list[str]) -> dict:
        return {
            "aggregations": {
                "doc_ids": {"buckets": [{"key": d} for d in doc_ids]}
            }
        }

    def test_all_ten_users_granted_to_doc_a(self):
        store, client = _make_store()
        store.grant_access(USER_IDS, DOC_A)

        actions = client.bulk.call_args[1]["body"]
        stored_users = [b["user_id"] for b in actions[1::2]]
        self.assertEqual(sorted(stored_users), sorted(USER_IDS))

    def test_only_even_users_see_doc_b(self):
        """Users 100, 102, 104, 106, 108 assigned to doc B."""
        store, client = _make_store()
        even_users = [uid for uid in USER_IDS if int(uid) % 2 == 0]

        store.grant_access(even_users, DOC_B)

        actions = client.bulk.call_args[1]["body"]
        stored_users = sorted([b["user_id"] for b in actions[1::2]])
        self.assertEqual(stored_users, sorted(even_users))

    def test_odd_user_sees_only_doc_a(self):
        """User 101 (odd) is assigned to doc_A only — doc_B filter returns it."""
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([DOC_A])

        doc_ids = store.get_accessible_doc_ids("101")
        self.assertEqual(doc_ids, [DOC_A])

    def test_even_user_sees_both_docs(self):
        """User 100 (even) is assigned to both doc_A and doc_B."""
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([DOC_A, DOC_B])

        doc_ids = store.get_accessible_doc_ids("100")
        self.assertIn(DOC_A, doc_ids)
        self.assertIn(DOC_B, doc_ids)

    def test_user_109_admin_sees_all(self):
        """User 109 promoted to admin — get_accessible_doc_ids returns None."""
        store, client = _make_store()
        client.search.return_value = self._mock_agg_response([_ADMIN_WILDCARD])

        doc_ids = store.get_accessible_doc_ids("109")
        self.assertIsNone(doc_ids)

    def test_revoking_user_101_from_doc_a(self):
        store, client = _make_store()
        store.revoke_access("101", DOC_A)

        client.delete.assert_called_once_with(
            index="test_recipient",
            id="101__doc_alpha",
            ignore=[404],
        )

    def test_delete_document_revokes_all_access(self):
        store, client = _make_store()
        # Suppress delete on documents index
        client.delete.return_value = {}
        store.delete_document(DOC_A)

        # delete_by_query called for sections, faqs, and access
        dq_calls = client.delete_by_query.call_args_list
        indices_called = [c[1]["index"] for c in dq_calls]
        self.assertIn("test_recipient", indices_called)
        # Access query filters by doc_id
        access_call = next(c for c in dq_calls if c[1]["index"] == "test_recipient")
        self.assertEqual(access_call[1]["body"]["query"]["term"]["doc_id"], DOC_A)


if __name__ == "__main__":
    unittest.main()
