"""OpenSearchVectorStore — the OpenSearch Adapter.

Implements the VectorStore port using the opensearch-py SDK.
The client is created lazily on first use.

Three search strategies
───────────────────────
keyword_search    multi_match with field-level boosting + AUTO fuzziness
similarity_search more_like_this across raw_text, summary, section_title
vector_search     knn query on the "embedding" knn_vector field

Access control
───────────────────────
Access is stored in a dedicated {prefix}_recipient index as flat
(user_id, doc_id) pairs — one record per assignment.  This keeps the
content indexes (sections, faqs, documents) lean regardless of how many
users are assigned to a document.

At query time the caller resolves a list of accessible doc_ids via
get_accessible_doc_ids(user_id) and passes it as a doc_id filter — the
same filter applies to all three section search types and to FAQ search.

Admin users: grant_admin_access(user_id) stores a wildcard record
(doc_id="*").  get_accessible_doc_ids returns None for these users,
signalling the retriever to skip the filter entirely.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from policygpt.config import Config
from policygpt.models.documents import DocumentRecord, SectionRecord
from policygpt.search.base import VectorStore
from policygpt.search.models import FaqResult, SearchQuery, SearchResult, SearchType
from .acl import OpenSearchACL, ADMIN_WILDCARD as _ADMIN_WILDCARD
from .client import create_client
from .mappings import recipient_mapping, documents_mapping, faqs_mapping, sections_mapping

logger = logging.getLogger(__name__)


class OpenSearchVectorStore(VectorStore):
    """OpenSearch implementation of VectorStore."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._sections_index  = f"{config.opensearch_index_prefix}_sections"
        self._documents_index = f"{config.opensearch_index_prefix}_documents"
        self._faqs_index      = f"{config.opensearch_index_prefix}_faqs"
        self._recipient_index = f"{config.opensearch_index_prefix}_recipient"
        self._client = None  # lazy — created on first use
        # ACL collaborator — wired up after the client is ready
        self._acl: OpenSearchACL | None = None

    # ── Client (lazy) ─────────────────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            self._client = create_client(
                host=self.config.opensearch_host,
                port=self.config.opensearch_port,
                username=self.config.opensearch_username,
                password=self.config.opensearch_password,
                use_ssl=self.config.opensearch_use_ssl,
                verify_certs=self.config.opensearch_verify_certs,
            )
            self._acl = OpenSearchACL(self._client, self._recipient_index)
        return self._client

    @property
    def acl(self) -> OpenSearchACL:
        """Ensure the client (and ACL) is initialised before use."""
        _ = self.client  # triggers lazy init
        return self._acl  # type: ignore[return-value]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            info = self.client.cluster.health()
            return info.get("status") in ("green", "yellow")
        except Exception as exc:
            logger.warning("OpenSearch health check failed: %s", exc)
            return False

    def ensure_index(self, embedding_dim: int) -> None:
        for index_name, mapping in [
            (self._sections_index,  sections_mapping(embedding_dim)),
            (self._documents_index, documents_mapping()),
            (self._faqs_index,      faqs_mapping(embedding_dim)),
            (self._recipient_index, recipient_mapping()),
        ]:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=mapping)
                logger.info("Created OpenSearch index: %s", index_name)

    # ── ACL — delegate to OpenSearchACL ──────────────────────────────────────

    def grant_access(self, user_ids: list[str | int], doc_id: str) -> None:
        self.acl.grant_access(user_ids, doc_id)

    def grant_admin_access(self, user_id: str | int) -> None:
        self.acl.grant_admin_access(user_id)

    def revoke_access(self, user_id: str | int, doc_id: str) -> None:
        self.acl.revoke_access(user_id, doc_id)

    def revoke_all_access_for_doc(self, doc_id: str) -> None:
        self.acl.revoke_all_access_for_doc(doc_id)

    def get_accessible_doc_ids(self, user_id: str | int) -> list[str] | None:
        return self.acl.get_accessible_doc_ids(user_id)

    # ── Write ─────────────────────────────────────────────────────────────────

    def index_document(
        self,
        document: DocumentRecord,
        user_ids: list[str | int],
        domain: str,
    ) -> None:
        """Upsert document record, sections, and access grants.

        user_ids are written to the access index only — not embedded in the
        document or section bodies — so adding/removing users never requires
        re-indexing content.
        """
        self.client.index(
            index=self._documents_index,
            id=document.doc_id,
            body={
                "doc_id":                document.doc_id,
                "title":                 document.title,
                "document_type":         document.document_type,
                "version":               document.version,
                "effective_date":        document.effective_date,
                "metadata_tags":         document.metadata_tags,
                "audiences":             document.audiences,
                "keywords":              document.keywords,
                "source_path":           document.source_path,
                "original_source_path":  document.original_source_path,
                "summary":               document.summary,
                "domain":                domain,
            },
        )

        for section in document.sections:
            self._index_section(section, document, domain)

        # Grant access — separate from content, idempotent
        str_user_ids = [str(uid) for uid in user_ids]
        self.grant_access(str_user_ids, document.doc_id)

        logger.debug(
            "Indexed document '%s' with %d sections (domain=%s, users=%d)",
            document.title, len(document.sections), domain, len(user_ids),
        )

    def _index_section(
        self,
        section: SectionRecord,
        document: DocumentRecord,
        domain: str,
    ) -> None:
        body: dict[str, Any] = {
            "section_id":     section.section_id,
            "doc_id":         section.doc_id,
            "document_title": document.title,
            "section_title":  section.title,
            "raw_text":       section.raw_text,
            "summary":        section.summary,
            "section_type":   section.section_type,
            "metadata_tags":  section.metadata_tags,
            "keywords":       section.keywords,
            "audiences":      document.audiences,
            "source_path":    section.source_path,
            "order_index":    section.order_index,
            "domain":         domain,
            "images":         section.images,
        }
        if section.summary_embedding is not None and section.summary_embedding.size > 0:
            body["embedding"] = section.summary_embedding.tolist()

        self.client.index(
            index=self._sections_index,
            id=section.section_id,
            body=body,
        )

    def get_cached_document(self, source_path: str) -> dict | None:
        """Fetch document + section metadata from OpenSearch for the given source_path."""
        try:
            doc_resp = self.client.search(
                index=self._documents_index,
                body={
                    "query": {"term": {"source_path": source_path}},
                    "size": 1,
                },
            )
            doc_hits = doc_resp.get("hits", {}).get("hits", [])
            if not doc_hits:
                return None
            doc_src = doc_hits[0]["_source"]
            doc_id = doc_src.get("doc_id", doc_hits[0]["_id"])

            sec_resp = self.client.search(
                index=self._sections_index,
                body={
                    "query": {"term": {"doc_id": doc_id}},
                    "size": 500,
                    "sort": [{"order_index": {"order": "asc"}}],
                    "_source": [
                        "section_id", "section_title", "source_path",
                        "order_index", "section_type", "metadata_tags",
                        "keywords", "summary", "raw_text", "images",
                    ],
                },
            )
            sections = []
            for hit in sec_resp.get("hits", {}).get("hits", []):
                s = hit["_source"]
                sections.append({
                    "section_id":    s.get("section_id", hit["_id"]),
                    "title":         s.get("section_title", ""),
                    "source_path":   s.get("source_path", source_path),
                    "order_index":   int(s.get("order_index", 0)),
                    "section_type":  s.get("section_type", "general"),
                    "metadata_tags": s.get("metadata_tags", []),
                    "keywords":      s.get("keywords", []),
                    "summary":       s.get("summary", ""),
                    "raw_text":      s.get("raw_text", ""),
                    "images":        s.get("images", []),
                    "masked_text":   s.get("raw_text", ""),
                })

            return {
                "doc_id":               doc_id,
                "title":                doc_src.get("title", ""),
                "source_path":          doc_src.get("source_path", source_path),
                "original_source_path": doc_src.get("original_source_path", ""),
                "summary":              doc_src.get("summary", ""),
                "version":              doc_src.get("version", ""),
                "effective_date":       doc_src.get("effective_date", ""),
                "document_type":        doc_src.get("document_type", "document"),
                "metadata_tags":        doc_src.get("metadata_tags", []),
                "audiences":            doc_src.get("audiences", []),
                "keywords":             doc_src.get("keywords", []),
                "sections":             sections,
            }
        except Exception as exc:
            logger.warning("get_cached_document failed for '%s': %s", source_path, exc)
            return None

    def count_documents(self) -> int:
        try:
            resp = self.client.count(
                index=self._documents_index,
                body={"query": {"match_all": {}}},
            )
            return int(resp.get("count", 0))
        except Exception:
            return 0

    def count_sections(self) -> int:
        try:
            resp = self.client.count(
                index=self._sections_index,
                body={"query": {"match_all": {}}},
            )
            return int(resp.get("count", 0))
        except Exception:
            return 0

    def document_indexed_for_path(self, source_path: str) -> bool:
        try:
            body = {
                "query": {"term": {"source_path": source_path}},
                "size": 1,
                "_source": False,
            }
            resp = self.client.search(index=self._documents_index, body=body)
            return resp.get("hits", {}).get("total", {}).get("value", 0) > 0
        except Exception as exc:
            logger.warning("document_indexed_for_path check failed for '%s': %s", source_path, exc)
            return False

    def delete_document(self, doc_id: str) -> None:
        try:
            self.client.delete(index=self._documents_index, id=doc_id, ignore=[404])
        except Exception:
            pass
        for index in (self._sections_index, self._faqs_index):
            self.client.delete_by_query(
                index=index,
                body={"query": {"term": {"doc_id": doc_id}}},
            )
        # Remove all user assignments for this document
        self.revoke_all_access_for_doc(doc_id)
        logger.debug("Deleted document %s from OpenSearch", doc_id)

    def index_faq_pairs(
        self,
        doc_id: str,
        document_title: str,
        source_path: str,
        qa_pairs: list[tuple[str, str]],
        q_embeddings: list[np.ndarray],
        user_ids: list[str],
        domain: str,
    ) -> None:
        """Index each Q/A pair as its own document in the FAQ index.

        user_ids is accepted for interface compatibility but not stored —
        access is resolved via the access index at query time.
        """
        if not qa_pairs:
            return

        for idx, ((question, answer), embedding) in enumerate(zip(qa_pairs, q_embeddings)):
            faq_id = f"{doc_id}_faq_{idx}"
            body: dict[str, Any] = {
                "faq_id":         faq_id,
                "doc_id":         doc_id,
                "document_title": document_title,
                "question":       question,
                "answer":         answer,
                "source_path":    source_path,
                "domain":         domain,
            }
            if embedding is not None and embedding.size > 0:
                body["question_embedding"] = embedding.tolist()

            self.client.index(
                index=self._faqs_index,
                id=faq_id,
                body=body,
            )

        logger.debug(
            "Indexed %d FAQ pairs for document '%s' (domain=%s)",
            len(qa_pairs), document_title, domain,
        )

    # ── FAQ search ────────────────────────────────────────────────────────────

    def faq_search(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        min_score: float = 0.92,
    ) -> FaqResult | None:
        if query_embedding is None or query_embedding.size == 0:
            return None

        doc_filter = self._build_doc_id_filter(user_id)
        body: dict[str, Any] = {
            "size": 1,
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "question_embedding": {
                                "vector": query_embedding.tolist(),
                                "k": 1,
                            }
                        }
                    },
                }
            },
        }
        if doc_filter:
            body["query"]["bool"]["filter"] = [doc_filter]

        response = self.client.search(index=self._faqs_index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return None

        hit = hits[0]
        score = float(hit.get("_score") or 0.0)
        if score < min_score:
            return None

        src = hit["_source"]
        return FaqResult(
            faq_id=         src.get("faq_id", hit["_id"]),
            doc_id=         src.get("doc_id", ""),
            document_title= src.get("document_title", ""),
            question=       src.get("question", ""),
            answer=         src.get("answer", ""),
            source_path=    src.get("source_path", ""),
            score=          score,
        )

    def search_faq_questions(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        top_k: int = 30,
    ) -> list[FaqResult]:
        if query_embedding is None or query_embedding.size == 0:
            return []

        doc_filter = self._build_doc_id_filter(user_id)
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "question_embedding": {
                                "vector": query_embedding.tolist(),
                                "k": top_k,
                            }
                        }
                    },
                }
            },
        }
        if doc_filter:
            body["query"]["bool"]["filter"] = [doc_filter]

        response = self.client.search(index=self._faqs_index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results: list[FaqResult] = []
        for hit in hits:
            src = hit["_source"]
            results.append(FaqResult(
                faq_id=         src.get("faq_id", hit["_id"]),
                doc_id=         src.get("doc_id", ""),
                document_title= src.get("document_title", ""),
                question=       src.get("question", ""),
                answer=         src.get("answer", ""),
                source_path=    src.get("source_path", ""),
                score=          float(hit.get("_score") or 0.0),
            ))
        return results

    # ── Document search (Google-style) ───────────────────────────────────────

    def search_documents(
        self,
        query_text: str,
        user_id: str,
        page: int = 1,
        size: int = 10,
    ) -> dict:
        fetch_size = min(size * 8, 200)
        must_clause: dict = {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "section_title^4",
                    "document_title^3",
                    "keywords^3",
                    "summary^2",
                    "raw_text^1",
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "operator": "or",
            }
        }
        body: dict[str, Any] = {
            "size": fetch_size,
            "_source": [
                "doc_id", "document_title", "section_title",
                "summary", "raw_text", "source_path", "order_index",
            ],
        }
        doc_filter = self._build_doc_id_filter(user_id)
        if doc_filter:
            body["query"] = {
                "bool": {"must": must_clause, "filter": [doc_filter]}
            }
        else:
            body["query"] = must_clause

        resp = self.client.search(index=self._sections_index, body=body)
        hits = resp.get("hits", {}).get("hits", [])

        seen: dict[str, dict] = {}
        for hit in hits:
            src = hit.get("_source", {})
            doc_id = src.get("doc_id", hit["_id"])
            if doc_id in seen:
                continue
            text = src.get("summary") or src.get("raw_text", "")
            snippet = text[:240].rstrip() + ("\u2026" if len(text) > 240 else "")
            seen[doc_id] = {
                "document_title": src.get("document_title", ""),
                "section_title":  src.get("section_title", ""),
                "snippet":        snippet,
                "source_path":    src.get("source_path", ""),
                "section_index":  int(src.get("order_index", 0)),
                "score":          round(float(hit.get("_score") or 0.0), 4),
            }

        all_results = list(seen.values())
        total = len(all_results)
        from_offset = (page - 1) * size
        return {
            "total":   total,
            "page":    page,
            "size":    size,
            "results": all_results[from_offset: from_offset + size],
        }

    # ── Section search strategies ─────────────────────────────────────────────

    def keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        body = {
            "size": query.top_k,
            "query": {
                "multi_match": {
                    "query": query.text,
                    "fields": [
                        "section_title^4",
                        "document_title^3",
                        "keywords^3",
                        "summary^2",
                        "raw_text^1",
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "or",
                }
            },
        }
        body = _apply_filters(body, query.filters)
        response = self.client.search(index=self._sections_index, body=body)
        return _parse_hits(response, SearchType.KEYWORD)

    def similarity_search(self, query: SearchQuery) -> list[SearchResult]:
        body = {
            "size": query.top_k,
            "query": {
                "more_like_this": {
                    "fields":               ["raw_text", "summary", "section_title"],
                    "like":                 query.text,
                    "min_term_freq":        1,
                    "min_doc_freq":         1,
                    "max_query_terms":      25,
                    "minimum_should_match": "30%",
                    "boost_terms":          1.0,
                }
            },
        }
        body = _apply_filters(body, query.filters)
        response = self.client.search(index=self._sections_index, body=body)
        return _parse_hits(response, SearchType.SIMILARITY)

    def vector_search(self, query: SearchQuery) -> list[SearchResult]:
        if query.embedding is None or query.embedding.size == 0:
            return []
        body = {
            "size": query.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query.embedding.tolist(),
                        "k": query.top_k,
                    }
                }
            },
        }
        body = _apply_filters(body, query.filters)
        response = self.client.search(index=self._sections_index, body=body)
        return _parse_hits(response, SearchType.VECTOR)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_doc_id_filter(self, user_id: str | int) -> dict | None:
        """Delegate to OpenSearchACL — resolves to an OS filter clause or None."""
        return self.acl.build_doc_id_filter(user_id)


# ── Module-level helpers ───────────────────────────────────────────────────


def _apply_filters(body: dict, filters: dict) -> dict:
    """Wrap the query in a bool/filter clause for metadata pre-filtering."""
    if not filters:
        return body
    original_query = body.pop("query")
    body["query"] = {
        "bool": {
            "must": original_query,
            "filter": [
                {"terms": {k: v if isinstance(v, list) else [v]}}
                for k, v in filters.items()
            ],
        }
    }
    return body


def _parse_hits(response: dict, search_type: SearchType) -> list[SearchResult]:
    results: list[SearchResult] = []
    for hit in response.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        results.append(
            SearchResult(
                section_id=     src.get("section_id", hit["_id"]),
                doc_id=         src.get("doc_id", ""),
                score=          float(hit.get("_score") or 0.0),
                document_title= src.get("document_title", ""),
                section_title=  src.get("section_title", ""),
                source_path=    src.get("source_path", ""),
                order_index=    int(src.get("order_index", 0)),
                raw_text=       src.get("raw_text", ""),
                summary=        src.get("summary", ""),
                section_type=   src.get("section_type", "general"),
                metadata_tags=  src.get("metadata_tags", []),
                keywords=       src.get("keywords", []),
                matched_by=     (search_type,),
            )
        )
    return results
