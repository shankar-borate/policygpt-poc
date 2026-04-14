"""OpenSearchVectorStore — the OpenSearch Adapter.

Implements the VectorStore port using the opensearch-py SDK.
The client is created lazily on first use.

Three search strategies
───────────────────────
keyword_search    multi_match with field-level boosting + AUTO fuzziness
similarity_search more_like_this across raw_text, summary, section_title
vector_search     knn query on the "embedding" knn_vector field

Metadata pre-filtering
───────────────────────
All three search methods apply query.filters as OpenSearch bool/filter terms
before scoring, so permission-based document scoping happens at the DB level.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from policygpt.config import Config
from policygpt.models.documents import DocumentRecord, SectionRecord
from policygpt.search.base import VectorStore
from policygpt.search.models import FaqResult, SearchQuery, SearchResult, SearchType
from .client import create_client
from .mappings import documents_mapping, faqs_mapping, sections_mapping

logger = logging.getLogger(__name__)


class OpenSearchVectorStore(VectorStore):
    """OpenSearch implementation of VectorStore."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._sections_index  = f"{config.opensearch_index_prefix}_sections"
        self._documents_index = f"{config.opensearch_index_prefix}_documents"
        self._faqs_index      = f"{config.opensearch_index_prefix}_faqs"
        self._client = None  # lazy — created on first use

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
        return self._client

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
        ]:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=mapping)
                logger.info("Created OpenSearch index: %s", index_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    def index_document(
        self,
        document: DocumentRecord,
        user_ids: list[str | int],
        domain: str,
    ) -> None:
        """Upsert document record and all its sections.

        user_ids and domain are stored on every document and section so that
        retrieval can filter by user_id at query time.  They are passed in
        by the caller — the store never derives them itself.
        """
        # Normalise user_ids to strings for consistent keyword matching in OS
        str_user_ids = [str(uid) for uid in user_ids]

        self.client.index(
            index=self._documents_index,
            id=document.doc_id,
            body={
                "doc_id":         document.doc_id,
                "title":          document.title,
                "document_type":  document.document_type,
                "version":        document.version,
                "effective_date": document.effective_date,
                "metadata_tags":  document.metadata_tags,
                "audiences":      document.audiences,
                "keywords":       document.keywords,
                "source_path":    document.source_path,
                "summary":        document.summary,
                "user_ids":       str_user_ids,
                "domain":         domain,
            },
        )
        for section in document.sections:
            self._index_section(section, document, str_user_ids, domain)

        logger.debug(
            "Indexed document '%s' with %d sections (domain=%s, user_ids=%s)",
            document.title, len(document.sections), domain, str_user_ids,
        )

    def _index_section(
        self,
        section: SectionRecord,
        document: DocumentRecord,
        user_ids: list[str],
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
            "user_ids":       user_ids,
            "domain":         domain,
        }
        if section.summary_embedding is not None and section.summary_embedding.size > 0:
            body["embedding"] = section.summary_embedding.tolist()

        self.client.index(
            index=self._sections_index,
            id=section.section_id,
            body=body,
        )

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
        """Index each Q/A pair as its own document in the FAQ index."""
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
                "user_ids":       user_ids,
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

    def faq_search(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        min_score: float = 0.92,
    ) -> FaqResult | None:
        """Return the top FAQ answer if its score meets min_score, else None."""
        if query_embedding is None or query_embedding.size == 0:
            return None

        body = {
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
                    "filter": [
                        {"terms": {"user_ids": [user_id]}}
                    ],
                }
            },
        }

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
            faq_id=        src.get("faq_id", hit["_id"]),
            doc_id=        src.get("doc_id", ""),
            document_title=src.get("document_title", ""),
            question=      src.get("question", ""),
            answer=        src.get("answer", ""),
            source_path=   src.get("source_path", ""),
            score=         score,
        )

    def search_faq_questions(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        top_k: int = 30,
    ) -> list[FaqResult]:
        """Return top-k FAQ results closest to query_embedding, filtered by user_id."""
        if query_embedding is None or query_embedding.size == 0:
            return []

        body = {
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
                    "filter": [
                        {"terms": {"user_ids": [user_id]}}
                    ],
                }
            },
        }

        response = self.client.search(index=self._faqs_index, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results: list[FaqResult] = []
        for hit in hits:
            src = hit["_source"]
            results.append(
                FaqResult(
                    faq_id=        src.get("faq_id", hit["_id"]),
                    doc_id=        src.get("doc_id", ""),
                    document_title=src.get("document_title", ""),
                    question=      src.get("question", ""),
                    answer=        src.get("answer", ""),
                    source_path=   src.get("source_path", ""),
                    score=         float(hit.get("_score") or 0.0),
                )
            )
        return results

    # ── Search strategies ─────────────────────────────────────────────────────

    def keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        """BM25 multi_match with field boosting and AUTO fuzziness."""
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
        """more_like_this — finds sections with similar vocabulary to the query."""
        body = {
            "size": query.top_k,
            "query": {
                "more_like_this": {
                    "fields":             ["raw_text", "summary", "section_title"],
                    "like":               query.text,
                    "min_term_freq":      1,
                    "min_doc_freq":       1,
                    "max_query_terms":    25,
                    "minimum_should_match": "30%",
                    "boost_terms":        1.0,
                }
            },
        }
        body = _apply_filters(body, query.filters)
        response = self.client.search(index=self._sections_index, body=body)
        return _parse_hits(response, SearchType.SIMILARITY)

    def vector_search(self, query: SearchQuery) -> list[SearchResult]:
        """kNN approximate nearest-neighbour on the embedding field."""
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


# ── Module-level helpers ───────────────────────────────────────────────────


def _apply_filters(body: dict, filters: dict) -> dict:
    """Wrap the query in a bool/filter clause for metadata pre-filtering.

    Filters are applied at the database level before scoring — essential for
    document-level permission scoping.
    """
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
                section_id=    src.get("section_id", hit["_id"]),
                doc_id=        src.get("doc_id", ""),
                score=         float(hit.get("_score") or 0.0),
                document_title=src.get("document_title", ""),
                section_title= src.get("section_title", ""),
                source_path=   src.get("source_path", ""),
                order_index=   int(src.get("order_index", 0)),
                raw_text=      src.get("raw_text", ""),
                summary=       src.get("summary", ""),
                section_type=  src.get("section_type", "general"),
                metadata_tags= src.get("metadata_tags", []),
                keywords=      src.get("keywords", []),
                matched_by=    (search_type,),
            )
        )
    return results
