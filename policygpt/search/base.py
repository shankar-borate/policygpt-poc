"""VectorStore — the provider-agnostic Port.

Any vector database backend (OpenSearch, Pinecone, Weaviate, pgvector …)
must implement this interface.  The rest of the application depends only on
this ABC; no provider SDK ever leaks outside its own package.

To add a new provider:
  1. Create  policygpt/search/providers/<name>/store.py
  2. Subclass VectorStore and implement all abstract methods
  3. Register the provider key in  policygpt/search/factory.py
  That's it — zero changes to corpus.py, bot.py, or config logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from policygpt.models.documents import DocumentRecord
from policygpt.search.models import FaqResult, SearchQuery, SearchResult


class VectorStore(ABC):
    """Abstract vector store interface (Port in hexagonal architecture)."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def ensure_index(self, embedding_dim: int) -> None:
        """Create the backing index / collection if it does not already exist.

        Safe to call repeatedly — must be idempotent.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the store is reachable and accepting requests."""

    # ── Write ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def index_document(
        self,
        document: DocumentRecord,
        user_ids: list[str | int],
        domain: str,
    ) -> None:
        """Upsert a document and all of its sections into the store.

        user_ids and domain are passed in by the caller (not derived from the
        document itself) so the ingestion pipeline stays generic and the same
        method can be called from a folder scan, an SQS consumer, or an API
        upload without any changes here.

        Re-indexing an existing document must overwrite without duplicating.
        """

    @abstractmethod
    def get_cached_document(self, source_path: str) -> dict | None:
        """Return stored document + section metadata for source_path, or None.

        Used by the ingestion pipeline to rebuild in-memory structures from
        OpenSearch when a document was already indexed in a previous run —
        avoiding expensive LLM re-processing while still populating the
        in-memory corpus so chat and search both work correctly.

        Returned dict shape:
            {
              doc_id, title, source_path, summary, version, effective_date,
              document_type, metadata_tags, audiences, keywords,
              sections: [{section_id, title, source_path, order_index,
                          section_type, metadata_tags, keywords, summary,
                          raw_text, masked_text}]
            }
        """
        raise NotImplementedError

    def document_indexed_for_path(self, source_path: str) -> bool:
        """Return True if a document with this source_path is already indexed.

        Used by the ingestion pipeline to skip expensive LLM processing for
        documents that have not changed since the last run.
        """

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """Remove a document, all its sections, and all its FAQ pairs."""

    @abstractmethod
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
        """Index FAQ Q/A pairs for a document.

        Each pair gets its own document in the FAQ index with the question
        embedding stored for fast vector similarity lookup.
        user_ids and domain are inherited from the parent document.
        """

    @abstractmethod
    def faq_search(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        min_score: float = 0.92,
    ) -> FaqResult | None:
        """Return the best-matching FAQ answer if score >= min_score, else None.

        Applies user_id permission filter before scoring — only FAQ entries
        the user is allowed to see are considered.
        """

    @abstractmethod
    def search_faq_questions(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        top_k: int = 30,
    ) -> list[FaqResult]:
        """Return top-k FAQ results scored against query_embedding, filtered by user_id.

        Unlike faq_search() this returns many results (not just the best one)
        for use in aggregate queries and related-question suggestions.
        """

    # ── Read (three independent search strategies) ────────────────────────────

    def search_documents(
        self,
        query_text: str,
        user_id: str,
        page: int = 1,
        size: int = 10,
    ) -> dict:
        """Full-text document search returning one result per document.

        Groups section hits by document and returns the top-scoring section
        as the representative snippet for each document.

        Returns a dict:
            {total, page, size, results: [{document_title, section_title,
            snippet, source_path, section_index, score}]}

        Raises NotImplementedError if the backend does not support this.
        """
        raise NotImplementedError("search_documents is not supported by this vector store")

    @abstractmethod
    def keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        """BM25 full-text search on text fields.

        Should boost title and keyword fields over body text.
        Should support fuzzy matching for typo tolerance.
        """

    @abstractmethod
    def similarity_search(self, query: SearchQuery) -> list[SearchResult]:
        """Term co-occurrence similarity (more-like-this style).

        Finds sections whose vocabulary statistically resembles the query text.
        Distinct from keyword search (no exact term requirement) and from
        vector search (no embedding involved).
        """

    @abstractmethod
    def vector_search(self, query: SearchQuery) -> list[SearchResult]:
        """Dense kNN vector search on the stored embedding field.

        Returns semantically similar sections regardless of exact wording.
        query.embedding must be provided; return [] if it is None.
        """
