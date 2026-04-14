"""OpenSearchRetriever — bridges the search layer back to corpus.py.

Translates SearchResult objects (provider-neutral) into the
list[tuple[SectionRecord, float]] that corpus.py and the reranker expect.

If the SectionRecord is available in the in-memory lookup (corpus.sections)
we use it directly — that keeps all the rich fields (embedding, token_counts,
etc.) intact.  If it is absent (e.g. the store has documents that were not
loaded in this process), we reconstruct a minimal SectionRecord from the
SearchResult fields so retrieval still works.
"""

from __future__ import annotations

import logging
import numpy as np

from policygpt.config import Config
from policygpt.core.retrieval.query_analyzer import QueryAnalysis
from policygpt.models.documents import SectionRecord
from policygpt.search.base import VectorStore
from policygpt.search.hybrid import HybridSearcher
from policygpt.search.models import SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class OpenSearchRetriever:
    """Retrieves sections from a VectorStore and resolves them to SectionRecords."""

    def __init__(self, store: VectorStore, config: Config) -> None:
        self.store = store
        self.config = config
        self._hybrid = HybridSearcher(store, config)

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        query_analysis: QueryAnalysis,
        section_lookup: dict[str, SectionRecord],
        user_id: str | int,
    ) -> list[tuple[SectionRecord, float]]:
        """Return top-k (SectionRecord, score) pairs using hybrid search.

        Args:
            query_text:      Raw user question text (used for keyword + similarity).
            query_embedding: Dense query embedding (used for vector search).
            top_k:           Maximum number of results to return.
            query_analysis:  Provides topic_hints for metadata tag filtering.
            section_lookup:  corpus.sections dict for in-memory resolution.
            allowed_doc_ids: Optional allowlist of doc_ids for permission-based
                             scoping. When provided only these documents are
                             searched — enforced at the DB level before scoring.
        """
        search_query = SearchQuery(
            text=query_text,
            embedding=query_embedding,
            # Over-fetch so the downstream reranker has enough candidates
            top_k=max(top_k * 2, self.config.rerank_section_candidates),
            filters=self._build_filters(user_id),
        )

        results = self._hybrid.search(search_query)

        output: list[tuple[SectionRecord, float]] = []
        for result in results:
            section = section_lookup.get(result.section_id)
            if section is None:
                section = _reconstruct_section(result)
            output.append((section, result.score))

        return output[:top_k]

    # ── Filter builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_filters(user_id: str | int) -> dict:
        """Build OpenSearch pre-filter from the request user_id.

        user_ids is a keyword array on every section. The filter requires the
        requesting user's id to be present in that array — enforced at the DB
        level before any scoring so no unauthorised sections ever surface.

        user_id is always required. Callers must supply it explicitly —
        there is no bypass path.
        """
        return {"user_ids": str(user_id)}


# ── Section reconstruction ─────────────────────────────────────────────────


def _reconstruct_section(result: SearchResult) -> SectionRecord:
    """Build a minimal SectionRecord from a SearchResult.

    Used when the section exists in OpenSearch but was not loaded into memory
    (e.g. the corpus was not re-ingested in this process).
    """
    return SectionRecord(
        section_id=result.section_id,
        doc_id=result.doc_id,
        title=result.section_title,
        raw_text=result.raw_text,
        masked_text=result.raw_text,
        summary=result.summary,
        summary_embedding=np.array([]),
        source_path=result.source_path,
        order_index=result.order_index,
        section_type=result.section_type,
        metadata_tags=result.metadata_tags,
        keywords=result.keywords,
    )
