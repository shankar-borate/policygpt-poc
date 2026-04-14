"""Reranker — wraps corpus section-reranking and diversity selection.

The actual reranking logic lives in DocumentCorpus._rerank_sections and
_select_diverse_sections.  This thin wrapper exposes them as a standalone
component so the retrieval pipeline can call them without holding a reference
to the full DocumentCorpus.

When the logic is eventually extracted from corpus.py it can be moved here
directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from policygpt.core.retrieval.query_analyzer import QueryAnalysis
from policygpt.models import SectionRecord

if TYPE_CHECKING:
    from policygpt.core.corpus import DocumentCorpus

logger = logging.getLogger(__name__)


class Reranker:
    """Re-scores and diversifies candidate sections for a query.

    Parameters
    ----------
    corpus:
        DocumentCorpus whose private _rerank_sections and _select_diverse_sections
        methods are delegated to.
    """

    def __init__(self, corpus: "DocumentCorpus") -> None:
        self._corpus = corpus

    def rerank(
        self,
        query_analysis: QueryAnalysis,
        candidates: list[tuple[SectionRecord, float]],
    ) -> list[tuple[SectionRecord, float]]:
        """Apply heuristic + LLM reranking to *candidates*.

        Parameters
        ----------
        query_analysis:
            Parsed query metadata (intents, focus terms, …).
        candidates:
            List of (section, initial_score) pairs.  Order does not matter —
            scores will be re-computed.

        Returns
        -------
        List of (section, final_score) sorted descending.
        """
        reranked = self._corpus._rerank_sections(
            query_analysis=query_analysis,
            candidate_sections=candidates,
        )
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked

    def select_diverse(
        self,
        query_analysis: QueryAnalysis,
        scored_sections: list[tuple[SectionRecord, float]],
        limit: int | None = None,
    ) -> list[tuple[SectionRecord, float]]:
        """Apply diversity selection after reranking.

        Parameters
        ----------
        limit:
            Override for the result limit.  When None the corpus config
            heuristic (_section_result_limit_for_query) is applied.
        """
        effective_limit = limit or self._corpus._section_result_limit_for_query(query_analysis)
        return self._corpus._select_diverse_sections(
            query_analysis=query_analysis,
            scored_sections=scored_sections,
            limit=effective_limit,
        )
