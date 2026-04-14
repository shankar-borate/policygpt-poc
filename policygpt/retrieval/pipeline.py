"""RetrievalPipeline — orchestrates query analysis, retrieval, and answer generation.

Flow per query
--------------
user_question
  → QueryAnalyzer          → QueryAnalysis (intents, focus terms, …)
    → FAQ fast-path        → early return if cosine ≥ 0.92 on FAQ embeddings
      → embed question
        → DocumentCorpus.retrieve_top_docs    → candidate documents
          → DocumentCorpus.retrieve_top_sections → candidate sections
            → Reranker.rerank + select_diverse  → final sections
              → PolicyGPTBot._build_answer       → answer + sources

Current state
-------------
The answer-generation and FAQ fast-path live in PolicyGPTBot.chat().  This
pipeline wraps the retrieval-only part (up to and including final section
selection) and can be composed with the chat layer.  Full decoupling of
answer generation from bot.py is a follow-up task.

Public API
----------
pipeline = RetrievalPipeline.from_bot(bot)
sections, analysis = pipeline.retrieve(
    question="...", user_id="100", preferred_section_ids=[...]
)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from policygpt.core.retrieval.query_analyzer import QueryAnalysis, QueryAnalyzer
from policygpt.models import SectionRecord
from policygpt.retrieval.reranker import Reranker

if TYPE_CHECKING:
    from policygpt.core.corpus import DocumentCorpus

logger = logging.getLogger(__name__)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


class RetrievalPipeline:
    """Wraps the full retrieval stack (analyze → embed → rank → diversify).

    Parameters
    ----------
    corpus:
        DocumentCorpus that owns the search indexes and ranking logic.
    query_analyzer:
        QueryAnalyzer instance (stateless; can be shared).
    reranker:
        Reranker wrapping corpus ranking methods.
    """

    def __init__(
        self,
        corpus: "DocumentCorpus",
        query_analyzer: QueryAnalyzer,
        reranker: Reranker,
    ) -> None:
        self._corpus = corpus
        self._query_analyzer = query_analyzer
        self._reranker = reranker

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_corpus(cls, corpus: "DocumentCorpus") -> "RetrievalPipeline":
        """Build a RetrievalPipeline from a DocumentCorpus."""
        return cls(
            corpus=corpus,
            query_analyzer=QueryAnalyzer(),
            reranker=Reranker(corpus),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        user_id: str | int | None = None,
        preferred_section_ids: list[str] | None = None,
        preferred_doc_ids: list[str] | None = None,
    ) -> tuple[list[tuple[SectionRecord, float]], QueryAnalysis]:
        """Retrieve the best sections for *question*.

        Parameters
        ----------
        question:
            Raw user question (un-analysed).
        user_id:
            Required when hybrid_search_enabled=True (OpenSearch access control).
        preferred_section_ids:
            Section IDs from prior conversation turns (boosted by +0.08).
        preferred_doc_ids:
            Document IDs preferred for this query (e.g. from explicit citations).

        Returns
        -------
        (sections, analysis)
            sections — list of (SectionRecord, score) sorted descending.
            analysis — QueryAnalysis with intents, focus terms, etc.
        """
        corpus = self._corpus
        config = corpus.config

        # 1. Analyse
        analysis = self._query_analyzer.analyze(
            question=question,
            documents=corpus.documents,
            sections=corpus.sections,
            entity_lookup=corpus.entity_lookup,
        )
        retrieval_query = analysis.retrieval_query or question

        # 2. Embed
        raw_vec = corpus.ai.embed_text(retrieval_query)
        query_vec = _l2_normalize(np.array(raw_vec, dtype=np.float32))

        # 3. Retrieve candidate documents
        top_docs = corpus.retrieve_top_docs(
            query_vec=query_vec,
            query_analysis=analysis,
            preferred_doc_ids=preferred_doc_ids,
        )
        if not top_docs:
            return [], analysis

        # 4. Retrieve candidate sections (includes OS path when enabled)
        raw_sections = corpus.retrieve_top_sections(
            query_vec=query_vec,
            query_analysis=analysis,
            top_docs=top_docs,
            preferred_section_ids=preferred_section_ids,
            user_id=user_id,
        )

        return raw_sections, analysis

    def faq_fastpath(
        self,
        question: str,
        user_id: str | int | None = None,
    ) -> str | None:
        """Check the FAQ fast-path.  Returns answer string or None.

        Delegates to DocumentCorpus.faq_fastpath_lookup which checks both the
        OS index (when hybrid search is enabled) and the in-memory FAQ store.
        """
        return self._corpus.faq_fastpath_lookup(
            question=question,
            user_id=user_id,
        )
