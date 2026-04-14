"""HybridSearcher — provider-agnostic hybrid retrieval orchestrator.

Runs keyword, similarity, and vector searches concurrently against any
VectorStore implementation, then combines the results using a configurable
weighted blend.  The weights are defined in Config and can be overridden
per domain in domain_defaults.py.

Score normalisation strategy
─────────────────────────────
Each search type returns raw scores on different scales (BM25 scores are
unbounded; cosine similarity scores are in [0, 1]).  Before blending we
min-max normalise each result set independently so all three live in [0, 1].

Blend formula
──────────────
  final_score = w_keyword    * norm_keyword_score
              + w_similarity * norm_similarity_score
              + w_vector     * norm_vector_score

Sections that appear in only some of the result sets receive 0 for the
missing types, so they are not unfairly penalised — the weights naturally
govern how much each type matters.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections import defaultdict

from policygpt.config import Config
from policygpt.search.base import VectorStore
from policygpt.search.models import SearchQuery, SearchResult, SearchType

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Orchestrates the three search strategies and blends their scores.

    This class is provider-agnostic: it only depends on the VectorStore
    interface, never on any specific backend SDK.
    """

    def __init__(self, store: VectorStore, config: Config) -> None:
        self.store = store
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Run enabled search types concurrently and return blended results."""
        weights = self._active_weights(query.search_types)
        if not weights:
            return []

        results_by_type = self._run_concurrent(query, weights)
        return self._blend(results_by_type, weights, query.top_k)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _active_weights(
        self, search_types: tuple[SearchType, ...]
    ) -> dict[SearchType, float]:
        """Return the weight for each requested + configured search type."""
        all_weights = {
            SearchType.KEYWORD:    self.config.hybrid_keyword_weight,
            SearchType.SIMILARITY: self.config.hybrid_similarity_weight,
            SearchType.VECTOR:     self.config.hybrid_vector_weight,
        }
        return {
            st: w
            for st, w in all_weights.items()
            if st in search_types and w > 0.0
        }

    def _run_concurrent(
        self,
        query: SearchQuery,
        weights: dict[SearchType, float],
    ) -> dict[SearchType, list[SearchResult]]:
        """Dispatch search calls concurrently; log and absorb individual failures."""
        dispatch = {
            SearchType.KEYWORD:    self.store.keyword_search,
            SearchType.SIMILARITY: self.store.similarity_search,
            SearchType.VECTOR:     self.store.vector_search,
        }

        results: dict[SearchType, list[SearchResult]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(dispatch[st], query): st
                for st in weights
            }
            for future in concurrent.futures.as_completed(futures):
                st = futures[future]
                try:
                    results[st] = future.result(timeout=15)
                except Exception as exc:
                    logger.warning("Search type %s failed: %s", st.value, exc)
                    results[st] = []

        return results

    @staticmethod
    def _normalise(results: list[SearchResult]) -> dict[str, float]:
        """Min-max normalise raw scores to [0, 1]; return section_id → score."""
        if not results:
            return {}
        scores = [r.score for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo or 1.0
        return {r.section_id: (r.score - lo) / span for r in results}

    def _blend(
        self,
        results_by_type: dict[SearchType, list[SearchResult]],
        weights: dict[SearchType, float],
        top_k: int,
    ) -> list[SearchResult]:
        # Normalise each type independently
        norm: dict[SearchType, dict[str, float]] = {
            st: self._normalise(results)
            for st, results in results_by_type.items()
        }

        # Collect all candidate section ids and a reference SearchResult object
        result_registry: dict[str, SearchResult] = {}
        matched_by: dict[str, set[SearchType]] = defaultdict(set)

        for st, results in results_by_type.items():
            for r in results:
                if r.section_id not in result_registry:
                    result_registry[r.section_id] = r
                matched_by[r.section_id].add(st)

        # Compute blended score for every candidate
        blended: list[tuple[str, float]] = []
        for sid in result_registry:
            score = sum(
                weights[st] * norm[st].get(sid, 0.0)
                for st in weights
            )
            blended.append((sid, score))

        blended.sort(key=lambda item: item[1], reverse=True)

        final: list[SearchResult] = []
        for sid, blended_score in blended[:top_k]:
            result = result_registry[sid]
            result.score = blended_score
            result.matched_by = tuple(matched_by[sid])
            final.append(result)

        return final
