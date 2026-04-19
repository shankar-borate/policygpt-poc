"""Retrieval and reranking configuration.

Defaults match the 'high' accuracy baseline.
Config.__post_init__ overwrites these based on accuracy_profile and
runtime_cost_profile.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    # ── Section / document counts ─────────────────────────────────────────────
    top_docs: int = 3
    top_sections_per_doc: int = 3
    max_sections_to_llm: int = 4
    rerank_section_candidates: int = 12
    exact_top_docs: int = 2
    exact_top_sections_per_doc: int = 4
    exact_max_sections_to_llm: int = 3
    exact_rerank_section_candidates: int = 10
    broad_top_docs: int = 6
    broad_top_sections_per_doc: int = 6
    broad_max_sections_to_llm: int = 8
    broad_rerank_section_candidates: int = 24

    # ── Evidence sizing ───────────────────────────────────────────────────────
    max_evidence_snippets_per_section: int = 3
    evidence_snippet_char_limit: int = 320
    embedding_raw_excerpt_chars: int = 600
    answer_context_doc_summary_char_limit: int = 260
    evidence_chunk_char_limit: int = 900
    evidence_neighboring_units: int = 1
    small_section_full_text_chars: int = 1500
    exact_answer_evidence_char_limit: int = 1600
    broad_answer_evidence_char_limit: int = 1200
    answer_evidence_block_limit_exact: int = 2
    answer_evidence_block_limit_broad: int = 2

    # ── FAQ fast-path ─────────────────────────────────────────────────────────
    faq_fastpath_enabled: bool = True
    faq_fastpath_min_score: float = 0.92
    aggregate_faq_top_k: int = 30

    # ── Scoring weights ───────────────────────────────────────────────────────
    doc_semantic_weight: float = 0.48
    doc_lexical_weight: float = 0.24
    doc_title_weight: float = 0.16
    doc_metadata_weight: float = 0.12
    section_semantic_weight: float = 0.36
    section_lexical_weight: float = 0.24
    section_parent_weight: float = 0.16
    section_title_weight: float = 0.12
    section_metadata_weight: float = 0.12

    # ── Answerability thresholds ──────────────────────────────────────────────
    answerability_min_section_score: float = 0.24
    answerability_high_confidence_score: float = 0.50
    answerability_min_support_matches: int = 1
    answerability_min_support_matches_multi_doc: int = 2
    answerability_min_exact_evidence_matches: int = 1
    exact_query_section_parent_weight_scale: float = 0.75

    # ── Grounding guard ───────────────────────────────────────────────────────
    grounding_guard_enabled: bool = True
    grounding_guard_max_output_tokens: int = 20

    # ── Confidence classification ─────────────────────────────────────────────
    confidence_high_score: float = 0.55
    confidence_medium_score: float = 0.38

    # ── Source filtering ──────────────────────────────────────────────────────
    source_score_min_scaling: float = 0.45
    related_questions_min_score: float = 0.55

    # ── Clarification ─────────────────────────────────────────────────────────
    ambiguous_query_min_length: int = 35
    followup_on_low_confidence: bool = True

    # ── Dual-answer ───────────────────────────────────────────────────────────
    dual_answer_enabled: bool = True
    dual_answer_score_ratio: float = 0.88

    # ── Score adjustments ─────────────────────────────────────────────────────
    preferred_doc_score_boost: float = 0.08
    document_lookup_score_threshold: float = 0.55
    topic_alignment_threshold: float = 0.55
    exact_score_floor_min: float = 0.55
    exact_score_floor_scale: float = 0.65

    # ── Intent classification ─────────────────────────────────────────────────
    intent_classification_max_tokens: int = 10
    clarifying_question_max_tokens: int = 60
