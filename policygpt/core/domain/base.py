"""Domain profile system — defines all domain-specific prompt text and entity
settings in one place.  Add a new file under core/domain/ to support a new
knowledge-base type; update Config.domain_type to switch between them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    # ── Core ──────────────────────────────────────────────────────────────────
    # Injected into every LLM system prompt (summarisation, entity extraction,
    # FAQ generation, query understanding, answer generation).
    domain_context: str

    # ── Bot: social / conversational replies ──────────────────────────────────
    persona_description: str    # used in social-reply system prompt
    greeting_reply: str         # canned fallback when LLM call fails
    identity_reply: str         # canned fallback for identity questions

    # ── Bot: intent classifier ────────────────────────────────────────────────
    intent_user_description: str     # who uses the system
    intent_policy_description: str   # what counts as a "policy" question

    # ── Bot: open-weight system prompt ────────────────────────────────────────
    doc_type_label: str   # e.g. "contest policy documents" / "policy documents"

    # ── Corpus: document-level summary prompts ────────────────────────────────
    doc_summary_focus: str          # _summarize_document_text
    chunk_summary_capture: str      # _summarize_document_chunk
    combine_summary_retain: str     # _combine_document_summaries
    finalize_summary_focus: str     # _finalize_document_summary

    # ── Corpus: section-level summary prompts ─────────────────────────────────
    section_summary_capture: str    # _summarize_section_text / _finalize_section_summary
    section_combine_preserve: str   # _combine_section_summaries
    user_label: str                 # "sales agent" | "employee" etc.

    # ── Corpus: FAQ generation ────────────────────────────────────────────────
    faq_cover: str   # topics to cover in FAQ Q&A generation

    # ── Bot: aggregate / listing queries ──────────────────────────────────────
    aggregate_response_hint: str
    aggregate_positive_markers: tuple

    # ── Entity extraction ─────────────────────────────────────────────────────
    entity_categories: frozenset          # recognised category names
    entity_global_categories: frozenset   # categories applied document-wide
    entity_extraction_rules: str          # bullet rules injected into extraction prompt
    entity_examples: str                  # example JSON block for few-shot guidance

    # ── Web UI ────────────────────────────────────────────────────────────────
    # Returned by /api/domain so the frontend is fully server-driven.
    ui_assistant_label: str              # header eyebrow, e.g. "Enterprise Policy Assistant"
    ui_eyebrow: str                      # hero card eyebrow, e.g. "Ask policy questions"
    ui_description: str                  # hero card subtitle
    ui_prompt_chips: tuple               # ((label, full_prompt), ...) shown as quick-start chips
    ui_sidebar_title: str                # sidebar <h1>, e.g. "Chat with contest docs"
    ui_sidebar_subtitle: str             # sidebar muted subtitle
    ui_search_placeholder: str           # search input placeholder
    ui_input_placeholder: str            # composer textarea placeholder


_REGISTRY: dict[str, DomainProfile] = {}


def register(name: str, profile: DomainProfile) -> None:
    """Register a domain profile under *name*."""
    _REGISTRY[name] = profile


def get_domain_profile(domain_type: str) -> DomainProfile:
    """Return the profile for *domain_type*, raising a clear error if unknown."""
    if domain_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown domain_type '{domain_type}'. "
            f"Available: {available}. "
            "Check the domain_type setting in Config."
        )
    return _REGISTRY[domain_type]
