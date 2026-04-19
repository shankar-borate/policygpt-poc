"""Document ingestion pipeline configuration.

Defaults match the 'high' accuracy baseline where relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from policygpt.constants import OCRProvider


@dataclass
class IngestionConfig:
    # ── Access control ────────────────────────────────────────────────────────
    ingestion_user_ids: tuple[str, ...] = field(
        default_factory=lambda: ("100", "101", "102", "103", "104", "105", "106", "107", "108", "109")
    )

    # ── Redaction ─────────────────────────────────────────────────────────────
    redaction_rules: dict[str, str] = field(
        default_factory=lambda: {"Kotak": "KKK", "kotak": "KKK", "KOTAK": "KKK"}
    )

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_enabled: bool = False
    ocr_provider: OCRProvider = OCRProvider.TEXTRACT
    ocr_min_confidence: float = 80.0
    image_max_bytes: int = 1 * 1024 * 1024

    # ── Document → HTML conversion ────────────────────────────────────────────
    to_html_enabled: bool = False
    pdf_to_html_enabled: bool = True
    docx_to_html_enabled: bool = True
    pptx_to_html_enabled: bool = True
    excel_to_html_enabled: bool = True
    image_to_html_enabled: bool = True

    # ── Policy rewriting ──────────────────────────────────────────────────────
    rewrite_policies_enabled: bool = True

    # ── FAQ generation ────────────────────────────────────────────────────────
    generate_faq: bool = True
    faq_max_questions: int = 30
    faq_max_output_tokens: int = 6000

    # ── Entity extraction ─────────────────────────────────────────────────────
    generate_entity_map: bool = True
    entity_map_max_output_tokens: int = 3600

    # ── Section chunking ──────────────────────────────────────────────────────
    min_section_chars: int = 300
    target_section_chars: int = 1800
    max_section_chars: int = 3200
    token_estimate_chars_per_token: int = 4
    token_estimate_tokens_per_word: float = 1.3

    # ── Summarisation token budgets (defaults = 'high' baseline) ─────────────
    doc_summary_input_token_budget: int = 6000
    doc_summary_combine_token_budget: int = 4500
    section_summary_input_token_budget: int = 2500
    min_recursive_summary_token_budget: int = 250
    skip_section_summary: bool = False
