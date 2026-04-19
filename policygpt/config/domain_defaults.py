"""Domain-specific Config overrides.

Each entry maps a domain_type string to a dict of Config field overrides that
are applied in Config.__post_init__ after all other preset/model scaling.
This keeps infrastructure sizing decisions (token budgets, retrieval limits)
separate from the LLM prompt and entity settings that live in core/domain/.

To add a new domain override:
  1. Add an entry below using the same domain_type key registered in core/domain/.
  2. Only list fields that differ from the base Config defaults.
  3. chat_max_output_tokens is a hard minimum — if the model floor (e.g. 120b = 6000)
     already exceeds this value, the higher value wins.
"""

from typing import Any


DOMAIN_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "policy": {
        # Enterprise policy answers are often lengthy — procedures, eligibility
        # tables, multi-step approval chains.  Set a high minimum so the model
        # has room to give complete, well-structured answers.
        "chat_max_output_tokens": 5400,

        # Policy folders often contain PDFs, PPTX decks, and Word docs alongside
        # HTML files — convert them all to HTML so the table-aware extractor
        # handles structured content correctly.
        "to_html_enabled": True,

        # Policy HTML documents frequently contain scanned images, org charts,
        # and approval matrices — OCR these so their text is searchable.
        "ocr_enabled": True,
        "ocr_min_confidence": 80.0,

        # Hybrid search weights for policy domain.
        # Policy queries tend to use exact defined terms (clause numbers, names)
        # so keyword search gets a stronger share than the default.
        "hybrid_keyword_weight": 0.50,
        "hybrid_similarity_weight": 0.15,
        "hybrid_vector_weight": 0.35,
    },
    "contest": {
        # Contest answers are focused and concise — thresholds, reward amounts,
        # timelines.  A moderate budget is sufficient.
        "chat_max_output_tokens": 2700,

        # Contest documents are PDFs (and may include PPTX/DOCX) — convert to
        # HTML first so the table-aware extractor can parse reward slabs and
        # eligibility criteria as structured rows/columns.
        # Converted files go to {debug_log_dir}/html/.
        "to_html_enabled": True,

        # After PDF→HTML conversion the PolicyRewriter runs to add metadata
        # block, classification banner, and TOC.  ISO/RBI regulatory tags are
        # not injected (no matching keywords in contest docs) but the structural
        # additions still improve search quality.  Files saved to
        # {debug_log_dir}/improved/ and cached for subsequent re-ingestions.
        "rewrite_policies_enabled": True,

        # Contest documents also embed reward tables and eligibility charts
        # as images — OCR them for complete coverage.
        "ocr_enabled": True,
        "ocr_min_confidence": 80.0,

        # Contest queries are often vague ("what do I win") so semantic vector
        # search should dominate.
        "hybrid_keyword_weight": 0.30,
        "hybrid_similarity_weight": 0.15,
        "hybrid_vector_weight": 0.55,
    },
    "product_technical": {
        # Technical answers often include multi-step procedures, architecture
        # descriptions, and configuration tables — allow a generous output budget.
        "chat_max_output_tokens": 5400,

        # Source documents include PDFs (design docs), PPTX (architecture decks),
        # DOCX (runbooks) — convert them all to HTML so the table-aware extractor
        # handles structured content correctly.
        "to_html_enabled": True,

        # Per-format conversion flags.
        # Excel is disabled — infrastructure-sizing workbooks can produce 6 MB+
        # HTML per sheet which hangs the extractor.  Re-enable once the extractor
        # handles very large tables efficiently.
        "pdf_to_html_enabled": True,
        "docx_to_html_enabled": True,
        "pptx_to_html_enabled": True,
        "excel_to_html_enabled": True,
        "image_to_html_enabled": True,

        # OCR is off by default — enable only when AWS Textract credentials are
        # available and the documents contain diagrams or screenshots.
        # "ocr_enabled": True,
        # "ocr_min_confidence": 75.0,

        # Technical queries use precise terminology (service names, config keys,
        # CLI flags) — keyword search gets a strong weight to match exact terms
        # while vector search covers paraphrased or high-level questions.
        "hybrid_keyword_weight": 0.60,
        "hybrid_similarity_weight": 0.10,
        "hybrid_vector_weight": 0.30,
    },
}
