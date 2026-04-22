"""ExplainerFactory — central coordinator for the explainability feature.

Owns the rules, context builder, and explainer instances.
Converters call explain_unit() per slide/page/sheet and build_context()
once per document (pass 1) for formats that need document context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from policygpt.ingestion.explainers.base import DocumentContext, UnitContent
from policygpt.ingestion.explainers.context_builder import DocumentContextBuilder
from policygpt.ingestion.explainers.rules import ExplainRules

if TYPE_CHECKING:
    from policygpt.core.ai.base import AIService
    from policygpt.ingestion.converters.vision import VisionDescriber
    from policygpt.ingestion.extraction.ocr import OcrExtractor

logger = logging.getLogger(__name__)


class ExplainerFactory:
    """Coordinates rules, context building, and per-unit explanation.

    Parameters
    ----------
    rules:
        ExplainRules instance (threshold + mode selection).
    vision:
        Optional VisionDescriber for image-dominant units.
    ocr:
        Optional OCR fallback used inside VisionPageExplainer.
    ai:
        AI service for text LLM calls (context builder + TextPageExplainer).
    """

    def __init__(
        self,
        rules: ExplainRules,
        vision: "VisionDescriber | None" = None,
        ocr: "OcrExtractor | None" = None,
        ai: "AIService | None" = None,
    ) -> None:
        self._rules = rules
        self._context_builder = DocumentContextBuilder(ai) if ai else None

        self._vision_explainer = None
        self._text_explainer = None

        if vision is not None:
            from policygpt.ingestion.explainers.vision_explainer import VisionPageExplainer
            self._vision_explainer = VisionPageExplainer(vision, ocr, ai)

        if ai is not None:
            from policygpt.ingestion.explainers.text_explainer import TextPageExplainer
            self._text_explainer = TextPageExplainer(ai)

    # ── Pass 1 ────────────────────────────────────────────────────────────────

    def build_context(
        self,
        unit_texts: list[str],
        doc_type: str,
        title: str,
    ) -> DocumentContext | None:
        """Build document context from all unit texts (pass 1).

        Returns None when no AI service is available.
        """
        if self._context_builder is None:
            return None
        return self._context_builder.build(unit_texts, doc_type, title)

    # ── Pass 2 ────────────────────────────────────────────────────────────────

    def explain_unit(
        self,
        unit: UnitContent,
        ctx: DocumentContext | None,
    ) -> str:
        """Return an HTML explanation fragment for one unit, or '' if not needed."""
        if not self._rules.should_explain(unit):
            return ""

        mode = self._rules.explainer_mode(unit)

        if mode == "vision":
            if self._vision_explainer is not None:
                return self._vision_explainer.explain(unit, ctx)
            # Vision unavailable — fall through to text if we have text
            if unit.char_count > 0 and self._text_explainer is not None:
                return self._text_explainer.explain(unit, ctx)
            return ""

        # mode == "text"
        if self._text_explainer is not None:
            return self._text_explainer.explain(unit, ctx)

        return ""
