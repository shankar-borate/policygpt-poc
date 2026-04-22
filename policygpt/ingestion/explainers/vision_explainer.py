"""VisionPageExplainer — explains image-dominant units using the vision model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from policygpt.ingestion.explainers.base import DocumentContext, PageExplainer, UnitContent

if TYPE_CHECKING:
    from policygpt.core.ai.base import AIService
    from policygpt.ingestion.converters.vision import VisionDescriber
    from policygpt.ingestion.extraction.ocr import OcrExtractor

logger = logging.getLogger(__name__)


def _build_context_prefix(ctx: DocumentContext | None, unit: UnitContent) -> str:
    if ctx is None:
        return ""
    lines = [
        f"Document: {ctx.title} ({ctx.doc_type}, {ctx.total_units} {unit.unit_label}s total)",
        f"Purpose: {ctx.summary}",
    ]
    if unit.prev_explanation:
        lines.append(f"Previous {unit.unit_label}: {unit.prev_explanation}")
    lines.append("")
    return "\n".join(lines)


class VisionPageExplainer(PageExplainer):
    """Explains image-dominant units via the vision model with document context.

    Falls back to OCR → text LLM when vision is unavailable or returns empty.

    Parameters
    ----------
    vision:
        VisionDescriber instance (Claude or OpenAI).  Its describe_page() is
        called with a context_prefix prepended to the standard prompt.
    ocr:
        Optional OCR fallback — extracted text is then passed to the text
        explainer for a contextual LLM explanation.
    ai:
        AI service used by the text explainer in the OCR fallback path.
    """

    def __init__(
        self,
        vision: "VisionDescriber",
        ocr: "OcrExtractor | None" = None,
        ai: "AIService | None" = None,
    ) -> None:
        self._vision = vision
        self._ocr = ocr
        self._ai = ai

    def explain(self, unit: UnitContent, ctx: DocumentContext | None) -> str:
        context_prefix = _build_context_prefix(ctx, unit)

        # 1. Vision with doc context injected into the prompt
        html_fragment = self._vision.describe_page(
            unit.image_bytes,
            unit.mime_type or "image/png",
            context_prefix=context_prefix,
        )
        if html_fragment.strip():
            logger.info(
                "VisionPageExplainer: %s %d — vision OK (%d chars)",
                unit.unit_label, unit.unit_index, len(html_fragment),
            )
            return (
                f'<div class="unit-explanation" data-source="vision">\n'
                f"{html_fragment}\n"
                f"</div>"
            )

        # 2. OCR → text LLM fallback
        if self._ocr is not None:
            ocr_text = self._ocr.extract_from_bytes(
                unit.image_bytes, unit.mime_type or "image/png"
            )
            if ocr_text.strip() and self._ai is not None:
                logger.info(
                    "VisionPageExplainer: %s %d — OCR fallback, %d chars",
                    unit.unit_label, unit.unit_index, len(ocr_text),
                )
                from policygpt.ingestion.explainers.text_explainer import TextPageExplainer
                ocr_unit = UnitContent(
                    unit_index=unit.unit_index,
                    unit_label=unit.unit_label,
                    text=ocr_text,
                    prev_explanation=unit.prev_explanation,
                )
                return TextPageExplainer(self._ai).explain(ocr_unit, ctx)

        logger.warning(
            "VisionPageExplainer: %s %d — no output from vision or OCR",
            unit.unit_label, unit.unit_index,
        )
        return ""
