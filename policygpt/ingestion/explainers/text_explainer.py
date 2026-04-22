"""TextPageExplainer — explains sparse-text units using the text LLM."""

from __future__ import annotations

import html as _html_lib
import logging
from typing import TYPE_CHECKING

from policygpt.ingestion.explainers.base import DocumentContext, PageExplainer, UnitContent

if TYPE_CHECKING:
    from policygpt.core.ai.base import AIService

logger = logging.getLogger(__name__)

_EXPLAIN_PROMPT = """\
{context_section}\
Analyze {unit_label} {unit_index} of this document.

Content:
{content}

Write a 2-3 sentence explanation covering:
- What this {unit_label} is about
- Key facts, figures, thresholds, or decisions it contains
- How it relates to the document topic (if context is provided above)

Output ONLY the explanation text. Be concise and factual.
"""

_CONTEXT_SECTION = """\
Document: {title} ({doc_type}, {total} {label}s total)
Purpose: {summary}
Previous {label}: {prev}

"""

_MAX_CONTENT_CHARS = 1500


class TextPageExplainer(PageExplainer):
    """Explains sparse-text units via a text LLM call with document context.

    Parameters
    ----------
    ai:
        AI service for the LLM call.
    max_tokens:
        Max tokens in the explanation response.
    """

    def __init__(self, ai: "AIService", max_tokens: int = 300) -> None:
        self._ai = ai
        self._max_tokens = max_tokens

    def explain(self, unit: UnitContent, ctx: DocumentContext | None) -> str:
        if not unit.text.strip():
            return ""
        try:
            context_section = _build_context_section(ctx, unit)
            prompt = _EXPLAIN_PROMPT.format(
                context_section=context_section,
                unit_label=unit.unit_label,
                unit_index=unit.unit_index,
                content=unit.text.strip()[:_MAX_CONTENT_CHARS],
            )
            explanation = self._ai.llm_text(
                system_prompt="You are a document analyst. Be concise and factual.",
                user_prompt=prompt,
                max_output_tokens=self._max_tokens,
            ).strip()
            if not explanation:
                return ""
            logger.info(
                "TextPageExplainer: %s %d — OK (%d chars)",
                unit.unit_label, unit.unit_index, len(explanation),
            )
            escaped = _html_lib.escape(explanation)
            return (
                f'<div class="unit-explanation" data-source="text-llm">\n'
                f"<p>{escaped}</p>\n"
                f"</div>"
            )
        except Exception as exc:
            logger.warning(
                "TextPageExplainer: %s %d — failed: %s",
                unit.unit_label, unit.unit_index, exc,
            )
            return ""


def _build_context_section(ctx: DocumentContext | None, unit: UnitContent) -> str:
    if ctx is None:
        return ""
    return _CONTEXT_SECTION.format(
        title=ctx.title,
        doc_type=ctx.doc_type,
        total=ctx.total_units,
        label=unit.unit_label,
        summary=ctx.summary,
        prev=unit.prev_explanation or "N/A",
    )
