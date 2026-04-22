"""DocumentContextBuilder — pass 1 of the explainability pipeline.

Scans all unit texts from a document and produces a DocumentContext
that is fed into every per-unit explanation in pass 2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from policygpt.ingestion.explainers.base import DocumentContext

if TYPE_CHECKING:
    from policygpt.core.ai.base import AIService

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = """\
Summarize the following document in 2-3 sentences.
Cover: what the document is about, its purpose, and the key topics it addresses.
Be specific — include domain, subject matter, and audience if apparent.
Output ONLY the summary text, no headings or labels.

Document title: {title}
Document type: {doc_type}
Content excerpt:
{excerpt}
"""

_EXCERPT_CHARS_PER_UNIT = 300
_MAX_EXCERPT_CHARS = 3000


class DocumentContextBuilder:
    """Builds a DocumentContext from a list of unit texts.

    Parameters
    ----------
    ai:
        AI service used for the summarization LLM call.
    max_tokens:
        Max output tokens for the summary (keep short — 2-3 sentences).
    """

    def __init__(self, ai: "AIService", max_tokens: int = 300) -> None:
        self._ai = ai
        self._max_tokens = max_tokens

    def build(
        self,
        unit_texts: list[str],
        doc_type: str,
        title: str,
    ) -> DocumentContext:
        total = len(unit_texts)
        excerpt = self._build_excerpt(unit_texts)
        summary = self._summarize(title, doc_type, excerpt)
        logger.info(
            "DocumentContextBuilder: built context for %r (%s, %d units)", title, doc_type, total
        )
        return DocumentContext(
            title=title,
            doc_type=doc_type,
            summary=summary,
            total_units=total,
        )

    def _build_excerpt(self, unit_texts: list[str]) -> str:
        parts: list[str] = []
        total_chars = 0
        for text in unit_texts:
            chunk = text.strip()[:_EXCERPT_CHARS_PER_UNIT]
            if not chunk:
                continue
            parts.append(chunk)
            total_chars += len(chunk)
            if total_chars >= _MAX_EXCERPT_CHARS:
                break
        return "\n\n".join(parts)

    def _summarize(self, title: str, doc_type: str, excerpt: str) -> str:
        if not excerpt.strip():
            return title
        try:
            prompt = _SUMMARIZE_PROMPT.format(
                title=title, doc_type=doc_type, excerpt=excerpt
            )
            return self._ai.llm_text(
                system_prompt="You are a document analyst. Be concise and factual.",
                user_prompt=prompt,
                max_output_tokens=self._max_tokens,
            ).strip()
        except Exception as exc:
            logger.warning("DocumentContextBuilder: summarize failed — %s", exc)
            return title
