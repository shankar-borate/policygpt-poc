"""Summarizer — generates document-level and section-level LLM summaries.

Delegates to the proven summarization logic inside DocumentCorpus
(_create_document_summary, _create_section_summary) so the prompt engineering
and recursive-chunk strategy are not duplicated.

When corpus is eventually broken apart, move the private methods here directly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from policygpt.ingestion.enrichment.base import EnrichedDocument, Enricher
from policygpt.ingestion.readers.base import IngestMessage

if TYPE_CHECKING:
    from policygpt.core.corpus import DocumentCorpus

logger = logging.getLogger(__name__)


class Summarizer(Enricher):
    """Generates LLM summaries for a document and each of its sections.

    Fills in:
    - EnrichedDocument.document_summary
    - EnrichedDocument.section_summaries   (parallel to extracted.sections)

    Parameters
    ----------
    corpus:
        DocumentCorpus instance whose private summarization methods are reused.
        This dependency will be removed once those methods are promoted to this
        module.
    """

    def __init__(self, corpus: "DocumentCorpus") -> None:
        self._corpus = corpus

    def enrich(self, doc: EnrichedDocument, message: IngestMessage) -> EnrichedDocument:
        masked_title = self._corpus.redactor.mask_text(doc.extracted.title)
        full_text = "\n\n".join(t for _, t in doc.extracted.sections).strip()
        masked_full_text = self._corpus.redactor.mask_text(full_text)

        # ── Document summary ───────────────────────────────────────────────
        try:
            doc.document_summary = self._corpus._create_document_summary(
                masked_title=masked_title,
                masked_text=masked_full_text,
            )
        except Exception as exc:
            logger.warning("Document summary failed for %s: %s", message.source_path, exc)
            doc.document_summary = masked_title  # minimal fallback

        # ── Section summaries ──────────────────────────────────────────────
        config = self._corpus.config
        doc.section_summaries = []
        for section_title, section_text in doc.extracted.sections:
            masked_section_title = self._corpus.redactor.mask_text(section_title)
            masked_section_text = self._corpus.redactor.mask_text(section_text)
            try:
                if config.skip_section_summary:
                    summary = self._corpus._build_fallback_section_summary(
                        section_title=masked_section_title,
                        section_text=masked_section_text,
                    )
                else:
                    summary = self._corpus._create_section_summary(
                        doc_title=masked_title,
                        section_title=masked_section_title,
                        section_text=masked_section_text,
                    )
            except Exception as exc:
                logger.warning(
                    "Section summary failed for %s / %s: %s",
                    message.source_path, section_title, exc,
                )
                compact = re.sub(r"\s+", " ", masked_section_text.strip())[:500]
                summary = f"{masked_section_title}: {compact}"
            doc.section_summaries.append(summary)

        return doc
