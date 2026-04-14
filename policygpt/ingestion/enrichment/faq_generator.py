"""FaqGenerator — generates Q&A pairs and embeds them for fast-path lookup.

Delegates to DocumentCorpus._generate_document_faq and _parse_faq_qa_pairs
so the prompt engineering is not duplicated.  Embeddings are produced via
the corpus AI service.

Fills in:
- EnrichedDocument.faq_qa_pairs
- EnrichedDocument.faq_q_embeddings
"""

from __future__ import annotations

import logging
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

from policygpt.ingestion.enrichment.base import EnrichedDocument, Enricher
from policygpt.ingestion.readers.base import IngestMessage

if TYPE_CHECKING:
    from policygpt.core.corpus import DocumentCorpus

logger = logging.getLogger(__name__)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


class FaqGenerator(Enricher):
    """Generates an LLM FAQ, parses Q/A pairs, and embeds each question.

    Only runs when config.generate_faq=True and config.faq_fastpath_enabled=True.
    If either flag is off the enricher is a no-op so the pipeline does not need
    to be re-wired.

    Parameters
    ----------
    corpus:
        DocumentCorpus instance whose FAQ generation methods are reused.
    """

    def __init__(self, corpus: "DocumentCorpus") -> None:
        self._corpus = corpus

    def enrich(self, doc: EnrichedDocument, message: IngestMessage) -> EnrichedDocument:
        config = self._corpus.config
        if not config.generate_faq:
            return doc

        masked_title = self._corpus.redactor.mask_text(doc.extracted.title)
        full_text = "\n\n".join(t for _, t in doc.extracted.sections).strip()
        masked_full_text = self._corpus.redactor.mask_text(full_text)

        try:
            faq_text = self._corpus._generate_document_faq(
                masked_title=masked_title,
                masked_text=masked_full_text,
            )
        except Exception as exc:
            logger.warning("FAQ generation failed for %s: %s", message.source_path, exc)
            return doc

        if not faq_text:
            return doc

        # Write FAQ file for inspection
        try:
            from pathlib import Path
            stem = Path(message.file_name).stem
            self._corpus._write_faq_file(message.source_path, message.file_name, faq_text)
        except Exception:
            pass  # debug output is non-critical

        if not config.faq_fastpath_enabled:
            return doc

        qa_pairs = self._corpus._parse_faq_qa_pairs(faq_text)
        if not qa_pairs:
            return doc

        # Embed "Q: ... A: ..." combined text so answer content is captured
        combined_texts = [f"Q: {q}\nA: {a}" for q, a in qa_pairs]
        try:
            raw_embeddings = self._corpus.ai.embed_texts(combined_texts)
            doc.faq_qa_pairs = qa_pairs
            doc.faq_q_embeddings = [_l2_normalize(e) for e in raw_embeddings]
        except Exception as exc:
            logger.warning("FAQ embedding failed for %s: %s", message.source_path, exc)

        return doc
