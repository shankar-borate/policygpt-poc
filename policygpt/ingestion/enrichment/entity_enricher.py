"""EntityEnricher — extracts named entities and their contextual meanings.

Thin wrapper around EntityExtractor.  The result is an DocumentEntityMap
stored in EnrichedDocument.entity_map so downstream steps (section tagging,
entity lookup at retrieval time) can use it.

Fills in:
- EnrichedDocument.entity_map
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from policygpt.ingestion.enrichment.base import EnrichedDocument, Enricher
from policygpt.ingestion.readers.base import IngestMessage

if TYPE_CHECKING:
    from policygpt.core.corpus import DocumentCorpus

logger = logging.getLogger(__name__)


class EntityEnricher(Enricher):
    """Extracts named entities with categories, context, and synonyms.

    Only runs when config.ingestion.generate_entity_map=True.

    Parameters
    ----------
    corpus:
        DocumentCorpus instance whose EntityExtractor and config are reused.
    """

    def __init__(self, corpus: "DocumentCorpus") -> None:
        self._corpus = corpus

    def enrich(self, doc: EnrichedDocument, message: IngestMessage) -> EnrichedDocument:
        if not self._corpus.config.ingestion.generate_entity_map:
            return doc

        masked_title = self._corpus.redactor.mask_text(doc.extracted.title)
        full_text = "\n\n".join(t for _, t in doc.extracted.sections).strip()
        masked_full_text = self._corpus.redactor.mask_text(full_text)

        config = self._corpus.config
        try:
            doc.entity_map = self._corpus.entity_extractor.extract(
                title=masked_title,
                masked_text=masked_full_text,
                max_output_tokens=config.ingestion.entity_map_max_output_tokens,
                char_budget=max(4000, config.ingestion.doc_summary_input_token_budget * 2),
            )
        except Exception as exc:
            logger.warning("Entity extraction failed for %s: %s", message.source_path, exc)

        if doc.entity_map.entities:
            try:
                self._corpus._write_entity_file(
                    message.source_path, message.file_name, doc.entity_map
                )
            except Exception:
                pass  # debug output is non-critical

        return doc
