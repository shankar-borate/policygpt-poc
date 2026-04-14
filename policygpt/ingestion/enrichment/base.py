"""Enricher contract — EnrichedDocument dataclass and Enricher ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from policygpt.extraction.entity_extractor import DocumentEntityMap
from policygpt.ingestion.extractors.base import ExtractedDocument
from policygpt.ingestion.readers.base import IngestMessage


@dataclass
class EnrichedDocument:
    """Result of running the enrichment stage over an ExtractedDocument.

    Built up by composing one or more Enrichers in IngestionPipeline.

    Attributes
    ----------
    extracted:
        The upstream ExtractedDocument (title + raw sections).
    document_summary:
        Retrieval-oriented summary of the full document.
    section_summaries:
        Per-section summaries in the same order as extracted.sections.
    document_embedding:
        L2-normalised embedding vector for the document (used for semantic
        search at the document level).
    section_embeddings:
        Per-section embedding vectors, same order as extracted.sections.
    faq_qa_pairs:
        Parsed (question, answer) tuples from the LLM-generated FAQ.
    faq_q_embeddings:
        Embedding for each FAQ entry (L2-normalised).  Parallel to faq_qa_pairs.
    entity_map:
        Extracted entities with categories and contextual descriptions.
    """

    extracted: ExtractedDocument
    document_summary: str = ""
    section_summaries: list[str] = field(default_factory=list)
    document_embedding: np.ndarray | None = None
    section_embeddings: list[np.ndarray] = field(default_factory=list)
    faq_qa_pairs: list[tuple[str, str]] = field(default_factory=list)
    faq_q_embeddings: list[np.ndarray] = field(default_factory=list)
    entity_map: DocumentEntityMap = field(default_factory=DocumentEntityMap)


class Enricher(ABC):
    """Abstract document enricher.

    Each Enricher adds one kind of metadata (summary, FAQ, entities, …)
    to an EnrichedDocument.  Enrichers are stateless and run in sequence;
    later enrichers can read fields set by earlier ones.

    Implementing a new enricher:
      1. Subclass Enricher and implement enrich().
      2. Add it to the pipeline's enricher list.
      Done — nothing else changes.
    """

    @abstractmethod
    def enrich(
        self, doc: EnrichedDocument, message: IngestMessage
    ) -> EnrichedDocument:
        """Add enrichment fields to *doc* and return it (mutate + return).

        Parameters
        ----------
        doc:
            Partially enriched document.  May already have fields set by
            earlier enrichers in the pipeline.
        message:
            Original IngestMessage for context (domain, user_ids, path, …).

        Returns
        -------
        EnrichedDocument
            The same object (mutated) with additional fields filled in.
        """
