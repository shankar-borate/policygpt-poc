"""Enrichment layer — adds LLM-generated metadata to extracted documents.

Each Enricher adds one kind of enrichment:
  Summarizer      → document summary + per-section summaries
  FaqGenerator    → Q/A pairs + question embeddings for fast-path FAQ lookup
  EntityEnricher  → named entity map (roles, dates, thresholds, locations, …)

IngestionPipeline runs enrichers in sequence, each receiving the accumulated
EnrichedDocument from the previous stage.
"""

from policygpt.ingestion.enrichment.base import EnrichedDocument, Enricher
from policygpt.ingestion.enrichment.summarizer import Summarizer
from policygpt.ingestion.enrichment.faq_generator import FaqGenerator
from policygpt.ingestion.enrichment.entity_enricher import EntityEnricher

__all__ = [
    "EnrichedDocument",
    "Enricher",
    "Summarizer",
    "FaqGenerator",
    "EntityEnricher",
]
