"""policygpt.ingestion — document ingestion pipeline.

Layers
------
readers/     Document sources (folder, SQS, API).  Each yields IngestMessage.
extractors/  Content extraction by file type (html, text, pdf, ppt, image).
enrichment/  LLM enrichment (summarise, FAQ, entity extraction).
pipeline.py  IngestionPipeline — orchestrates all layers end-to-end.
"""

from policygpt.ingestion.pipeline import IngestionPipeline
from policygpt.ingestion.readers import IngestMessage, Reader, FolderReader
from policygpt.ingestion.extractors import ExtractedDocument, Extractor, ExtractorRegistry
from policygpt.ingestion.enrichment import EnrichedDocument, Enricher

__all__ = [
    "IngestionPipeline",
    "IngestMessage",
    "Reader",
    "FolderReader",
    "ExtractedDocument",
    "Extractor",
    "ExtractorRegistry",
    "EnrichedDocument",
    "Enricher",
]
