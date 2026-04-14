"""Extractor layer — converts raw file content into structured sections.

Each Extractor handles one content_type and returns an ExtractedDocument
(title + list of (section_title, section_text) tuples).

ExtractorRegistry picks the right extractor based on IngestMessage.content_type.
"""

from policygpt.ingestion.extractors.base import ExtractedDocument, Extractor
from policygpt.ingestion.extractors.registry import ExtractorRegistry
from policygpt.ingestion.extractors.html_extractor import HtmlExtractor
from policygpt.ingestion.extractors.text_extractor import TextExtractor
from policygpt.ingestion.extractors.pdf_extractor import PdfExtractor
from policygpt.ingestion.extractors.ppt_extractor import PptExtractor
from policygpt.ingestion.extractors.image_extractor import ImageExtractor

__all__ = [
    "ExtractedDocument",
    "Extractor",
    "ExtractorRegistry",
    "HtmlExtractor",
    "TextExtractor",
    "PdfExtractor",
    "PptExtractor",
    "ImageExtractor",
]
