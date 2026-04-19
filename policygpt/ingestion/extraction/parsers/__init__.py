"""policygpt.ingestion.extraction.parsers — strategy-pattern file extractors."""

from policygpt.ingestion.extraction.parsers.base import BaseExtractor
from policygpt.ingestion.extraction.parsers.html_extractor import HtmlExtractor
from policygpt.ingestion.extraction.parsers.pdf_extractor import PdfExtractor
from policygpt.ingestion.extraction.parsers.text_extractor import PlainTextExtractor

__all__ = [
    "BaseExtractor",
    "HtmlExtractor",
    "PdfExtractor",
    "PlainTextExtractor",
]
