"""HtmlExtractor — converts HTML content into title + sections."""

from __future__ import annotations

import logging

from policygpt.config import Config
from policygpt.ingestion.extraction.parsers.html_extractor import HtmlExtractor as _HtmlExtractor
from policygpt.ingestion.pipeline_extractors.base import ExtractedDocument, Extractor
from policygpt.ingestion.readers.base import IngestMessage

logger = logging.getLogger(__name__)


class HtmlExtractor(Extractor):
    """Extracts title and sections from HTML documents.

    Uses BeautifulSoup for semantic unit extraction with a block-level fallback
    and an optional OCR pass for embedded images.
    """

    def __init__(self, config: Config) -> None:
        self._delegate = _HtmlExtractor(config)

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({"html"})

    def extract(self, message: IngestMessage) -> ExtractedDocument:
        if message.content_type not in self.supported_content_types:
            raise ValueError(
                f"HtmlExtractor does not handle content_type={message.content_type!r}"
            )
        title, sections = self._delegate.extract_from_html(message.source_path)
        return ExtractedDocument(title=title, sections=sections)
