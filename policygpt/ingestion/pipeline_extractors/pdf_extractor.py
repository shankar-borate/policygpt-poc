"""PdfExtractor — converts PDF content into title + sections."""

from __future__ import annotations

import logging

from policygpt.config import Config
from policygpt.ingestion.extraction.parsers.pdf_extractor import PdfExtractor as _PdfExtractor
from policygpt.ingestion.pipeline_extractors.base import ExtractedDocument, Extractor
from policygpt.ingestion.readers.base import IngestMessage

logger = logging.getLogger(__name__)


class PdfExtractor(Extractor):
    """Extracts title and sections from PDF documents using pypdf."""

    def __init__(self, config: Config) -> None:
        self._delegate = _PdfExtractor(config)

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({"pdf"})

    def extract(self, message: IngestMessage) -> ExtractedDocument:
        if message.content_type not in self.supported_content_types:
            raise ValueError(
                f"PdfExtractor does not handle content_type={message.content_type!r}"
            )
        title, sections = self._delegate.extract_from_pdf(message.source_path)
        return ExtractedDocument(title=title, sections=sections)
