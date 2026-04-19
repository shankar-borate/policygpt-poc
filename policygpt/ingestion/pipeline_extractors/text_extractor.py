"""TextExtractor — converts plain-text content into title + sections."""

from __future__ import annotations

import logging

from policygpt.config import Config
from policygpt.ingestion.extraction.parsers.text_extractor import PlainTextExtractor
from policygpt.ingestion.pipeline_extractors.base import ExtractedDocument, Extractor
from policygpt.ingestion.readers.base import IngestMessage

logger = logging.getLogger(__name__)


class TextExtractor(Extractor):
    """Extracts title and sections from plain-text (.txt) documents."""

    def __init__(self, config: Config) -> None:
        self._delegate = PlainTextExtractor(config)

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({"text"})

    def extract(self, message: IngestMessage) -> ExtractedDocument:
        if message.content_type not in self.supported_content_types:
            raise ValueError(
                f"TextExtractor does not handle content_type={message.content_type!r}"
            )
        title, sections = self._delegate.extract_from_plain_text(message.source_path)
        return ExtractedDocument(title=title, sections=sections)
