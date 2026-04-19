"""Extractor contract — ExtractedSection/ExtractedDocument dataclasses and Extractor ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from policygpt.ingestion.readers.base import IngestMessage
from policygpt.models.extraction import ExtractedSection  # re-exported for convenience

__all__ = ["ExtractedSection", "ExtractedDocument", "Extractor"]


@dataclass
class ExtractedDocument:
    """Result of running an Extractor over an IngestMessage.

    Attributes
    ----------
    title:
        Human-readable document title derived from the content or file name.
    sections:
        Ordered list of ExtractedSection objects.
    """

    title: str
    sections: list[ExtractedSection] = field(default_factory=list)


class Extractor(ABC):
    """Abstract content extractor.

    Each concrete implementation handles exactly one content type
    (html, text, pdf, ppt, image).  The pipeline selects the right
    one via ExtractorRegistry — no if/elif chains in business logic.

    Implementing a new format:
      1. Subclass Extractor and implement extract() and supported_content_types.
      2. Register it in ExtractorRegistry.
      Done — nothing else changes.
    """

    @property
    @abstractmethod
    def supported_content_types(self) -> frozenset[str]:
        """Content types this extractor handles (e.g. frozenset{"html"})."""

    @abstractmethod
    def extract(self, message: IngestMessage) -> ExtractedDocument:
        """Convert raw IngestMessage content into title + sections.

        Raises
        ------
        NotImplementedError
            For placeholder extractors that are not yet implemented.
        ValueError
            If message.content_type is not in supported_content_types.
        """
