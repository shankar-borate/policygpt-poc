"""ExtractorRegistry — maps content_type → Extractor instance.

Usage
-----
registry = ExtractorRegistry(config)
extractor = registry.get("html")        # raises if not found
document  = extractor.extract(message)

Registering a new format
-------------------------
1. Subclass Extractor, implement extract() and supported_content_types.
2. Import the class below and add it to _EXTRACTOR_CLASSES.
Done — nothing else changes.
"""

from __future__ import annotations

import logging
from typing import Type

from policygpt.config import Config
from policygpt.ingestion.pipeline_extractors.base import Extractor
from policygpt.ingestion.pipeline_extractors.html_extractor import HtmlExtractor
from policygpt.ingestion.pipeline_extractors.image_extractor import ImageExtractor
from policygpt.ingestion.pipeline_extractors.pdf_extractor import PdfExtractor
from policygpt.ingestion.pipeline_extractors.ppt_extractor import PptExtractor
from policygpt.ingestion.pipeline_extractors.text_extractor import TextExtractor

logger = logging.getLogger(__name__)

# Ordered list of extractor classes to register at startup.
# Add new classes here — their supported_content_types drive the mapping.
_EXTRACTOR_CLASSES: list[Type[Extractor]] = [
    HtmlExtractor,
    TextExtractor,
    PdfExtractor,
    PptExtractor,
    ImageExtractor,
]


class ExtractorRegistry:
    """Central registry that selects the right Extractor for a content type.

    All extractors are instantiated once at construction time so any
    heavy init (lazy boto3 clients, model loads) happens at startup,
    not on the first document.

    Parameters
    ----------
    config:
        Application config forwarded to each extractor constructor.
    """

    def __init__(self, config: Config) -> None:
        self._registry: dict[str, Extractor] = {}
        for cls in _EXTRACTOR_CLASSES:
            instance = cls(config)  # type: ignore[call-arg]
            for ct in instance.supported_content_types:
                if ct in self._registry:
                    logger.warning(
                        "Content-type %r already registered by %s — overriding with %s",
                        ct,
                        type(self._registry[ct]).__name__,
                        cls.__name__,
                    )
                self._registry[ct] = instance
        logger.debug(
            "ExtractorRegistry ready: %s", sorted(self._registry.keys())
        )

    def get(self, content_type: str) -> Extractor:
        """Return the Extractor registered for *content_type*.

        Raises
        ------
        KeyError
            If no extractor is registered for the given content type.
        """
        try:
            return self._registry[content_type]
        except KeyError:
            supported = sorted(self._registry.keys())
            raise KeyError(
                f"No extractor registered for content_type={content_type!r}. "
                f"Supported: {supported}"
            ) from None

    def supports(self, content_type: str) -> bool:
        """Return True if an extractor is registered for *content_type*."""
        return content_type in self._registry

    @property
    def supported_content_types(self) -> frozenset[str]:
        """All content types this registry can handle."""
        return frozenset(self._registry.keys())
