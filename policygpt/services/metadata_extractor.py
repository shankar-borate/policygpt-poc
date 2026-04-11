"""Backward-compatibility shim — import from canonical location."""
from policygpt.extraction.metadata_extractor import (
    DocumentMetadata,
    SectionMetadata,
    MetadataExtractor,
    VERSION_PATTERN,
    DATE_PATTERN,
)

__all__ = ["DocumentMetadata", "SectionMetadata", "MetadataExtractor", "VERSION_PATTERN", "DATE_PATTERN"]
