"""Re-export metadata extraction from the canonical location.

The implementation lives in policygpt.extraction.metadata_extractor.
Move the body here when the extraction/ package is retired.
"""

from policygpt.extraction.metadata_extractor import (  # noqa: F401
    DocumentMetadata,
    SectionMetadata,
    MetadataExtractor,
)
