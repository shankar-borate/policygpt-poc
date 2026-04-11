"""Backward-compatibility shim — import from canonical location."""
from policygpt.extraction.entity_extractor import (
    ExtractedEntity,
    DocumentEntityMap,
    EntityExtractor,
)

__all__ = ["ExtractedEntity", "DocumentEntityMap", "EntityExtractor"]
