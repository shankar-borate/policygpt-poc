"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.corpus import (
    DocumentCorpus,
    ProgressCallback,
    l2_normalize,
    cosine_similarity,
)

__all__ = ["DocumentCorpus", "ProgressCallback", "l2_normalize", "cosine_similarity"]
