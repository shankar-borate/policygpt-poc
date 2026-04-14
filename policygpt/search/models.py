"""Provider-agnostic search DTOs.

All search types and results are expressed in terms of these plain dataclasses.
No provider SDK types leak beyond the provider package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SearchType(str, Enum):
    """The three complementary retrieval mechanisms."""
    KEYWORD    = "keyword"     # BM25 exact/fuzzy term matching
    SIMILARITY = "similarity"  # more_like_this — term co-occurrence patterns
    VECTOR     = "vector"      # kNN dense embedding — semantic meaning


@dataclass
class SearchQuery:
    """Encapsulates everything a search call needs."""
    text: str
    embedding: Optional[np.ndarray] = None
    top_k: int = 10
    # Optional metadata pre-filters applied before scoring (e.g. doc_id list,
    # audience, effective_date).  Format: {"field": value_or_list}
    filters: dict = field(default_factory=dict)
    # Which search types to activate; all three by default
    search_types: tuple[SearchType, ...] = (
        SearchType.KEYWORD,
        SearchType.SIMILARITY,
        SearchType.VECTOR,
    )


@dataclass
class FaqResult:
    """A single FAQ Q/A pair returned from the FAQ index."""
    faq_id: str
    doc_id: str
    document_title: str
    question: str
    answer: str
    source_path: str
    score: float


@dataclass
class SearchResult:
    """A single section candidate returned by any provider."""
    section_id: str
    doc_id: str
    score: float              # normalised blended score in [0, 1]
    document_title: str
    section_title: str
    source_path: str
    order_index: int
    raw_text: str = ""
    summary: str = ""
    section_type: str = "general"
    metadata_tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # Which search type(s) contributed to this result
    matched_by: tuple[SearchType, ...] = field(default_factory=tuple)
