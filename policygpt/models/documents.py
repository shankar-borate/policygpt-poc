from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SectionRecord:
    section_id: str
    title: str
    raw_text: str
    masked_text: str
    summary: str
    summary_embedding: np.ndarray
    source_path: str
    doc_id: str
    order_index: int
    normalized_title: str = ""
    section_type: str = "general"
    metadata_tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    title_terms: list[str] = field(default_factory=list)
    token_counts: dict[str, int] = field(default_factory=dict)
    token_length: int = 0


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source_path: str
    raw_text: str
    masked_text: str
    summary: str
    summary_embedding: np.ndarray
    sections: list[SectionRecord] = field(default_factory=list)
    normalized_title: str = ""
    canonical_title: str = ""
    document_type: str = "document"
    version: str = ""
    effective_date: str = ""
    metadata_tags: list[str] = field(default_factory=list)
    audiences: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    title_terms: list[str] = field(default_factory=list)
    token_counts: dict[str, int] = field(default_factory=dict)
    token_length: int = 0
    faq: str = ""
    # Per-question embeddings for FAQ fast-path retrieval.
    faq_qa_pairs: list[tuple[str, str]] = field(default_factory=list)
    faq_q_embeddings: list[np.ndarray] = field(default_factory=list)
    # DocumentEntityMap instance — stored as Any to avoid circular import.
    entity_map: Any = None
