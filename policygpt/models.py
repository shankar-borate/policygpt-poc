from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    # List of (question_text, embedding) pairs — built during ingest when
    # generate_faq=True and used at query time to short-circuit RAG for
    # near-exact FAQ matches (cosine ≥ faq_fastpath_min_score).
    faq_qa_pairs: list[tuple[str, str]] = field(default_factory=list)
    faq_q_embeddings: list[np.ndarray] = field(default_factory=list)
    # DocumentEntityMap instance — stored as Any to avoid a circular import.
    # Access via corpus.entity_lookup which is rebuilt on every ingest.
    entity_map: Any = None


@dataclass
class Message:
    role: str
    content: str


@dataclass
class SourceReference:
    document_title: str
    section_title: str
    source_path: str
    score: float
    section_order_index: int = 0


@dataclass
class ChatResult:
    thread_id: str
    answer: str
    sources: list[SourceReference]


@dataclass
class ThreadState:
    thread_id: str
    recent_messages: list[Message] = field(default_factory=list)
    display_messages: list[Message] = field(default_factory=list)
    conversation_summary: str = ""
    active_doc_ids: list[str] = field(default_factory=list)
    active_section_ids: list[str] = field(default_factory=list)
    current_topic: str = ""
    title: str = "New chat"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    last_answer_sources: list[SourceReference] = field(default_factory=list)
