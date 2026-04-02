from dataclasses import dataclass, field
from datetime import datetime, timezone

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
