from dataclasses import dataclass


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
