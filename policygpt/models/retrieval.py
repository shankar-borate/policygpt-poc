from dataclasses import dataclass, field


@dataclass
class SourceReference:
    document_title: str
    section_title: str
    source_path: str
    score: float
    section_order_index: int = 0
    original_source_path: str = ""


@dataclass
class ChatResult:
    thread_id: str
    answer: str
    sources: list[SourceReference]
    thread: object = None  # ThreadState — set by bot.chat() so callers skip the extra OS load
