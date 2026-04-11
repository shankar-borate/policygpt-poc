from dataclasses import dataclass, field

from policygpt.models.utils import utc_now_iso


@dataclass
class Message:
    role: str
    content: str


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
    last_answer_sources: list = field(default_factory=list)
