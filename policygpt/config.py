import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    document_folder: str = r"D:\policy-mgmt\policies"
    supported_file_patterns: tuple[str, ...] = ("*.html", "*.htm", "*.txt")
    excluded_file_name_parts: tuple[str, ...] = ("_summary",)

    chat_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-large"
    public_base_url: str = "http://127.0.0.1:8000"

    top_docs: int = 3
    top_sections_per_doc: int = 3
    max_sections_to_llm: int = 4

    max_recent_messages: int = 6
    summarize_after_turns: int = 8

    min_section_chars: int = 300
    target_section_chars: int = 1800
    max_section_chars: int = 3200

    redaction_rules: dict[str, str] = field(
        default_factory=lambda: {
            "Kotak": "KKK",
            "kotak": "KKK",
            "KOTAK": "KKK",
        }
    )

    doc_summary_max_output_tokens: int = 400
    section_summary_max_output_tokens: int = 220
    chat_max_output_tokens: int = 900
    conversation_summary_max_output_tokens: int = 250

    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            public_base_url=os.getenv("POLICY_GPT_PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            debug=os.getenv("POLICY_GPT_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"},
        )
