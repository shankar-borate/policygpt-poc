import os
from dataclasses import dataclass, field


AI_PROFILE_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "ai_provider": "openai",
        "chat_model": "gpt-4.1",
        "embedding_model": "text-embedding-3-large",
        "bedrock_gpt_model_size": "",
    },
    "bedrock-20b": {
        "ai_provider": "bedrock",
        "chat_model": "openai.gpt-oss-20b-1:0",
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "bedrock_gpt_model_size": "20b",
    },
    "bedrock-120b": {
        "ai_provider": "bedrock",
        "chat_model": "openai.gpt-oss-120b-1:0",
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "bedrock_gpt_model_size": "120b",
    },
}


@dataclass(frozen=True)
class Config:
    document_folder: str = r"D:\policy-mgmt\data\durandhar_html"
    supported_file_patterns: tuple[str, ...] = ("*.html", "*.htm", "*.txt", "*.pdf")
    excluded_file_name_parts: tuple[str, ...] = ("_summary",)

    # Change this one value to switch model stack.
    ai_profile: str = "openai"  # openai | bedrock-20b | bedrock-120b
    ai_provider: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    public_base_url: str = "http://127.0.0.1:8010"
    bedrock_region: str = "ap-south-1"
    bedrock_gpt_model_size: str = ""
    debug_log_dir: str = r"D:\policy-mgmt\data\durandhar_html\metadata"

    top_docs: int = 3
    top_sections_per_doc: int = 3
    max_sections_to_llm: int = 4
    rerank_section_candidates: int = 12
    max_evidence_snippets_per_section: int = 3
    evidence_snippet_char_limit: int = 320

    max_recent_messages: int = 6
    summarize_after_turns: int = 8

    min_section_chars: int = 300
    target_section_chars: int = 1800
    max_section_chars: int = 3200
    token_estimate_chars_per_token: int = 4
    token_estimate_tokens_per_word: float = 1.3
    doc_summary_input_token_budget: int = 6000
    doc_summary_combine_token_budget: int = 4500
    section_summary_input_token_budget: int = 2500
    min_recursive_summary_token_budget: int = 250

    redaction_rules: dict[str, str] = field(
        default_factory=lambda: {
            "Kotak": "KKK",
            "kotak": "KKK",
            "KOTAK": "KKK",
        }
    )

    doc_summary_max_output_tokens: int = 400
    doc_summary_chunk_max_output_tokens: int = 220
    section_summary_max_output_tokens: int = 220
    chat_max_output_tokens: int = 900
    conversation_summary_max_output_tokens: int = 250
    ai_rate_limit_retries: int = 2
    ai_rate_limit_backoff_seconds: float = 8.0
    doc_semantic_weight: float = 0.48
    doc_lexical_weight: float = 0.24
    doc_title_weight: float = 0.16
    doc_metadata_weight: float = 0.12
    section_semantic_weight: float = 0.36
    section_lexical_weight: float = 0.24
    section_parent_weight: float = 0.16
    section_title_weight: float = 0.12
    section_metadata_weight: float = 0.12
    answerability_min_section_score: float = 0.24
    answerability_min_support_matches: int = 1

    debug: bool = True

    def __post_init__(self) -> None:
        profile = (self.ai_profile or "openai").strip().lower()
        preset = AI_PROFILE_PRESETS.get(profile)
        if preset is None:
            supported_profiles = ", ".join(sorted(AI_PROFILE_PRESETS))
            raise ValueError(f"Config.ai_profile must be one of: {supported_profiles}")

        object.__setattr__(self, "ai_profile", profile)
        object.__setattr__(self, "ai_provider", preset["ai_provider"])
        object.__setattr__(self, "chat_model", (self.chat_model or "").strip() or preset["chat_model"])
        object.__setattr__(self, "embedding_model", (self.embedding_model or "").strip() or preset["embedding_model"])
        object.__setattr__(self, "bedrock_gpt_model_size", preset["bedrock_gpt_model_size"])

    @classmethod
    def from_env(cls) -> "Config":
        base_config = cls()
        debug_log_dir_env = os.getenv("POLICY_GPT_DEBUG_LOG_DIR")
        debug_env = os.getenv("POLICY_GPT_DEBUG")
        return cls(
            ai_profile=base_config.ai_profile,
            chat_model=base_config.chat_model,
            embedding_model=base_config.embedding_model,
            public_base_url=os.getenv("POLICY_GPT_PUBLIC_BASE_URL", base_config.public_base_url).rstrip("/"),
            bedrock_region=os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", base_config.bedrock_region)).strip()
            or base_config.bedrock_region,
            debug_log_dir=base_config.debug_log_dir if debug_log_dir_env is None else debug_log_dir_env.strip(),
            debug=base_config.debug if debug_env is None else debug_env.strip().lower() in {"1", "true", "yes", "on"},
        )
