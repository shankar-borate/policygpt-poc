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
    "bedrock-claude-sonnet-4-6": {
        "ai_provider": "bedrock",
        "chat_model": "global.anthropic.claude-sonnet-4-6",
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "bedrock_gpt_model_size": "",
    },
    "bedrock-claude-opus-4-6": {
        "ai_provider": "bedrock",
        "chat_model": "global.anthropic.claude-opus-4-6-v1",
        "embedding_model": "amazon.titan-embed-text-v2:0",
        "bedrock_gpt_model_size": "",
    },
}

ACCURACY_PROFILE_PRESETS: dict[str, dict[str, int]] = {
    "high": {
        "top_docs": 3,
        "top_sections_per_doc": 3,
        "max_sections_to_llm": 4,
        "rerank_section_candidates": 12,
        "exact_top_docs": 2,
        "exact_top_sections_per_doc": 4,
        "exact_max_sections_to_llm": 3,
        "exact_rerank_section_candidates": 10,
        "broad_top_docs": 6,
        "broad_top_sections_per_doc": 6,
        "broad_max_sections_to_llm": 8,
        "broad_rerank_section_candidates": 24,
        "max_evidence_snippets_per_section": 3,
        "evidence_snippet_char_limit": 320,
        "embedding_raw_excerpt_chars": 600,
        "answer_context_doc_summary_char_limit": 260,
        "evidence_chunk_char_limit": 900,
        "evidence_neighboring_units": 1,
        "small_section_full_text_chars": 1500,
        "exact_answer_evidence_char_limit": 1600,
        "broad_answer_evidence_char_limit": 1200,
        "answer_evidence_block_limit_exact": 2,
        "answer_evidence_block_limit_broad": 2,
        "max_recent_messages": 6,
        "doc_summary_input_token_budget": 6000,
        "doc_summary_combine_token_budget": 4500,
        "section_summary_input_token_budget": 2500,
        "min_recursive_summary_token_budget": 250,
        "doc_summary_max_output_tokens": 400,
        "doc_summary_max_output_tokens_cap": 800,
        "doc_summary_chunk_max_output_tokens": 220,
        "section_summary_max_output_tokens": 220,
        "chat_max_output_tokens": 900,
        "conversation_summary_max_output_tokens": 250,
    },
    "medium": {
        "top_docs": 2,
        "top_sections_per_doc": 2,
        "max_sections_to_llm": 3,
        "rerank_section_candidates": 8,
        "exact_top_docs": 2,
        "exact_top_sections_per_doc": 3,
        "exact_max_sections_to_llm": 2,
        "exact_rerank_section_candidates": 6,
        "broad_top_docs": 4,
        "broad_top_sections_per_doc": 4,
        "broad_max_sections_to_llm": 5,
        "broad_rerank_section_candidates": 12,
        "max_evidence_snippets_per_section": 2,
        "evidence_snippet_char_limit": 240,
        "embedding_raw_excerpt_chars": 350,
        "answer_context_doc_summary_char_limit": 180,
        "evidence_chunk_char_limit": 650,
        "evidence_neighboring_units": 1,
        "small_section_full_text_chars": 1000,
        "exact_answer_evidence_char_limit": 1100,
        "broad_answer_evidence_char_limit": 850,
        "answer_evidence_block_limit_exact": 1,
        "answer_evidence_block_limit_broad": 2,
        "max_recent_messages": 4,
        "doc_summary_input_token_budget": 4200,
        "doc_summary_combine_token_budget": 3000,
        "section_summary_input_token_budget": 1800,
        "min_recursive_summary_token_budget": 250,
        "doc_summary_max_output_tokens": 280,
        "doc_summary_max_output_tokens_cap": 500,
        "doc_summary_chunk_max_output_tokens": 160,
        "section_summary_max_output_tokens": 160,
        "chat_max_output_tokens": 650,
        "conversation_summary_max_output_tokens": 160,
    },
    "low": {
        "top_docs": 2,
        "top_sections_per_doc": 2,
        "max_sections_to_llm": 2,
        "rerank_section_candidates": 5,
        "exact_top_docs": 1,
        "exact_top_sections_per_doc": 2,
        "exact_max_sections_to_llm": 1,
        "exact_rerank_section_candidates": 4,
        "broad_top_docs": 3,
        "broad_top_sections_per_doc": 3,
        "broad_max_sections_to_llm": 3,
        "broad_rerank_section_candidates": 8,
        "max_evidence_snippets_per_section": 1,
        "evidence_snippet_char_limit": 180,
        "embedding_raw_excerpt_chars": 180,
        "answer_context_doc_summary_char_limit": 120,
        "evidence_chunk_char_limit": 450,
        "evidence_neighboring_units": 0,
        "small_section_full_text_chars": 700,
        "exact_answer_evidence_char_limit": 800,
        "broad_answer_evidence_char_limit": 600,
        "answer_evidence_block_limit_exact": 1,
        "answer_evidence_block_limit_broad": 1,
        "max_recent_messages": 3,
        "doc_summary_input_token_budget": 2600,
        "doc_summary_combine_token_budget": 1800,
        "section_summary_input_token_budget": 1200,
        "min_recursive_summary_token_budget": 250,
        "doc_summary_max_output_tokens": 180,
        "doc_summary_max_output_tokens_cap": 320,
        "doc_summary_chunk_max_output_tokens": 120,
        "section_summary_max_output_tokens": 120,
        "chat_max_output_tokens": 400,
        "conversation_summary_max_output_tokens": 100,
    },
}


@dataclass(frozen=True)
class Config:
    document_folder: str = r"D:\policy-mgmt\data\durandhar_html"
    supported_file_patterns: tuple[str, ...] = ("*.html", "*.htm", "*.txt", "*.pdf")
    excluded_file_name_parts: tuple[str, ...] = ("_summary",)

    # Change this one value to switch model stack.
    ai_profile: str = (
        "bedrock-claude-sonnet-4-6"  # openai | bedrock-20b | bedrock-120b | bedrock-claude-sonnet-4-6 | bedrock-claude-opus-4-6
    )
    # Change this one value to tune cost/quality across all models.
    accuracy_profile: str = "high"  # high | medium | low
    ai_provider: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    public_base_url: str = "http://127.0.0.1:8012"
    bedrock_region: str = "ap-south-1"
    bedrock_gpt_model_size: str = ""
    debug_log_dir: str = r"D:\policy-mgmt\data\durandhar_html\metadata"

    top_docs: int = 3
    top_sections_per_doc: int = 3
    max_sections_to_llm: int = 4
    rerank_section_candidates: int = 12
    exact_top_docs: int = 2
    exact_top_sections_per_doc: int = 4
    exact_max_sections_to_llm: int = 3
    exact_rerank_section_candidates: int = 10
    broad_top_docs: int = 6
    broad_top_sections_per_doc: int = 6
    broad_max_sections_to_llm: int = 8
    broad_rerank_section_candidates: int = 24
    max_evidence_snippets_per_section: int = 3
    evidence_snippet_char_limit: int = 320
    embedding_raw_excerpt_chars: int = 600
    answer_context_doc_summary_char_limit: int = 260
    evidence_chunk_char_limit: int = 900
    evidence_neighboring_units: int = 1
    small_section_full_text_chars: int = 1500
    exact_answer_evidence_char_limit: int = 1600
    broad_answer_evidence_char_limit: int = 1200
    answer_evidence_block_limit_exact: int = 2
    answer_evidence_block_limit_broad: int = 2

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
    doc_summary_max_output_tokens_cap: int = 800
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
    answerability_high_confidence_score: float = 0.50
    answerability_min_support_matches: int = 1
    answerability_min_support_matches_multi_doc: int = 2
    answerability_min_exact_evidence_matches: int = 1
    exact_query_section_parent_weight_scale: float = 0.75

    debug: bool = True

    def __post_init__(self) -> None:
        ai_profile = (self.ai_profile or "openai").strip().lower()
        ai_preset = AI_PROFILE_PRESETS.get(ai_profile)
        if ai_preset is None:
            supported_profiles = ", ".join(sorted(AI_PROFILE_PRESETS))
            raise ValueError(f"Config.ai_profile must be one of: {supported_profiles}")

        accuracy_profile = (self.accuracy_profile or "high").strip().lower()
        accuracy_preset = ACCURACY_PROFILE_PRESETS.get(accuracy_profile)
        if accuracy_preset is None:
            supported_profiles = ", ".join(sorted(ACCURACY_PROFILE_PRESETS))
            raise ValueError(f"Config.accuracy_profile must be one of: {supported_profiles}")

        object.__setattr__(self, "ai_profile", ai_profile)
        object.__setattr__(self, "accuracy_profile", accuracy_profile)
        object.__setattr__(self, "ai_provider", ai_preset["ai_provider"])
        object.__setattr__(self, "chat_model", (self.chat_model or "").strip() or ai_preset["chat_model"])
        object.__setattr__(self, "embedding_model", (self.embedding_model or "").strip() or ai_preset["embedding_model"])
        object.__setattr__(self, "bedrock_gpt_model_size", ai_preset["bedrock_gpt_model_size"])

        # Accuracy profiles provide model-agnostic defaults for retrieval
        # breadth, prompt context size, and output budgets. If a field has
        # already been tuned away from the high-accuracy baseline, preserve
        # that explicit override instead of replacing it with the preset.
        high_accuracy_preset = ACCURACY_PROFILE_PRESETS["high"]
        for field_name, preset_value in accuracy_preset.items():
            current_value = getattr(self, field_name)
            if accuracy_profile == "high" or current_value == high_accuracy_preset[field_name]:
                object.__setattr__(self, field_name, preset_value)

    @classmethod
    def from_env(cls) -> "Config":
        base_config = cls()
        debug_log_dir_env = os.getenv("POLICY_GPT_DEBUG_LOG_DIR")
        debug_env = os.getenv("POLICY_GPT_DEBUG")
        return cls(
            ai_profile=base_config.ai_profile,
            accuracy_profile=os.getenv("POLICY_GPT_ACCURACY_PROFILE", base_config.accuracy_profile).strip()
            or base_config.accuracy_profile,
            chat_model=base_config.chat_model,
            embedding_model=base_config.embedding_model,
            public_base_url=os.getenv("POLICY_GPT_PUBLIC_BASE_URL", base_config.public_base_url).rstrip("/"),
            bedrock_region=os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", base_config.bedrock_region)).strip()
            or base_config.bedrock_region,
            debug_log_dir=base_config.debug_log_dir if debug_log_dir_env is None else debug_log_dir_env.strip(),
            debug=base_config.debug if debug_env is None else debug_env.strip().lower() in {"1", "true", "yes", "on"},
        )
