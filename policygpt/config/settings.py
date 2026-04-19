"""Core Config dataclass — composes typed sub-configs for each concern.

Switching domains, models, or accuracy/cost trade-offs is a one-line change:
  domain_type          — "contest" | "policy" | "product_technical"
  ai_profile           — "openai" | "bedrock-20b" | "bedrock-120b" | ...
  accuracy_profile     — "vhigh" | "high" | "medium" | "low"
  runtime_cost_profile — "standard" | "aggressive"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from policygpt.config.ai_config import AIConfig
from policygpt.config.cache_config import CacheConfig
from policygpt.config.conversation_config import ConversationConfig
from policygpt.config.domain_defaults import DOMAIN_CONFIG_OVERRIDES
from policygpt.config.env_loader import _env_bool, _env_float, _env_int
from policygpt.config.ingestion_config import IngestionConfig
from policygpt.config.output_config import OutputConfig
from policygpt.config.presets import (
    ACCURACY_PROFILE_PRESETS,
    AI_PROFILE_PRESETS,
    RUNTIME_COST_PROFILE_PRESETS,
)
from policygpt.config.retrieval_config import RetrievalConfig
from policygpt.config.search_config import SearchConfig
from policygpt.config.storage_config import StorageConfig

# Maps each preset field name → the sub-config attribute that owns it.
_PRESET_OWNERS: dict[str, str] = {
    # RetrievalConfig
    "top_docs": "retrieval", "top_sections_per_doc": "retrieval",
    "max_sections_to_llm": "retrieval", "rerank_section_candidates": "retrieval",
    "exact_top_docs": "retrieval", "exact_top_sections_per_doc": "retrieval",
    "exact_max_sections_to_llm": "retrieval", "exact_rerank_section_candidates": "retrieval",
    "broad_top_docs": "retrieval", "broad_top_sections_per_doc": "retrieval",
    "broad_max_sections_to_llm": "retrieval", "broad_rerank_section_candidates": "retrieval",
    "max_evidence_snippets_per_section": "retrieval",
    "evidence_snippet_char_limit": "retrieval",
    "embedding_raw_excerpt_chars": "retrieval",
    "answer_context_doc_summary_char_limit": "retrieval",
    "evidence_chunk_char_limit": "retrieval",
    "evidence_neighboring_units": "retrieval",
    "small_section_full_text_chars": "retrieval",
    "exact_answer_evidence_char_limit": "retrieval",
    "broad_answer_evidence_char_limit": "retrieval",
    "answer_evidence_block_limit_exact": "retrieval",
    "answer_evidence_block_limit_broad": "retrieval",
    # ConversationConfig
    "max_recent_messages": "conversation",
    "summarize_after_turns": "conversation",
    # IngestionConfig
    "doc_summary_input_token_budget": "ingestion",
    "doc_summary_combine_token_budget": "ingestion",
    "section_summary_input_token_budget": "ingestion",
    "min_recursive_summary_token_budget": "ingestion",
    # OutputConfig
    "doc_summary_max_output_tokens": "output",
    "doc_summary_max_output_tokens_cap": "output",
    "doc_summary_chunk_max_output_tokens": "output",
    "section_summary_max_output_tokens": "output",
    "chat_max_output_tokens": "output",
    "conversation_summary_max_output_tokens": "output",
    "include_document_metadata_in_answers": "output",
    "include_section_metadata_in_answers": "output",
    "include_document_orientation_in_answers": "output",
    "include_section_orientation_in_answers": "output",
}

_OUTPUT_BOOL_FIELDS = frozenset({
    "include_document_metadata_in_answers",
    "include_section_metadata_in_answers",
    "include_document_orientation_in_answers",
    "include_section_orientation_in_answers",
})


@dataclass(frozen=True)
class Config:
    # ── Top-level profile knobs ───────────────────────────────────────────────
    ai_profile: str = "bedrock-120b"
    accuracy_profile: str = "vhigh"
    runtime_cost_profile: str = "standard"
    domain_type: str = "contest"

    # ── Sub-configs (mutable so __post_init__ can apply presets) ─────────────
    ai: AIConfig = field(default_factory=AIConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    # ── Domain computed properties ────────────────────────────────────────────

    @property
    def domain_profile(self):
        from policygpt.core.domain import get_domain_profile
        return get_domain_profile(self.domain_type)

    @property
    def domain_context(self) -> str:
        return self.domain_profile.domain_context

    # ── Initialisation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # 1. Resolve and validate ai_profile
        ai_profile = (self.ai_profile or "openai").strip().lower()
        ai_preset = AI_PROFILE_PRESETS.get(ai_profile)
        if ai_preset is None:
            raise ValueError(f"Config.ai_profile must be one of: {', '.join(sorted(AI_PROFILE_PRESETS))}")
        object.__setattr__(self, "ai_profile", ai_profile)
        self.ai.ai_provider = ai_preset["ai_provider"]
        self.ai.chat_model = (self.ai.chat_model or "").strip() or ai_preset["chat_model"]
        self.ai.embedding_model = (self.ai.embedding_model or "").strip() or ai_preset["embedding_model"]
        self.ai.bedrock_gpt_model_size = ai_preset["bedrock_gpt_model_size"]

        # 2. Resolve and validate accuracy_profile; apply preset
        accuracy_profile = (self.accuracy_profile or "high").strip().lower()
        accuracy_preset = ACCURACY_PROFILE_PRESETS.get(accuracy_profile)
        if accuracy_preset is None:
            raise ValueError(f"Config.accuracy_profile must be one of: {', '.join(sorted(ACCURACY_PROFILE_PRESETS))}")
        object.__setattr__(self, "accuracy_profile", accuracy_profile)

        high_preset = ACCURACY_PROFILE_PRESETS["high"]
        for field_name, preset_value in accuracy_preset.items():
            owner = getattr(self, _PRESET_OWNERS[field_name])
            if getattr(owner, field_name) == high_preset[field_name]:
                setattr(owner, field_name, preset_value)

        # 3. Resolve and validate runtime_cost_profile; apply preset
        runtime_cost_profile = (self.runtime_cost_profile or "standard").strip().lower()
        runtime_cost_preset = RUNTIME_COST_PROFILE_PRESETS.get(runtime_cost_profile)
        if runtime_cost_preset is None:
            raise ValueError(f"Config.runtime_cost_profile must be one of: {', '.join(sorted(RUNTIME_COST_PROFILE_PRESETS))}")
        object.__setattr__(self, "runtime_cost_profile", runtime_cost_profile)

        standard_preset = RUNTIME_COST_PROFILE_PRESETS["standard"]
        if runtime_cost_profile == "standard":
            for field_name, preset_value in standard_preset.items():
                owner = getattr(self, _PRESET_OWNERS[field_name])
                if field_name in _OUTPUT_BOOL_FIELDS and getattr(owner, field_name) is None:
                    setattr(owner, field_name, preset_value)
        else:
            for field_name, preset_value in runtime_cost_preset.items():
                owner = getattr(self, _PRESET_OWNERS[field_name])
                current = getattr(owner, field_name)
                if field_name in _OUTPUT_BOOL_FIELDS:
                    if current is None:
                        setattr(owner, field_name, preset_value)
                    continue
                baseline = accuracy_preset.get(field_name, standard_preset[field_name])
                if current == baseline:
                    setattr(owner, field_name, preset_value)

        # 4. Scale up limits for the 120B model's larger context window
        if self.ai.bedrock_gpt_model_size == "120b":
            for field_name, floor in {
                "exact_answer_evidence_char_limit": 3000,
                "broad_answer_evidence_char_limit": 2500,
                "evidence_chunk_char_limit": 2000,
                "answer_evidence_block_limit_exact": 4,
                "answer_evidence_block_limit_broad": 4,
                "answer_context_doc_summary_char_limit": 500,
                "small_section_full_text_chars": 3000,
            }.items():
                if getattr(self.retrieval, field_name) < floor:
                    setattr(self.retrieval, field_name, floor)

            for field_name, floor in {
                "doc_summary_input_token_budget": 12000,
                "doc_summary_combine_token_budget": 9000,
            }.items():
                if getattr(self.ingestion, field_name) < floor:
                    setattr(self.ingestion, field_name, floor)

            if self.output.chat_max_output_tokens < 6000:
                self.output.chat_max_output_tokens = 6000

        # 5. Apply domain-specific Config overrides
        domain_overrides = DOMAIN_CONFIG_OVERRIDES.get(self.domain_type, {})

        if "chat_max_output_tokens" in domain_overrides:
            self.output.chat_max_output_tokens = max(
                self.output.chat_max_output_tokens,
                int(domain_overrides["chat_max_output_tokens"]),
            )

        for field_name in (
            "ocr_enabled", "ocr_min_confidence", "rewrite_policies_enabled",
            "to_html_enabled", "pdf_to_html_enabled", "docx_to_html_enabled",
            "pptx_to_html_enabled", "excel_to_html_enabled", "image_to_html_enabled",
        ):
            if field_name in domain_overrides:
                default_value = IngestionConfig.__dataclass_fields__[field_name].default
                if getattr(self.ingestion, field_name) == default_value:
                    setattr(self.ingestion, field_name, domain_overrides[field_name])

        for field_name in ("hybrid_keyword_weight", "hybrid_similarity_weight", "hybrid_vector_weight"):
            if field_name in domain_overrides:
                default_value = SearchConfig.__dataclass_fields__[field_name].default
                if getattr(self.search, field_name) == default_value:
                    setattr(self.search, field_name, domain_overrides[field_name])

        # 6. Derive debug_log_dir and supplementary_facts_file from document_folder
        metadata_dir = os.path.join(self.storage.document_folder, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        if not self.storage.debug_log_dir:
            self.storage.debug_log_dir = metadata_dir
        if not self.storage.supplementary_facts_file:
            self.storage.supplementary_facts_file = os.path.join(metadata_dir, "supplementary_facts.txt")

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables, falling back to defaults."""
        base = cls()
        debug_env = os.getenv("POLICY_GPT_DEBUG")

        ai = AIConfig(
            bedrock_region=(
                os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", base.ai.bedrock_region)).strip()
                or base.ai.bedrock_region
            ),
        )

        storage = StorageConfig(
            public_base_url=os.getenv("POLICY_GPT_PUBLIC_BASE_URL", base.storage.public_base_url).rstrip("/"),
            usd_to_inr_exchange_rate=_env_float("POLICY_GPT_USD_TO_INR_RATE", base.storage.usd_to_inr_exchange_rate),
            debug_log_dir=(os.getenv("POLICY_GPT_DEBUG_LOG_DIR") or base.storage.debug_log_dir) or None,
            debug=(
                base.storage.debug
                if debug_env is None
                else debug_env.strip().lower() in {"1", "true", "yes", "on"}
            ),
        )

        ingestion = IngestionConfig(
            doc_summary_input_token_budget=_env_int(
                "POLICY_GPT_DOC_SUMMARY_INPUT_TOKEN_BUDGET", base.ingestion.doc_summary_input_token_budget
            ),
            doc_summary_combine_token_budget=_env_int(
                "POLICY_GPT_DOC_SUMMARY_COMBINE_TOKEN_BUDGET", base.ingestion.doc_summary_combine_token_budget
            ),
            section_summary_input_token_budget=_env_int(
                "POLICY_GPT_SECTION_SUMMARY_INPUT_TOKEN_BUDGET", base.ingestion.section_summary_input_token_budget
            ),
            min_recursive_summary_token_budget=_env_int(
                "POLICY_GPT_MIN_RECURSIVE_SUMMARY_TOKEN_BUDGET", base.ingestion.min_recursive_summary_token_budget
            ),
            skip_section_summary=bool(_env_bool("POLICY_GPT_SKIP_SECTION_SUMMARY", base.ingestion.skip_section_summary)),
            image_max_bytes=_env_int("POLICY_GPT_IMAGE_MAX_BYTES", base.ingestion.image_max_bytes),
            vision_provider=os.getenv("VISION_PROVIDER", base.ingestion.vision_provider).strip().lower()
            or base.ingestion.vision_provider,
            vision_model=os.getenv("VISION_MODEL", base.ingestion.vision_model).strip(),
            ingestion_user_ids=(
                tuple(
                    uid.strip()
                    for uid in os.getenv("POLICY_GPT_INGESTION_USER_IDS", "").split(",")
                    if uid.strip()
                )
                or base.ingestion.ingestion_user_ids
            ),
        )

        retrieval = RetrievalConfig(
            answer_context_doc_summary_char_limit=_env_int(
                "POLICY_GPT_ANSWER_CONTEXT_DOC_SUMMARY_CHAR_LIMIT",
                base.retrieval.answer_context_doc_summary_char_limit,
            ),
        )

        output = OutputConfig(
            doc_summary_max_output_tokens=_env_int(
                "POLICY_GPT_DOC_SUMMARY_MAX_OUTPUT_TOKENS", base.output.doc_summary_max_output_tokens
            ),
            doc_summary_chunk_max_output_tokens=_env_int(
                "POLICY_GPT_DOC_SUMMARY_CHUNK_MAX_OUTPUT_TOKENS", base.output.doc_summary_chunk_max_output_tokens
            ),
            section_summary_max_output_tokens=_env_int(
                "POLICY_GPT_SECTION_SUMMARY_MAX_OUTPUT_TOKENS", base.output.section_summary_max_output_tokens
            ),
            include_document_metadata_in_answers=_env_bool(
                "POLICY_GPT_INCLUDE_DOCUMENT_METADATA_IN_ANSWERS", base.output.include_document_metadata_in_answers
            ),
            include_section_metadata_in_answers=_env_bool(
                "POLICY_GPT_INCLUDE_SECTION_METADATA_IN_ANSWERS", base.output.include_section_metadata_in_answers
            ),
            include_document_orientation_in_answers=_env_bool(
                "POLICY_GPT_INCLUDE_DOCUMENT_ORIENTATION_IN_ANSWERS",
                base.output.include_document_orientation_in_answers,
            ),
            include_section_orientation_in_answers=_env_bool(
                "POLICY_GPT_INCLUDE_SECTION_ORIENTATION_IN_ANSWERS",
                base.output.include_section_orientation_in_answers,
            ),
        )

        conversation = ConversationConfig(
            recent_chat_message_char_limit=_env_int(
                "POLICY_GPT_RECENT_CHAT_MESSAGE_CHAR_LIMIT", base.conversation.recent_chat_message_char_limit
            ),
        )

        cache = CacheConfig(
            cache_provider=os.getenv("CACHE_PROVIDER", base.cache.cache_provider).strip().lower()
            or base.cache.cache_provider,
            redis_host=os.getenv("REDIS_HOST", base.cache.redis_host).strip(),
            redis_port=_env_int("REDIS_PORT", base.cache.redis_port),
            redis_db=_env_int("REDIS_DB", base.cache.redis_db),
            redis_password=os.getenv("REDIS_PASSWORD", base.cache.redis_password),
            redis_ssl=bool(_env_bool("REDIS_SSL", base.cache.redis_ssl)),
            redis_ssl_ca_certs=os.getenv("REDIS_SSL_CA_CERTS", base.cache.redis_ssl_ca_certs),
            redis_key_prefix=os.getenv("REDIS_KEY_PREFIX", base.cache.redis_key_prefix).strip()
            or base.cache.redis_key_prefix,
        )

        search = SearchConfig(
            opensearch_host=(os.getenv("OS_HOST") or base.search.opensearch_host) or None,
            opensearch_port=_env_int("OS_PORT", base.search.opensearch_port),
            opensearch_username=(os.getenv("OS_USERNAME") or base.search.opensearch_username) or None,
            opensearch_password=(os.getenv("OS_PASSWORD") or base.search.opensearch_password) or None,
            opensearch_use_ssl=bool(_env_bool("OS_USE_SSL", base.search.opensearch_use_ssl)),
            opensearch_verify_certs=bool(_env_bool("OS_VERIFY_CERTS", base.search.opensearch_verify_certs)),
            opensearch_index_prefix=os.getenv("OS_INDEX_PREFIX", base.search.opensearch_index_prefix).strip()
            or base.search.opensearch_index_prefix,
        )

        return cls(
            ai_profile=os.getenv("POLICY_GPT_AI_PROFILE", base.ai_profile).strip() or base.ai_profile,
            accuracy_profile=os.getenv("POLICY_GPT_ACCURACY_PROFILE", base.accuracy_profile).strip()
            or base.accuracy_profile,
            runtime_cost_profile=os.getenv("POLICY_GPT_RUNTIME_COST_PROFILE", base.runtime_cost_profile).strip()
            or base.runtime_cost_profile,
            domain_type=os.getenv("POLICY_GPT_DOMAIN_TYPE", base.domain_type).strip() or base.domain_type,
            ai=ai,
            storage=storage,
            ingestion=ingestion,
            retrieval=retrieval,
            output=output,
            conversation=conversation,
            cache=cache,
            search=search,
        )
