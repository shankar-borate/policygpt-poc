"""Core Config dataclass and from_env() loader.

Config is the single object passed to every component (bot, corpus, server).
It is frozen (immutable after construction) so components can safely cache
references to it.

Switching domains, models, or accuracy/cost trade-offs is a one-line change:
  domain_type      — "contest" | "policy" | any registered domain
  ai_profile       — "openai" | "bedrock-20b" | "bedrock-120b" | ...
  accuracy_profile — "vhigh" | "high" | "medium" | "low"
  runtime_cost_profile — "standard" | "aggressive"
"""

import os
from dataclasses import dataclass, field

from policygpt.config.domain_defaults import DOMAIN_CONFIG_OVERRIDES
from policygpt.config.env_loader import _env_bool, _env_float, _env_int
from policygpt.config.presets import (
    ACCURACY_PROFILE_PRESETS,
    AI_PROFILE_PRESETS,
    RUNTIME_COST_PROFILE_PRESETS,
)


@dataclass(frozen=True)
class Config:
    # ── Storage ───────────────────────────────────────────────────────────────
    document_folder: str = r"D:\policy-mgmt\data\test"
    supported_file_patterns: tuple[str, ...] = ("*.html", "*.htm", "*.txt", "*.pdf")
    excluded_file_name_parts: tuple[str, ...] = ("_summary",)

    # ── AI model selection ────────────────────────────────────────────────────
    # Change this one value to switch the entire model stack.
    # Options: openai | bedrock-20b | bedrock-120b | bedrock-claude-sonnet-4-6 | bedrock-claude-opus-4-6
    ai_profile: str = "bedrock-120b"

    # ── Quality / cost knobs ──────────────────────────────────────────────────
    # Change accuracy_profile to tune retrieval breadth and output budgets.
    # Options: vhigh | high | medium | low
    accuracy_profile: str = "vhigh"

    # Change runtime_cost_profile to tune per-request cost without affecting
    # ingestion quality or model selection.
    # Options: standard | aggressive
    runtime_cost_profile: str = "standard"

    # ── Resolved at __post_init__ from ai_profile ─────────────────────────────
    ai_provider: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    bedrock_gpt_model_size: str = ""

    # ── Network / endpoints ───────────────────────────────────────────────────
    public_base_url: str = "http://127.0.0.1:8012"
    bedrock_region: str = "ap-south-1"

    # ── Currency ──────────────────────────────────────────────────────────────
    usd_to_inr_exchange_rate: float = 93.0

    # ── Debug / observability ─────────────────────────────────────────────────
    debug_log_dir: str = r"D:\policy-mgmt\data\test\metadata"
    debug: bool = True

    # ── Supplementary context ─────────────────────────────────────────────────
    # Path to a plain-text file with extra facts (e.g. business rules) injected
    # into every LLM prompt but never shown to the user as a source.
    supplementary_facts_file: str = r"D:\policy-mgmt\data\test\metadata\supplementary_facts.txt"

    # ── Domain ────────────────────────────────────────────────────────────────
    # Selects domain profile (prompt text, entity categories, etc.) from domain/.
    # Built-in values: "contest" | "policy"
    # Add a new file under policygpt/domain/ to register additional types.
    domain_type: str = "policy"

    # ── Ingestion: access control ─────────────────────────────────────────────
    # User IDs that are granted access to every document ingested at server
    # startup.  Set via POLICY_GPT_INGESTION_USER_IDS (comma-separated) so the
    # documents are queryable by those users through OpenSearch.
    # Leave empty (default) only when hybrid_search_enabled=False, because an
    # empty list means the OpenSearch user_id filter never matches anything.
    ingestion_user_ids: tuple[str, ...] = ()

    # ── Redaction ─────────────────────────────────────────────────────────────
    redaction_rules: dict[str, str] = field(
        default_factory=lambda: {
            "Kotak": "KKK",
            "kotak": "KKK",
            "KOTAK": "KKK",
        }
    )

    # ── OCR ───────────────────────────────────────────────────────────────────
    # When enabled, images inside HTML files are passed through AWS Textract
    # and the extracted text is indexed alongside the surrounding content.
    # Configured per-domain in config/domain_defaults.py.
    ocr_enabled: bool = False
    ocr_provider: str = "textract"   # only "textract" supported today
    ocr_min_confidence: float = 80.0  # Textract LINE block confidence threshold (0–100)

    # ── Ingestion: FAQ generation ─────────────────────────────────────────────
    generate_faq: bool = True
    faq_max_questions: int = 30
    faq_max_output_tokens: int = 2700

    # ── Ingestion: entity extraction ──────────────────────────────────────────
    generate_entity_map: bool = True
    entity_map_max_output_tokens: int = 3600

    # ── Retrieval: FAQ fast-path ──────────────────────────────────────────────
    faq_fastpath_enabled: bool = True
    faq_fastpath_min_score: float = 0.92
    aggregate_faq_top_k: int = 30

    # ── Retrieval: section / document counts ──────────────────────────────────
    # Defaults match the "high" accuracy preset; __post_init__ overwrites these
    # based on accuracy_profile and runtime_cost_profile.
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

    # ── Retrieval: evidence sizing ────────────────────────────────────────────
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

    # ── Conversation ──────────────────────────────────────────────────────────
    max_recent_messages: int = 6
    recent_chat_message_char_limit: int = 0
    summarize_after_turns: int = 8

    # ── Ingestion: section chunking ───────────────────────────────────────────
    min_section_chars: int = 300
    target_section_chars: int = 1800
    max_section_chars: int = 3200
    token_estimate_chars_per_token: int = 4
    token_estimate_tokens_per_word: float = 1.3

    # ── Ingestion: summarisation token budgets ────────────────────────────────
    doc_summary_input_token_budget: int = 6000
    doc_summary_combine_token_budget: int = 4500
    section_summary_input_token_budget: int = 2500
    min_recursive_summary_token_budget: int = 250
    skip_section_summary: bool = False

    # ── Output token limits ───────────────────────────────────────────────────
    doc_summary_max_output_tokens: int = 1200
    doc_summary_max_output_tokens_cap: int = 2400
    doc_summary_chunk_max_output_tokens: int = 660
    section_summary_max_output_tokens: int = 660
    chat_max_output_tokens: int = 2700
    conversation_summary_max_output_tokens: int = 750

    # ── Answer context flags ──────────────────────────────────────────────────
    # None means "use the runtime_cost_profile default".
    include_document_metadata_in_answers: bool | None = None
    include_section_metadata_in_answers: bool | None = None
    include_document_orientation_in_answers: bool | None = None
    include_section_orientation_in_answers: bool | None = None

    # ── AI rate limiting ──────────────────────────────────────────────────────
    ai_rate_limit_retries: int = 2
    ai_rate_limit_backoff_seconds: float = 8.0

    # ── Hybrid search ─────────────────────────────────────────────────────────
    # Master switch — set True to route retrieval through an external vector store.
    # When False (default) the existing in-memory path is used unchanged.
    hybrid_search_enabled: bool = True

    # Which vector store backend to use.
    # Must match a key in policygpt/search/factory.py registry.
    # Swap to "pinecone" / "weaviate" / "pgvector" to switch backends with
    # zero other code changes.
    hybrid_search_provider: str = "opensearch"

    # ── OpenSearch provider config ────────────────────────────────────────────
    # Used only when hybrid_search_provider = "opensearch".
    # Credentials must be set via environment variables (loaded from opensearch.env).
    # See opensearch.env.example — never put real credentials in this file.
    opensearch_host: str = ""
    opensearch_port: int = 9200
    opensearch_username: str = ""
    opensearch_password: str = ""
    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = False
    opensearch_index_prefix: str = "policygpt"

    # ── Hybrid search weights ─────────────────────────────────────────────────
    # Controls the blend of the three complementary retrieval mechanisms.
    # Values are normalised at blend time so they do not need to sum to 1.0,
    # but it is clearest if they do.
    # Override per domain in config/domain_defaults.py.
    #
    #   keyword    — BM25 exact/fuzzy term match (strong for clause numbers,
    #                defined terms, exact policy names)
    #   similarity — more_like_this vocabulary overlap (good for paraphrases
    #                and synonym-rich policy language)
    #   vector     — dense kNN semantic search (best for intent/meaning queries)
    hybrid_keyword_weight: float = 0.30
    hybrid_similarity_weight: float = 0.20
    hybrid_vector_weight: float = 0.50

    # ── Retrieval scoring weights ─────────────────────────────────────────────
    doc_semantic_weight: float = 0.48
    doc_lexical_weight: float = 0.24
    doc_title_weight: float = 0.16
    doc_metadata_weight: float = 0.12
    section_semantic_weight: float = 0.36
    section_lexical_weight: float = 0.24
    section_parent_weight: float = 0.16
    section_title_weight: float = 0.12
    section_metadata_weight: float = 0.12

    # ── Answerability thresholds ──────────────────────────────────────────────
    answerability_min_section_score: float = 0.24
    answerability_high_confidence_score: float = 0.50
    answerability_min_support_matches: int = 1
    answerability_min_support_matches_multi_doc: int = 2
    answerability_min_exact_evidence_matches: int = 1
    exact_query_section_parent_weight_scale: float = 0.75

    # ── Grounding guard ───────────────────────────────────────────────────────
    # After the LLM produces an answer, a cheap second call checks whether every
    # factual claim is supported by the evidence. Flags the answer if not.
    grounding_guard_enabled: bool = True
    grounding_guard_max_output_tokens: int = 20

    # ── Domain profile (computed properties) ──────────────────────────────────

    @property
    def domain_profile(self):
        from policygpt.core.domain import get_domain_profile
        return get_domain_profile(self.domain_type)

    @property
    def domain_context(self) -> str:
        return self.domain_profile.domain_context

    # ── Initialisation ────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Validate and resolve ai_profile
        ai_profile = (self.ai_profile or "openai").strip().lower()
        ai_preset = AI_PROFILE_PRESETS.get(ai_profile)
        if ai_preset is None:
            supported = ", ".join(sorted(AI_PROFILE_PRESETS))
            raise ValueError(f"Config.ai_profile must be one of: {supported}")

        # Validate and resolve accuracy_profile
        accuracy_profile = (self.accuracy_profile or "high").strip().lower()
        accuracy_preset = ACCURACY_PROFILE_PRESETS.get(accuracy_profile)
        if accuracy_preset is None:
            supported = ", ".join(sorted(ACCURACY_PROFILE_PRESETS))
            raise ValueError(f"Config.accuracy_profile must be one of: {supported}")

        # Validate and resolve runtime_cost_profile
        runtime_cost_profile = (self.runtime_cost_profile or "standard").strip().lower()
        runtime_cost_preset = RUNTIME_COST_PROFILE_PRESETS.get(runtime_cost_profile)
        if runtime_cost_preset is None:
            supported = ", ".join(sorted(RUNTIME_COST_PROFILE_PRESETS))
            raise ValueError(f"Config.runtime_cost_profile must be one of: {supported}")

        object.__setattr__(self, "ai_profile", ai_profile)
        object.__setattr__(self, "accuracy_profile", accuracy_profile)
        object.__setattr__(self, "runtime_cost_profile", runtime_cost_profile)
        object.__setattr__(self, "ai_provider", ai_preset["ai_provider"])
        object.__setattr__(self, "chat_model", (self.chat_model or "").strip() or ai_preset["chat_model"])
        object.__setattr__(self, "embedding_model", (self.embedding_model or "").strip() or ai_preset["embedding_model"])
        object.__setattr__(self, "bedrock_gpt_model_size", ai_preset["bedrock_gpt_model_size"])

        # Apply accuracy preset — preserve any field already tuned away from
        # the "high" baseline so explicit overrides are not clobbered.
        high_accuracy_preset = ACCURACY_PROFILE_PRESETS["high"]
        for field_name, preset_value in accuracy_preset.items():
            if getattr(self, field_name) == high_accuracy_preset[field_name]:
                object.__setattr__(self, field_name, preset_value)

        # Apply runtime cost preset
        standard_runtime_cost_preset = RUNTIME_COST_PROFILE_PRESETS["standard"]
        explicit_runtime_bool_fields = {
            "include_document_metadata_in_answers",
            "include_section_metadata_in_answers",
            "include_document_orientation_in_answers",
            "include_section_orientation_in_answers",
        }
        if runtime_cost_profile == "standard":
            for field_name, preset_value in standard_runtime_cost_preset.items():
                if field_name in explicit_runtime_bool_fields and getattr(self, field_name) is None:
                    object.__setattr__(self, field_name, preset_value)
        else:
            for field_name, preset_value in runtime_cost_preset.items():
                current_value = getattr(self, field_name)
                if field_name in explicit_runtime_bool_fields:
                    if current_value is None:
                        object.__setattr__(self, field_name, preset_value)
                    continue
                baseline_value = accuracy_preset.get(field_name, standard_runtime_cost_preset[field_name])
                if current_value == baseline_value:
                    object.__setattr__(self, field_name, preset_value)

        # Scale up limits for the 120B model's larger context window.
        # These are floor values — already-tuned-higher fields are left alone.
        if self.bedrock_gpt_model_size == "120b":
            _120b_floors: dict[str, int] = {
                "chat_max_output_tokens": 6000,
                "exact_answer_evidence_char_limit": 3000,
                "broad_answer_evidence_char_limit": 2500,
                "evidence_chunk_char_limit": 2000,
                "answer_evidence_block_limit_exact": 4,
                "answer_evidence_block_limit_broad": 4,
                "doc_summary_input_token_budget": 12000,
                "doc_summary_combine_token_budget": 9000,
                "answer_context_doc_summary_char_limit": 500,
                "small_section_full_text_chars": 3000,
            }
            for field_name, floor_value in _120b_floors.items():
                if getattr(self, field_name) < floor_value:
                    object.__setattr__(self, field_name, floor_value)

        # Apply domain-specific Config overrides (config/domain_defaults.py).
        # Runs last so values compound on top of model-floor adjustments above.
        domain_overrides = DOMAIN_CONFIG_OVERRIDES.get(self.domain_type, {})

        # chat_max_output_tokens — treated as a floor (higher of model or domain wins).
        if "chat_max_output_tokens" in domain_overrides:
            object.__setattr__(
                self,
                "chat_max_output_tokens",
                max(self.chat_max_output_tokens, int(domain_overrides["chat_max_output_tokens"])),
            )

        # Boolean / scalar domain overrides — only apply when the field is
        # still at its dataclass default (respects explicit constructor args).
        for field_name in (
            "ocr_enabled",
            "ocr_min_confidence",
            "hybrid_keyword_weight",
            "hybrid_similarity_weight",
            "hybrid_vector_weight",
        ):
            if field_name in domain_overrides:
                default_value = Config.__dataclass_fields__[field_name].default
                if getattr(self, field_name) == default_value:
                    object.__setattr__(self, field_name, domain_overrides[field_name])

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables, falling back to defaults."""
        base = cls()
        debug_log_dir_env = os.getenv("POLICY_GPT_DEBUG_LOG_DIR")
        debug_env = os.getenv("POLICY_GPT_DEBUG")
        return cls(
            ai_profile=os.getenv("POLICY_GPT_AI_PROFILE", base.ai_profile).strip() or base.ai_profile,
            accuracy_profile=os.getenv("POLICY_GPT_ACCURACY_PROFILE", base.accuracy_profile).strip() or base.accuracy_profile,
            runtime_cost_profile=os.getenv("POLICY_GPT_RUNTIME_COST_PROFILE", base.runtime_cost_profile).strip() or base.runtime_cost_profile,
            chat_model=base.chat_model,
            embedding_model=base.embedding_model,
            public_base_url=os.getenv("POLICY_GPT_PUBLIC_BASE_URL", base.public_base_url).rstrip("/"),
            bedrock_region=os.getenv("AWS_BEDROCK_REGION", os.getenv("AWS_REGION", base.bedrock_region)).strip() or base.bedrock_region,
            usd_to_inr_exchange_rate=_env_float("POLICY_GPT_USD_TO_INR_RATE", base.usd_to_inr_exchange_rate),
            doc_summary_input_token_budget=_env_int("POLICY_GPT_DOC_SUMMARY_INPUT_TOKEN_BUDGET", base.doc_summary_input_token_budget),
            doc_summary_combine_token_budget=_env_int("POLICY_GPT_DOC_SUMMARY_COMBINE_TOKEN_BUDGET", base.doc_summary_combine_token_budget),
            section_summary_input_token_budget=_env_int("POLICY_GPT_SECTION_SUMMARY_INPUT_TOKEN_BUDGET", base.section_summary_input_token_budget),
            min_recursive_summary_token_budget=_env_int("POLICY_GPT_MIN_RECURSIVE_SUMMARY_TOKEN_BUDGET", base.min_recursive_summary_token_budget),
            skip_section_summary=bool(_env_bool("POLICY_GPT_SKIP_SECTION_SUMMARY", base.skip_section_summary)),
            doc_summary_max_output_tokens=_env_int("POLICY_GPT_DOC_SUMMARY_MAX_OUTPUT_TOKENS", base.doc_summary_max_output_tokens),
            doc_summary_chunk_max_output_tokens=_env_int("POLICY_GPT_DOC_SUMMARY_CHUNK_MAX_OUTPUT_TOKENS", base.doc_summary_chunk_max_output_tokens),
            section_summary_max_output_tokens=_env_int("POLICY_GPT_SECTION_SUMMARY_MAX_OUTPUT_TOKENS", base.section_summary_max_output_tokens),
            answer_context_doc_summary_char_limit=_env_int("POLICY_GPT_ANSWER_CONTEXT_DOC_SUMMARY_CHAR_LIMIT", base.answer_context_doc_summary_char_limit),
            recent_chat_message_char_limit=_env_int("POLICY_GPT_RECENT_CHAT_MESSAGE_CHAR_LIMIT", base.recent_chat_message_char_limit),
            include_document_metadata_in_answers=_env_bool("POLICY_GPT_INCLUDE_DOCUMENT_METADATA_IN_ANSWERS", base.include_document_metadata_in_answers),
            include_section_metadata_in_answers=_env_bool("POLICY_GPT_INCLUDE_SECTION_METADATA_IN_ANSWERS", base.include_section_metadata_in_answers),
            include_document_orientation_in_answers=_env_bool("POLICY_GPT_INCLUDE_DOCUMENT_ORIENTATION_IN_ANSWERS", base.include_document_orientation_in_answers),
            include_section_orientation_in_answers=_env_bool("POLICY_GPT_INCLUDE_SECTION_ORIENTATION_IN_ANSWERS", base.include_section_orientation_in_answers),
            debug_log_dir=base.debug_log_dir if debug_log_dir_env is None else debug_log_dir_env.strip(),
            debug=base.debug if debug_env is None else debug_env.strip().lower() in {"1", "true", "yes", "on"},
            domain_type=os.getenv("POLICY_GPT_DOMAIN_TYPE", base.domain_type).strip() or base.domain_type,
            # OpenSearch credentials — read from env only, never from code defaults
            opensearch_host=os.getenv("OS_HOST", base.opensearch_host).strip(),
            opensearch_port=_env_int("OS_PORT", base.opensearch_port),
            opensearch_username=os.getenv("OS_USERNAME", base.opensearch_username).strip(),
            opensearch_password=os.getenv("OS_PASSWORD", base.opensearch_password),
            opensearch_use_ssl=bool(_env_bool("OS_USE_SSL", base.opensearch_use_ssl)),
            opensearch_verify_certs=bool(_env_bool("OS_VERIFY_CERTS", base.opensearch_verify_certs)),
            opensearch_index_prefix=os.getenv("OS_INDEX_PREFIX", base.opensearch_index_prefix).strip() or base.opensearch_index_prefix,
            ingestion_user_ids=tuple(
                uid.strip()
                for uid in os.getenv("POLICY_GPT_INGESTION_USER_IDS", "").split(",")
                if uid.strip()
            ),
        )
