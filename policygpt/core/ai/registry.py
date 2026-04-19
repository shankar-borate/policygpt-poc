"""AI provider factory — builds the right AIService from Config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from policygpt.core.ai.base import AIService

if TYPE_CHECKING:
    from policygpt.config import Config
    from policygpt.observability.usage_metrics import LLMUsageTracker


def build_ai_service(config: "Config", usage_tracker: "LLMUsageTracker | None" = None) -> AIService:
    """Return the correct AIService implementation for the given Config."""
    if config.ai.ai_provider == "openai":
        from policygpt.core.ai.providers.openai_provider import OpenAIService
        return OpenAIService(
            chat_model=config.ai.chat_model,
            embedding_model=config.ai.embedding_model,
            rate_limit_retries=config.ai.ai_rate_limit_retries,
            rate_limit_backoff_seconds=config.ai.ai_rate_limit_backoff_seconds,
            usage_tracker=usage_tracker,
        )

    from policygpt.core.ai.providers.bedrock_provider import BedrockService
    return BedrockService(
        chat_model=config.ai.chat_model,
        embedding_model=config.ai.embedding_model,
        region_name=config.ai.bedrock_region,
        rate_limit_retries=config.ai.ai_rate_limit_retries,
        rate_limit_backoff_seconds=config.ai.ai_rate_limit_backoff_seconds,
        usage_tracker=usage_tracker,
    )
