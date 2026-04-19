"""AI model and rate-limit configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIConfig:
    # "bedrock" | "openai" | "anthropic"
    ai_provider: str | None = None

    # Chat model ID — depends on provider.
    # bedrock examples : "amazon.nova-pro-v1:0", "anthropic.claude-3-5-sonnet-20241022-v2:0"
    # openai  examples : "gpt-4o", "gpt-4o-mini"
    # anthropic examples: "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
    chat_model: str | None = None

    # Embedding model ID — depends on provider.
    # bedrock example : "amazon.titan-embed-text-v2:0"
    # openai  example : "text-embedding-3-small"
    embedding_model: str | None = None

    # Bedrock model size hint used to select the right model variant.
    # "small" | "medium" | "large" | "120b" (or leave None for provider default)
    bedrock_gpt_model_size: str | None = None

    # AWS region for Bedrock API calls.
    # Examples: "ap-south-1" | "us-east-1" | "eu-west-1"
    bedrock_region: str = "ap-south-1"

    ai_rate_limit_retries: int = 2
    ai_rate_limit_backoff_seconds: float = 8.0
