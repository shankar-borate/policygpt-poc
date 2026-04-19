"""AI model and rate-limit configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIConfig:
    ai_provider: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    bedrock_gpt_model_size: str = ""
    bedrock_region: str = "ap-south-1"
    ai_rate_limit_retries: int = 2
    ai_rate_limit_backoff_seconds: float = 8.0
