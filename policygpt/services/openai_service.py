"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.ai.providers.openai_provider import OpenAIService
from policygpt.core.ai.base import AIRequestTooLargeError as OpenAIRequestTooLargeError

__all__ = ["OpenAIService", "OpenAIRequestTooLargeError"]
