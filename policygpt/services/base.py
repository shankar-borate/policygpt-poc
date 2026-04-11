"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.ai.base import AIRequestTooLargeError, AIService

__all__ = ["AIRequestTooLargeError", "AIService"]
