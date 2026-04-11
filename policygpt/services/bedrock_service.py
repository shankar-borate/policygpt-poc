"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.ai.providers.bedrock_provider import BedrockService

__all__ = ["BedrockService"]
