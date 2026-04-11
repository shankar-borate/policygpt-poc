"""Backward-compatibility shim — import from canonical location."""
from policygpt.api.routes.chat import PolicyApiServer, ChatRequest

__all__ = ["PolicyApiServer", "ChatRequest"]
