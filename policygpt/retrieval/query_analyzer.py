"""Re-export QueryAnalysis and QueryAnalyzer from their canonical location.

The canonical implementation lives in policygpt.core.retrieval.query_analyzer.
This module re-exports it so callers outside the core layer can import from
the top-level retrieval package without referencing core internals.
"""

from policygpt.core.retrieval.query_analyzer import (  # noqa: F401
    QueryAnalysis,
    QueryAnalyzer,
    detect_conversational_intent,
)

__all__ = ["QueryAnalysis", "QueryAnalyzer", "detect_conversational_intent"]
