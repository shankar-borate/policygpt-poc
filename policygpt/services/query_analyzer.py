"""Backward-compatibility shim — import from canonical location."""
from policygpt.core.retrieval.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    normalize_numeric_expressions,
    detect_conversational_intent,
    DETAIL_PHRASES,
    MULTI_DOC_COVERAGE_PHRASES,
    CONTEXT_REFERENCE_PHRASES,
    CONTEXT_REFERENCE_TOKENS,
    DOCUMENT_LOOKUP_PHRASES,
    DOCUMENT_LOOKUP_NOUNS,
    INTENT_TO_SECTION_TYPES,
)

__all__ = [
    "QueryAnalysis",
    "QueryAnalyzer",
    "normalize_numeric_expressions",
    "detect_conversational_intent",
    "DETAIL_PHRASES",
    "MULTI_DOC_COVERAGE_PHRASES",
    "CONTEXT_REFERENCE_PHRASES",
    "CONTEXT_REFERENCE_TOKENS",
    "DOCUMENT_LOOKUP_PHRASES",
    "DOCUMENT_LOOKUP_NOUNS",
    "INTENT_TO_SECTION_TYPES",
]
