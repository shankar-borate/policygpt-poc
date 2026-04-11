"""Backward-compatibility shim — import from canonical location."""
from policygpt.extraction.taxonomy import (
    STOPWORDS,
    GENERIC_POLICY_TERMS,
    DOMAIN_TOPIC_SYNONYMS,
    INTENT_PATTERNS,
    DOCUMENT_TYPE_KEYWORDS,
    SECTION_TYPE_KEYWORDS,
    AUDIENCE_KEYWORDS,
    normalize_text,
    tokenize_text,
    keywordize_text,
    unique_preserving_order,
    humanize_term,
    is_informative_term,
    detect_matching_labels,
)

__all__ = [
    "STOPWORDS",
    "GENERIC_POLICY_TERMS",
    "DOMAIN_TOPIC_SYNONYMS",
    "INTENT_PATTERNS",
    "DOCUMENT_TYPE_KEYWORDS",
    "SECTION_TYPE_KEYWORDS",
    "AUDIENCE_KEYWORDS",
    "normalize_text",
    "tokenize_text",
    "keywordize_text",
    "unique_preserving_order",
    "humanize_term",
    "is_informative_term",
    "detect_matching_labels",
]
