"""Re-export taxonomy utilities from the canonical location.

The implementation lives in policygpt.extraction.taxonomy to avoid breaking
existing imports. Move the body here when the extraction/ package is retired.
"""

from policygpt.extraction.taxonomy import (  # noqa: F401
    STOPWORDS,
    AUDIENCE_KEYWORDS,
    DOCUMENT_TYPE_KEYWORDS,
    SECTION_TYPE_KEYWORDS,
    DOMAIN_TOPIC_SYNONYMS,
    INTENT_PATTERNS,
    detect_matching_labels,
    humanize_term,
    is_informative_term,
    keywordize_text,
    normalize_text,
    tokenize_text,
    unique_preserving_order,
)
