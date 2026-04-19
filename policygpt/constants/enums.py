"""Shared string-valued enumerations.

Using ``str``-based enums means existing code that compares against plain
string literals (``intent == "greeting"``) continues to work — the enum
members compare equal to their string values — allowing a gradual migration.
"""

from __future__ import annotations

from enum import Enum


class ConversationalIntent(str, Enum):
    """Intents that bypass the RAG pipeline entirely."""
    GREETING        = "greeting"
    FAREWELL        = "farewell"
    THANKS          = "thanks"
    IDENTITY        = "identity"
    CHITCHAT        = "chitchat"
    SELF_REFERENTIAL = "self_referential"  # "why did you miss X", "you forgot Y in previous answer"
    POLICY          = "policy"   # default / fallback — goes through RAG


class QueryIntent(str, Enum):
    """Semantic intents extracted from a user's policy question."""
    ELIGIBILITY        = "eligibility"
    THRESHOLD          = "threshold"
    REWARD             = "reward"
    BENEFIT            = "benefit"
    APPROVAL           = "approval"
    APPROVAL_PATH      = "approval_path"
    PROCESS            = "process"
    CHECKLIST          = "checklist"
    TIMELINE           = "timeline"
    DOCUMENTS_REQUIRED = "documents_required"
    AGGREGATE          = "aggregate"
    COMPARISON         = "comparison"
    DOCUMENT_LOOKUP    = "document_lookup"
    DETAIL             = "detail"


class SectionType(str, Enum):
    """Content classification of an extracted document section."""
    GENERAL     = "general"
    OVERVIEW    = "overview"
    ELIGIBILITY = "eligibility"
    PROCEDURE   = "procedure"
    APPROVAL    = "approval"
    EXCEPTION   = "exception"
    DEFINITION  = "definition"
    TIMELINE    = "timeline"
    THRESHOLD   = "threshold"
    ROLE        = "role"
    TABLE       = "table"
    FAQ         = "faq"


class DocumentType(str, Enum):
    """High-level classification of an ingested document."""
    DOCUMENT    = "document"
    POLICY      = "policy"
    PROCEDURE   = "procedure"
    HANDBOOK    = "handbook"
    CONTEST     = "contest"
    MANUAL      = "manual"
    GUIDE       = "guide"
    MATRIX      = "matrix"
    CHECKLIST   = "checklist"
    PROCESS     = "process"


class AIProvider(str, Enum):
    """Supported LLM / embedding back-ends."""
    OPENAI  = "openai"
    BEDROCK = "bedrock"


class AIProfile(str, Enum):
    """Pre-configured model stacks selectable via ``Config.ai_profile``."""
    OPENAI          = "openai"
    BEDROCK_20B     = "bedrock-20b"
    BEDROCK_120B    = "bedrock-120b"
    BEDROCK_SONNET  = "bedrock-claude-sonnet-4-6"
    BEDROCK_OPUS    = "bedrock-claude-opus-4-6"


class AccuracyProfile(str, Enum):
    """Retrieval quality / cost trade-off presets."""
    VERY_HIGH = "vhigh"
    HIGH      = "high"
    MEDIUM    = "medium"
    LOW       = "low"


class RuntimeCostProfile(str, Enum):
    """Per-request cost presets (independent of accuracy)."""
    STANDARD   = "standard"
    AGGRESSIVE = "aggressive"


class DomainType(str, Enum):
    """Built-in domain profiles."""
    POLICY            = "policy"
    CONTEST           = "contest"
    PRODUCT_TECHNICAL = "product_technical"


class OCRProvider(str, Enum):
    """OCR engine to use when ``ocr_enabled=True``."""
    TEXTRACT = "textract"
    CLAUDE   = "claude"


class SearchProvider(str, Enum):
    """Vector store / hybrid search back-end."""
    OPENSEARCH = "opensearch"


class ConfidenceLevel(str, Enum):
    """Answer confidence shown to the end user."""
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


class BedRockModelSize(str, Enum):
    """Open-weight model sizes hosted on AWS Bedrock."""
    SMALL  = "20b"
    LARGE  = "120b"


class EntityCategory(str, Enum):
    """Named-entity categories extracted during ingestion."""
    # Shared across domains
    ROLE         = "role"
    LOCATION     = "location"
    TIME_PERIOD  = "time_period"
    ACTION       = "action"
    ABBREVIATION = "abbreviation"
    OTHER        = "other"
    # Contest-domain specific
    REWARD   = "reward"
    PRODUCT  = "product"
    CONTEST  = "contest"
    # Policy-domain specific
    BENEFIT  = "benefit"
    PROCESS  = "process"
    POLICY   = "policy"
    # Shared numeric
    THRESHOLD = "threshold"


class FileExtension(str, Enum):
    """Lowercase file-extension strings (including the leading dot)."""
    HTML = ".html"
    HTM  = ".htm"
    TXT  = ".txt"
    PDF  = ".pdf"
    PPTX = ".pptx"
    PPT  = ".ppt"
    DOCX = ".docx"
    DOC  = ".doc"
    XLSX = ".xlsx"
    XLS  = ".xls"
    PNG  = ".png"
    JPG  = ".jpg"
    JPEG = ".jpeg"
    GIF  = ".gif"
    BMP  = ".bmp"
    TIFF = ".tiff"
    TIF  = ".tif"
    WEBP = ".webp"


class ContentType(str, Enum):
    """Logical content-type identifiers used by the ingestion pipeline."""
    HTML  = "html"
    TEXT  = "text"
    PDF   = "pdf"
    PPTX  = "pptx"
    PPT   = "ppt"
    DOCX  = "docx"
    DOC   = "doc"
    XLSX  = "xlsx"
    XLS   = "xls"
    JPG   = "jpg"
    JPEG  = "jpeg"
    PNG   = "png"
    GIF   = "gif"
    BMP   = "bmp"
    TIFF  = "tiff"
    TIF   = "tif"
    WEBP  = "webp"
