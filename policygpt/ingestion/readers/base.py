"""Reader contract — IngestMessage dataclass and Reader abstract base class.

Every document source (folder, SQS, API, …) implements Reader and yields
IngestMessage objects.  The pipeline and extractors downstream only ever
see IngestMessage — they are completely decoupled from the source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class IngestMessage:
    """Single document unit flowing through the ingestion pipeline.

    Attributes
    ----------
    content:
        Raw file bytes (pdf, image, ppt) or decoded text (html, txt).
        Callers should check content_type before casting.
    content_type:
        One of: "html" | "text" | "pdf" | "ppt" | "image".
        Drives extractor selection in ExtractorRegistry.
    file_name:
        Original file name including extension (e.g. "policy.html").
    source_path:
        Fully qualified origin — local file path, S3 key, or API URL.
        Used as a stable identifier for the document in OpenSearch.
    domain:
        Domain name registered in core/domain/ (e.g. "policy", "contest").
    user_ids:
        Access-control list.  OpenSearch filters restrict retrieval to users
        whose ID appears in this list at index time.
    metadata:
        Source-specific extras (SQS message ID, API correlation ID, S3 ETag,
        etc.) forwarded verbatim — the pipeline stores them on DocumentRecord.
    """

    content: bytes | str
    content_type: str
    file_name: str
    source_path: str
    domain: str
    user_ids: list[str]
    metadata: dict = field(default_factory=dict)
    original_source_path: str = ""   # set by pipeline when source is converted to HTML


class Reader(ABC):
    """Abstract document source.

    Subclasses implement read() and yield one IngestMessage per document.
    The pipeline calls read() once; readers may be lazy (streaming from SQS)
    or eager (scanning a folder upfront).
    """

    @abstractmethod
    def read(self) -> Iterator[IngestMessage]:
        """Yield IngestMessage objects from the underlying source."""
