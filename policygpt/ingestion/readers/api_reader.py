"""APIReader — fetches IngestMessages from an HTTP API endpoint.

STATUS: placeholder — not yet implemented.

Expected API response (JSON)
-----------------------------
Single document endpoint (GET /documents/{id}):
{
    "content":      "<base64-encoded bytes OR plain text string>",
    "content_type": "html | text | pdf | ppt | image",
    "file_name":    "document.html",
    "source_path":  "https://intranet.example.com/policies/document.html",
    "domain":       "policy",
    "user_ids":     ["alice", "bob"],
    "metadata":     {}
}

Bulk listing endpoint (GET /documents?page=1&size=50):
{
    "items": [ <same structure as above>, ... ],
    "next_page": 2   // null when exhausted
}

Implementation notes (when building this out)
----------------------------------------------
- Use urllib.request or httpx for HTTP calls (avoid adding new deps lightly).
- Support Bearer token auth via an Authorization header.
- Paginate using a "next_page" cursor returned by the listing endpoint.
- Retry transient 5xx errors with exponential back-off.
- Base64-decode content when content_type is "pdf", "ppt", or "image".
"""

from __future__ import annotations

import logging
from typing import Iterator

from policygpt.ingestion.readers.base import IngestMessage, Reader

logger = logging.getLogger(__name__)


class APIReader(Reader):
    """Fetches documents from an HTTP API and yields IngestMessages.

    Parameters
    ----------
    base_url:   Root URL of the document API (no trailing slash).
    api_token:  Bearer token for Authorization header (optional).
    page_size:  Number of documents to fetch per page.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str = "",
        page_size: int = 50,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.page_size = page_size

    def read(self) -> Iterator[IngestMessage]:
        raise NotImplementedError(
            "APIReader is not yet implemented. "
            "See the docstring in this file for the expected response format "
            "and implementation notes."
        )
