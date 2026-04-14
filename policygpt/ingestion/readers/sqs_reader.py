"""SQSReader — reads IngestMessages from an AWS SQS queue.

STATUS: placeholder — not yet implemented.

Expected SQS message body (JSON)
---------------------------------
{
    "content":      "<base64-encoded bytes OR plain text string>",
    "content_type": "html | text | pdf | ppt | image",
    "file_name":    "document.html",
    "source_path":  "s3://my-bucket/documents/document.html",
    "domain":       "policy",
    "user_ids":     ["alice", "bob"],
    "metadata":     {}          // optional, any extra fields
}

Implementation notes (when building this out)
----------------------------------------------
- Use boto3.client("sqs") with long-polling (WaitTimeSeconds=20).
- Delete each message from the queue only after successful ingestion.
- Dead-letter any message that fails after MAX_RETRIES attempts.
- Base64-decode content when content_type is "pdf", "ppt", or "image".
- Respect visibility timeout: delete before it expires or extend it.
"""

from __future__ import annotations

import logging
from typing import Iterator

from policygpt.ingestion.readers.base import IngestMessage, Reader

logger = logging.getLogger(__name__)


class SQSReader(Reader):
    """Polls an AWS SQS queue and yields IngestMessages.

    Parameters
    ----------
    queue_url:   Full SQS queue URL.
    region_name: AWS region (e.g. "ap-south-1").
    max_messages: Maximum messages to fetch per poll (1–10).
    wait_seconds: Long-poll duration in seconds (0–20).
    """

    def __init__(
        self,
        queue_url: str,
        region_name: str = "ap-south-1",
        max_messages: int = 10,
        wait_seconds: int = 20,
    ) -> None:
        self.queue_url = queue_url
        self.region_name = region_name
        self.max_messages = max_messages
        self.wait_seconds = wait_seconds

    def read(self) -> Iterator[IngestMessage]:
        raise NotImplementedError(
            "SQSReader is not yet implemented. "
            "See the docstring in this file for the expected message format "
            "and implementation notes."
        )
