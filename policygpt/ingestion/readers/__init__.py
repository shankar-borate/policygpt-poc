"""Reader layer — abstracts document sources.

Every reader yields IngestMessage objects. The rest of the pipeline never
knows whether documents came from a local folder, an SQS queue, or an API.
"""

from policygpt.ingestion.readers.base import IngestMessage, Reader
from policygpt.ingestion.readers.folder_reader import FolderReader
from policygpt.ingestion.readers.sqs_reader import SQSReader
from policygpt.ingestion.readers.api_reader import APIReader

__all__ = ["IngestMessage", "Reader", "FolderReader", "SQSReader", "APIReader"]
