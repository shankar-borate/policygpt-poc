"""IngestionPipeline — orchestrates reading, extraction, enrichment, and indexing.

Flow per document
-----------------
Reader → IngestMessage
  → ExtractorRegistry → ExtractedDocument
    → Enrichers (Summarizer, FaqGenerator, EntityEnricher) → EnrichedDocument
      → DocumentCorpus.index_enriched_document() (stores in memory + OpenSearch)

Current state
-------------
The enrichment logic still lives inside DocumentCorpus private methods, which
this pipeline delegates to via DocumentCorpus.ingest_file().  Once those methods
are promoted into the Enricher classes the delegation can be replaced by direct
calls, giving full separation.

Public API
----------
pipeline = IngestionPipeline.from_corpus(corpus, reader)
pipeline.run(progress_callback=...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from policygpt.ingestion.extractors.registry import ExtractorRegistry
from policygpt.ingestion.readers.base import IngestMessage, Reader

if TYPE_CHECKING:
    from policygpt.core.corpus import DocumentCorpus

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str | None, int, int], None]


class IngestionPipeline:
    """Orchestrates the full document ingestion flow.

    Reads → extracts → enriches → indexes each document that arrives from
    the configured Reader.

    Parameters
    ----------
    corpus:
        DocumentCorpus that owns the enrichment logic and the in-memory / OS
        search indexes.  This dependency will shrink as enrichment logic is
        moved into the Enricher layer.
    reader:
        Reader implementation that yields IngestMessage objects.
    extractor_registry:
        Maps content_type → Extractor.  Used to delegate format-specific
        parsing (currently the extraction result feeds back into corpus,
        but will feed the Enricher chain directly once corpus is decoupled).
    default_user_ids:
        Fallback user IDs when IngestMessage.user_ids is empty.
    default_domain:
        Fallback domain when IngestMessage.domain is empty.
    """

    def __init__(
        self,
        corpus: "DocumentCorpus",
        reader: Reader,
        extractor_registry: ExtractorRegistry,
        default_user_ids: list[str] | None = None,
        default_domain: str = "",
    ) -> None:
        self._corpus = corpus
        self._reader = reader
        self._registry = extractor_registry
        self._default_user_ids = default_user_ids or []
        self._default_domain = default_domain

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_corpus(
        cls,
        corpus: "DocumentCorpus",
        reader: Reader,
        default_user_ids: list[str] | None = None,
        default_domain: str = "",
    ) -> "IngestionPipeline":
        """Convenience factory that builds the ExtractorRegistry from corpus config."""
        registry = ExtractorRegistry(corpus.config)
        return cls(
            corpus=corpus,
            reader=reader,
            extractor_registry=registry,
            default_user_ids=default_user_ids,
            default_domain=default_domain,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Process all messages from the reader.

        Returns
        -------
        dict with keys:
            processed   — number of documents successfully ingested
            skipped     — documents skipped (empty, unsupported type, …)
            failed      — documents that raised unhandled exceptions
        """
        messages = list(self._reader.read())
        total = len(messages)
        counts = {"processed": 0, "skipped": 0, "failed": 0}

        self._emit(progress_callback, 0, total, "Starting ingestion")

        for index, message in enumerate(messages, start=1):
            label = message.file_name
            self._emit(progress_callback, index - 1, total, f"{label} — reading")

            try:
                status, detail = self._ingest_one(
                    message=message,
                    progress_callback=progress_callback,
                    processed_index=index - 1,
                    total=total,
                )
            except Exception as exc:
                logger.exception("Unhandled error ingesting %s", message.source_path)
                status, detail = "failed", f"{type(exc).__name__}: {exc}"

            # corpus.ingest_file returns "ingested" on success; normalise to "processed"
            bucket = "processed" if status == "ingested" else status
            counts[bucket] = counts.get(bucket, 0) + 1
            self._emit(progress_callback, index, total, f"{label} — {detail}")

        if counts["processed"] > 0:
            self._emit(progress_callback, total, total, "Rebuilding retrieval indexes")
            self._corpus.rebuild_indexes()

        logger.info(
            "Ingestion complete: %d processed, %d skipped, %d failed",
            counts["processed"], counts["skipped"], counts["failed"],
        )
        return counts

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ingest_one(
        self,
        message: IngestMessage,
        progress_callback: ProgressCallback | None,
        processed_index: int,
        total: int,
    ) -> tuple[str, str]:
        """Ingest a single document.

        Delegates to corpus.ingest_file() which currently contains all
        enrichment logic (summarization, FAQ, entity extraction, embedding,
        OpenSearch indexing).  As those private methods migrate into the
        Enricher classes this method will call them directly.

        Returns (status, detail) where status is one of
        "ingested" | "skipped" | "failed".
        """
        if not self._registry.supports(message.content_type):
            reason = f"no extractor for content_type={message.content_type!r}"
            logger.debug("Skipping %s: %s", message.source_path, reason)
            return ("skipped", reason)

        user_ids: list[str] = message.user_ids or self._default_user_ids
        domain: str = message.domain or self._default_domain

        return self._corpus.ingest_file(
            path=message.source_path,
            progress_callback=progress_callback,
            processed_files=processed_index,
            total_files=total,
            user_ids=user_ids,
            domain=domain,
        )

    def _emit(
        self,
        cb: ProgressCallback | None,
        current: int,
        total: int,
        label: str,
    ) -> None:
        if cb is not None:
            try:
                cb(
                    current,
                    total,
                    label,
                    len(self._corpus.documents),
                    len(self._corpus.sections),
                )
            except Exception:
                pass
