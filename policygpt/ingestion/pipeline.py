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
from typing import Callable, TYPE_CHECKING

from policygpt.ingestion.converters.registry import HtmlConverterRegistry
from policygpt.ingestion.pipeline_extractors.registry import ExtractorRegistry
from policygpt.ingestion.readers.base import IngestMessage, Reader
from policygpt.ingestion.rewriter.policy_rewriter import PolicyRewriter

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
        rewriter: PolicyRewriter | None = None,
        html_converter_registry: HtmlConverterRegistry | None = None,
    ) -> None:
        self._corpus = corpus
        self._reader = reader
        self._registry = extractor_registry
        self._default_user_ids = default_user_ids or []
        self._default_domain = default_domain
        self._rewriter = rewriter
        self._html_converter_registry = html_converter_registry
        self._pending_extra_messages: list[IngestMessage] = []

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

        # Build rewriter only when rewrite_policies_enabled=True (ingestion config).
        # rewrite_save_to_disk (debug/log config) controls whether the improved
        # HTML is persisted to {debug_log_dir}/improved/ or kept in-memory only.
        rewriter: PolicyRewriter | None = None
        if corpus.config.ingestion.rewrite_policies_enabled:
            debug_log_dir = (corpus.config.storage.debug_log_dir or "").strip()
            save_to_disk  = corpus.config.storage.rewrite_save_to_disk and bool(debug_log_dir)
            output_dir    = Path(debug_log_dir) / "improved" if save_to_disk else None
            rewriter = PolicyRewriter(
                output_dir=output_dir,
                save_to_disk=save_to_disk,
            )
            logger.info(
                "PolicyRewriter enabled — save_to_disk=%s output=%s",
                save_to_disk, output_dir,
            )

        # Build the multi-format HTML converter registry when enabled.
        # Converted files are saved to {debug_log_dir}/html/ and cached for
        # subsequent runs.  Per-format flags in config control which converters
        # are active (pdf_to_html_enabled, docx_to_html_enabled, etc.).
        html_converter_registry: HtmlConverterRegistry | None = None
        if corpus.config.ingestion.to_html_enabled:
            debug_log_dir = (corpus.config.storage.debug_log_dir or "").strip()
            if debug_log_dir:
                html_dir = Path(debug_log_dir) / "html"

                # Build the set of content-types to skip based on per-format flags.
                cfg = corpus.config
                skip_cts: set[str] = set()
                if not cfg.ingestion.pdf_to_html_enabled:
                    skip_cts.update({"pdf"})
                if not cfg.ingestion.docx_to_html_enabled:
                    skip_cts.update({"docx", "doc"})
                if not cfg.ingestion.pptx_to_html_enabled:
                    skip_cts.update({"pptx", "ppt"})
                if not cfg.ingestion.excel_to_html_enabled:
                    skip_cts.update({"xlsx", "xls"})
                if not cfg.ingestion.image_to_html_enabled:
                    skip_cts.update({"png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp"})

                # Build vision describer — LLM-based description of image-only pages.
                from policygpt.ingestion.converters.vision import build_vision_describer
                vision_describer = None
                if cfg.ingestion.vision_provider:
                    try:
                        vision_describer = build_vision_describer(
                            provider=cfg.ingestion.vision_provider,
                            model=cfg.ingestion.vision_model,
                        )
                    except ValueError as exc:
                        logger.warning("VisionDescriber not built: %s", exc)

                # Build OCR extractor — text fallback for image-only pages.
                from policygpt.ingestion.extraction.ocr import build_ocr_extractor
                ocr = None
                if cfg.ingestion.ocr_enabled:
                    try:
                        ocr = build_ocr_extractor(
                            provider=cfg.ingestion.ocr_provider.value
                            if hasattr(cfg.ingestion.ocr_provider, "value")
                            else str(cfg.ingestion.ocr_provider),
                            region=cfg.ai.bedrock_region,
                            min_confidence=cfg.ingestion.ocr_min_confidence,
                        )
                    except ValueError as exc:
                        logger.warning("OcrExtractor not built: %s", exc)

                # Build explainer factory when explain_enabled is set.
                explainer = None
                if cfg.ingestion.explain_enabled:
                    from policygpt.ingestion.explainers.factory import ExplainerFactory
                    from policygpt.ingestion.explainers.rules import ExplainRules
                    explainer = ExplainerFactory(
                        rules=ExplainRules(min_chars=cfg.ingestion.explain_min_chars),
                        vision=vision_describer,
                        ocr=ocr,
                        ai=corpus.ai,
                    )
                    logger.info(
                        "ExplainerFactory enabled — min_chars=%d vision=%s",
                        cfg.ingestion.explain_min_chars,
                        cfg.ingestion.vision_provider or "disabled",
                    )

                html_converter_registry = HtmlConverterRegistry(
                    output_dir=html_dir,
                    skip_content_types=frozenset(skip_cts),
                    vision_describer=vision_describer,
                    ocr=ocr,
                    explainer=explainer,
                )
                logger.info(
                    "HtmlConverterRegistry enabled — output=%s formats=%s vision=%s ocr=%s",
                    html_dir,
                    sorted(html_converter_registry.supported_content_types),
                    cfg.ingestion.vision_provider or "disabled",
                    cfg.ingestion.ocr_provider if cfg.ingestion.ocr_enabled else "disabled",
                )

        return cls(
            corpus=corpus,
            reader=reader,
            extractor_registry=registry,
            default_user_ids=default_user_ids,
            default_domain=default_domain,
            rewriter=rewriter,
            html_converter_registry=html_converter_registry,
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
        import time as _time
        messages = list(self._reader.read())
        total = len(messages)
        counts = {"processed": 0, "skipped": 0, "failed": 0}

        print(f"[Ingestion] {total} file(s) found — starting …", flush=True)
        self._emit(progress_callback, 0, total, "Starting ingestion")

        for index, message in enumerate(messages, start=1):
            label = message.file_name
            print(f"[Ingestion] [{index}/{total}] {label} …", flush=True)
            self._emit(progress_callback, index - 1, total, f"{label} — reading")

            t0 = _time.perf_counter()
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

            elapsed = _time.perf_counter() - t0
            # corpus.ingest_file returns "ingested" on success; normalise to "processed"
            bucket = "processed" if status == "ingested" else status
            counts[bucket] = counts.get(bucket, 0) + 1
            print(
                f"[Ingestion] [{index}/{total}] {label} → {bucket} ({elapsed:.1f}s) — {detail}",
                flush=True,
            )
            self._emit(progress_callback, index, total, f"{label} — {detail}")

            # Drain any extra messages produced by multi-sheet Excel splitting.
            while self._pending_extra_messages:
                extra = self._pending_extra_messages.pop(0)
                extra_label = extra.file_name
                print(f"[Ingestion] [{index}/{total}] {extra_label} (extra sheet) …", flush=True)
                t1 = _time.perf_counter()
                try:
                    s2, d2 = self._ingest_one(
                        message=extra,
                        progress_callback=progress_callback,
                        processed_index=index - 1,
                        total=total,
                    )
                except Exception as exc:
                    logger.exception("Unhandled error ingesting extra sheet %s", extra.source_path)
                    s2, d2 = "failed", f"{type(exc).__name__}: {exc}"
                e2 = _time.perf_counter() - t1
                b2 = "processed" if s2 == "ingested" else s2
                counts[b2] = counts.get(b2, 0) + 1
                print(
                    f"[Ingestion] [{index}/{total}] {extra_label} → {b2} ({e2:.1f}s) — {d2}",
                    flush=True,
                )

        if counts["processed"] > 0:
            print("[Ingestion] Rebuilding retrieval indexes …", flush=True)
            self._emit(progress_callback, total, total, "Rebuilding retrieval indexes")
            self._corpus.rebuild_indexes()

        print(
            f"[Ingestion] Complete — {counts['processed']} processed, "
            f"{counts['skipped']} skipped, {counts.get('failed', 0)} failed",
            flush=True,
        )
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
        # ── Step 1: Convert source format to HTML ────────────────────────────
        # Runs before the rewriter so all formats land as HTML in the rest of
        # the pipeline.  Saves converted file to {debug_log_dir}/html/{stem}.html.
        # The message content_type is updated to "html" so the HTML extractor
        # (with table support) picks it up downstream.
        # For multi-sheet Excel files convert_all() returns multiple (path, html)
        # pairs — only the first is processed here; additional ones are returned
        # as extra messages to be ingested by the caller.
        import time as _time

        if (
            self._html_converter_registry is not None
            and self._html_converter_registry.supports(message.content_type)
        ):
            # Capture the original path before replacing with the HTML path so
            # reference links in the UI can open the source file (PDF, XLSX, etc.).
            original_path = message.source_path

            all_conversions = self._html_converter_registry.convert_all(
                message.content_type, message.source_path
            )
            # Use first result for this message; extras are handled via _extra_messages
            html_path, html_content = all_conversions[0]
            logger.debug(
                "HtmlConverterRegistry: converted %s (%s) → %s (%d part(s))",
                message.file_name, message.content_type, html_path, len(all_conversions),
            )
            message.content = html_content
            message.content_type = "html"
            message.original_source_path = original_path
            message.source_path = html_path
            message.file_name = Path(html_path).name

            # Store extra per-sheet messages on the instance for run() to pick up.
            if len(all_conversions) > 1:
                from policygpt.ingestion.readers.base import IngestMessage as _IM
                self._pending_extra_messages.extend(
                    _IM(
                        source_path=hp,
                        content=hc,
                        content_type="html",
                        file_name=Path(hp).name,
                        user_ids=message.user_ids,
                        domain=message.domain,
                        original_source_path=original_path,
                    )
                    for hp, hc in all_conversions[1:]
                )

        if not self._registry.supports(message.content_type):
            reason = f"no extractor for content_type={message.content_type!r}"
            logger.debug("Skipping %s: %s", message.source_path, reason)
            return ("skipped", reason)

        # ── Step 2: Rewrite HTML for PolicyGPT optimisation ──────────────────
        if self._rewriter is not None and message.content_type == "html":
            print(f"  [Rewrite]  {message.file_name} …", flush=True)
            _t = _time.perf_counter()
            improved_path, improved_content = self._rewriter.rewrite(message.source_path)
            message.content = improved_content
            if improved_path != message.source_path:
                message.source_path = improved_path
            print(f"  [Rewrite]  {message.file_name} — done in {_time.perf_counter()-_t:.1f}s", flush=True)
            logger.debug("PolicyRewriter: applied to %s", message.file_name)

        user_ids: list[str] = message.user_ids or self._default_user_ids
        domain: str = message.domain or self._default_domain

        print(f"  [Ingest]   {message.file_name} — extracting + embedding …", flush=True)
        return self._corpus.ingest_file(
            path=message.source_path,
            progress_callback=progress_callback,
            processed_files=processed_index,
            total_files=total,
            user_ids=user_ids,
            domain=domain,
            original_source_path=message.original_source_path,
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
