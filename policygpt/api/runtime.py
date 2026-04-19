from threading import RLock, Thread

from fastapi import HTTPException

from policygpt.core.bot import PolicyGPTBot
from policygpt.config import Config
from policygpt.observability.pricing.pricing_loader import ModelPricingLoader
from policygpt.observability.usage_metrics import LLMUsageTracker


class ServerRuntime:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = RLock()
        self.bot: PolicyGPTBot | None = None
        self.usage_tracker = LLMUsageTracker(
            config.ai.chat_model,
            usd_to_inr_exchange_rate=config.storage.usd_to_inr_exchange_rate,
        )
        self.pricing_loader = ModelPricingLoader()
        self.status = "starting"
        self.error: str | None = None
        self.worker: Thread | None = None
        # Ingestion progress
        self.indexing_processed_files = 0
        self.indexing_total_files = 0
        self.indexing_current_file: str | None = None
        self.ingestion_running = False
        # Cached counts — updated by the progress callback (in-memory, no OpenSearch query).
        self._cached_document_count: int = 0
        self._cached_section_count: int = 0

    def start_indexing(self) -> None:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return

            self.bot = None
            self.usage_tracker.reset(
                self.config.ai.chat_model,
                usd_to_inr_exchange_rate=self.config.storage.usd_to_inr_exchange_rate,
            )
            self.usage_tracker.set_pricing_snapshot(self.pricing_loader.load_snapshot(self.config))
            self.status = "in_progress"
            self.error = None
            self.indexing_processed_files = 0
            self.indexing_total_files = 0
            self.indexing_current_file = None
            self.ingestion_running = False
            self._cached_document_count = 0
            self._cached_section_count = 0
            self.worker = Thread(target=self._initialize_worker, daemon=True)
            self.worker.start()

    def require_bot(self) -> PolicyGPTBot:
        if self.bot is None or self.status not in ("ready", "ingesting"):
            if self.status == "in_progress":
                detail = self.progress_detail()
            else:
                detail = self.error or "Policy GPT is still initializing."
            raise HTTPException(status_code=503, detail=detail)
        return self.bot

    def progress_detail(self) -> str:
        if self.indexing_total_files <= 0:
            return "Indexing is in progress."
        processed_files = min(self.indexing_processed_files, self.indexing_total_files)
        current_step = self.indexing_current_file or "the next indexing step"
        return f"Indexing is in progress ({processed_files}/{self.indexing_total_files}). Current step: {current_step}"

    def progress_payload(self) -> dict:
        total_files = self.indexing_total_files
        processed_files = min(self.indexing_processed_files, total_files) if total_files else 0
        percent = round((processed_files / total_files) * 100, 1) if total_files else 0.0
        return {
            "processed_files": processed_files,
            "total_files": total_files,
            "current_file": self.indexing_current_file,
            "percent": percent,
        }

    def get_document_count(self) -> int:
        """Cached document count — updated by the ingestion progress callback.
        Returns in-memory count immediately without blocking the async event loop."""
        if self._cached_document_count > 0:
            return self._cached_document_count
        bot = self.bot
        return len(bot.documents) if bot is not None else 0

    def get_section_count(self) -> int:
        """Cached section count — updated by the ingestion progress callback.
        Returns in-memory count immediately without blocking the async event loop."""
        if self._cached_section_count > 0:
            return self._cached_section_count
        bot = self.bot
        return len(bot.sections) if bot is not None else 0

    # ── Worker ────────────────────────────────────────────────────────────────

    def _initialize_worker(self) -> None:
        """
        Two-phase startup:
          Phase 1 — Create the bot (fast).  If OpenSearch is reachable the bot
                    is marked "ready" immediately so chat/search work at once.
          Phase 2 — Run the ingestion pipeline in the same thread.  Any new or
                    changed documents are indexed to OpenSearch.  The bot stays
                    in "ready" state throughout; ingestion progress is surfaced
                    via the /api/health endpoint.
        """
        try:
            # ── Phase 1: create the bot ───────────────────────────────────────
            print(f"[Policy GPT] Starting up — document folder: {self.config.storage.document_folder}", flush=True)

            from policygpt.factory import _build_thread_repo
            thread_repo = _build_thread_repo(self.config)

            bot = PolicyGPTBot(
                config=self.config,
                usage_tracker=self.usage_tracker,
                thread_repo=thread_repo,
            )

            with self.lock:
                self.bot = bot
                self.status = "ready"
                self.error = None

            print("[Policy GPT] Bot ready.  Starting background ingestion …", flush=True)

            # ── Phase 2: ingest (runs while status == "ready") ────────────────
            from policygpt.ingestion import IngestionPipeline
            from policygpt.ingestion.readers import FolderReader

            user_ids = list(self.config.ingestion.ingestion_user_ids)
            domain = self.config.domain_type

            with self.lock:
                self.ingestion_running = True
                self.status = "ingesting"

            reader = FolderReader(
                folder_path=self.config.storage.document_folder,
                user_ids=user_ids,
                domain=domain,
            )
            pipeline = IngestionPipeline.from_corpus(
                corpus=bot.corpus,
                reader=reader,
                default_user_ids=user_ids,
                default_domain=domain,
            )
            pipeline.run(progress_callback=self._update_progress)

            with self.lock:
                self.ingestion_running = False
                self.status = "ready"
                self.indexing_current_file = None

            print("[Policy GPT] Ingestion complete.", flush=True)

        except Exception as exc:
            with self.lock:
                if self.bot is None:
                    # Failed before bot was created — hard error
                    self.status = "error"
                    self.error = f"{type(exc).__name__}: {exc}"
                else:
                    # Bot is up but ingestion failed — stay ready, log warning
                    self.status = "ready"
                    self.ingestion_running = False
                    self.indexing_current_file = None
                    print(f"[Policy GPT] Ingestion error (bot still serving): {exc}", flush=True)
        finally:
            with self.lock:
                self.worker = None

    def _update_progress(
        self,
        processed_files: int,
        total_files: int,
        current_file: str | None,
        document_count: int = 0,
        section_count: int = 0,
    ) -> None:
        with self.lock:
            self.indexing_processed_files = processed_files
            self.indexing_total_files = total_files
            self.indexing_current_file = current_file
            if document_count > 0:
                self._cached_document_count = document_count
            if section_count > 0:
                self._cached_section_count = section_count

    def usage_payload(self) -> dict:
        return self.usage_tracker.snapshot()
