from threading import RLock, Thread

from fastapi import HTTPException

from policygpt.core.bot import PolicyGPTBot
from policygpt.config import Config
from policygpt.factory import create_ready_bot
from policygpt.observability.pricing.pricing_loader import ModelPricingLoader
from policygpt.observability.usage_metrics import LLMUsageTracker


class ServerRuntime:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = RLock()
        self.bot: PolicyGPTBot | None = None
        self.usage_tracker = LLMUsageTracker(
            config.chat_model,
            usd_to_inr_exchange_rate=config.usd_to_inr_exchange_rate,
        )
        self.pricing_loader = ModelPricingLoader()
        self.status = "starting"
        self.error: str | None = None
        self.document_folder = config.document_folder
        self.worker: Thread | None = None
        self.indexing_processed_files = 0
        self.indexing_total_files = 0
        self.indexing_current_file: str | None = None
        self.document_count = 0
        self.section_count = 0

    def start_indexing(self) -> None:
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                return

            self.bot = None
            self.usage_tracker.reset(
                self.config.chat_model,
                usd_to_inr_exchange_rate=self.config.usd_to_inr_exchange_rate,
            )
            self.usage_tracker.set_pricing_snapshot(self.pricing_loader.load_snapshot(self.config))
            self.status = "in_progress"
            self.error = None
            self.indexing_processed_files = 0
            self.indexing_total_files = 0
            self.indexing_current_file = None
            self.document_count = 0
            self.section_count = 0
            self.worker = Thread(target=self._initialize_worker, daemon=True)
            self.worker.start()

    def require_bot(self) -> PolicyGPTBot:
        if self.bot is None or self.status != "ready":
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

    def _initialize_worker(self) -> None:
        try:
            print(f"[Policy GPT] Indexing started for {self.document_folder}", flush=True)
            bot = create_ready_bot(
                folder=self.document_folder,
                progress_callback=self._update_progress,
                config=self.config,
                usage_tracker=self.usage_tracker,
            )
            with self.lock:
                self.bot = bot
                self.status = "ready"
                self.error = None
                self.indexing_processed_files = self.indexing_total_files
                self.indexing_current_file = None
                self.document_count = len(bot.documents)
                self.section_count = len(bot.sections)
            print(
                f"[Policy GPT] Indexing complete: {len(bot.documents)} documents, {len(bot.sections)} sections.",
                flush=True,
            )
        except Exception as exc:
            with self.lock:
                self.bot = None
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self.indexing_current_file = None
            print(f"[Policy GPT] Indexing failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            with self.lock:
                self.worker = None

    def _update_progress(
        self,
        processed_files: int,
        total_files: int,
        current_file: str | None,
        document_count: int,
        section_count: int,
    ) -> None:
        with self.lock:
            self.status = "in_progress"
            self.error = None
            self.indexing_processed_files = processed_files
            self.indexing_total_files = total_files
            self.indexing_current_file = current_file
            self.document_count = document_count
            self.section_count = section_count

    def usage_payload(self) -> dict:
        return self.usage_tracker.snapshot()
