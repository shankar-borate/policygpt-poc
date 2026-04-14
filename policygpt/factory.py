import os

from policygpt.core.bot import PolicyGPTBot
from policygpt.config import Config
from policygpt.core.corpus import ProgressCallback
from policygpt.ingestion import IngestionPipeline
from policygpt.ingestion.readers import FolderReader
from policygpt.observability.usage_metrics import LLMUsageTracker


def _build_thread_repo(config: Config):
    """Return a ThreadRepository when OpenSearch is configured, else None."""
    if not config.hybrid_search_enabled:
        return None
    try:
        from policygpt.storage.threads import ThreadRepository
        repo = ThreadRepository(config)
        repo.ensure_index()
        return repo
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("ThreadRepository init failed: %s", exc)
        return None


def create_bot(
    config: Config | None = None,
    usage_tracker: LLMUsageTracker | None = None,
) -> PolicyGPTBot:
    """Create a PolicyGPTBot without running ingestion.

    Ingestion is a separate background concern — call run_ingestion() or
    use ServerRuntime which handles it automatically.
    """
    resolved_config = config or Config.from_env()
    if resolved_config.ai_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")
    thread_repo = _build_thread_repo(resolved_config)
    return PolicyGPTBot(
        config=resolved_config,
        usage_tracker=usage_tracker,
        thread_repo=thread_repo,
    )


def create_ready_bot(
    folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
    config: Config | None = None,
    usage_tracker: LLMUsageTracker | None = None,
) -> PolicyGPTBot:
    """Create a bot AND run ingestion synchronously.

    Kept for tests / CLI usage.  The production server (ServerRuntime) uses
    create_bot() + a background ingestion thread instead.
    """
    resolved_config = config or Config.from_env()
    resolved_folder = folder or resolved_config.document_folder
    user_ids = list(resolved_config.ingestion_user_ids)
    domain = resolved_config.domain_type

    bot = create_bot(config=resolved_config, usage_tracker=usage_tracker)

    reader = FolderReader(folder_path=resolved_folder, user_ids=user_ids, domain=domain)
    pipeline = IngestionPipeline.from_corpus(
        corpus=bot.corpus,
        reader=reader,
        default_user_ids=user_ids,
        default_domain=domain,
    )
    pipeline.run(progress_callback=progress_callback)
    return bot
