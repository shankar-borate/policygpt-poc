import os

from policygpt.core.bot import PolicyGPTBot
from policygpt.config import Config
from policygpt.core.corpus import ProgressCallback
from policygpt.observability.usage_metrics import LLMUsageTracker


def create_ready_bot(
    folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
    config: Config | None = None,
    usage_tracker: LLMUsageTracker | None = None,
) -> PolicyGPTBot:
    resolved_config = config or Config.from_env()
    if resolved_config.ai_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")
    resolved_folder = folder or resolved_config.document_folder
    bot = PolicyGPTBot(config=resolved_config, usage_tracker=usage_tracker)
    bot.ingest_folder(resolved_folder, progress_callback=progress_callback)
    return bot
