import os

from policygpt.bot import PolicyGPTBot
from policygpt.config import Config
from policygpt.corpus import ProgressCallback


def create_ready_bot(
    folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
    config: Config | None = None,
) -> PolicyGPTBot:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

    resolved_config = config or Config.from_env()
    resolved_folder = folder or resolved_config.document_folder
    bot = PolicyGPTBot(config=resolved_config)
    bot.ingest_folder(resolved_folder, progress_callback=progress_callback)
    return bot
