from pathlib import Path

from policygpt.config import Config
from policygpt.server.api import PolicyApiServer


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
CONFIG = Config.from_env()
SERVER = PolicyApiServer(config=CONFIG, web_dir=WEB_DIR)
app = SERVER.build_app()


if __name__ == "__main__":
    import uvicorn

    print(
        "[Policy GPT] Starting server on http://127.0.0.1:8010 (plain HTTP, not HTTPS).",
        flush=True,
    )
    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=False)
