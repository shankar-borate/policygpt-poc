from pathlib import Path

from policygpt.config import Config
from policygpt.api.routes.chat import PolicyApiServer


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Load opensearch.env if present — keeps credentials out of source control.
# Variables already set in the environment take precedence (12-factor style).
_opensearch_env = BASE_DIR / "policygpt" / "config" / "opensearch.env"
if _opensearch_env.exists():
    import os
    for _line in _opensearch_env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

CONFIG = Config.from_env()
SERVER = PolicyApiServer(config=CONFIG, web_dir=WEB_DIR)
app = SERVER.build_app()


if __name__ == "__main__":
    import uvicorn

    print(
        "[Policy GPT] Starting server on http://127.0.0.1:8010 (plain HTTP, not HTTPS).",
        flush=True,
    )
    uvicorn.run("app:app", host="0.0.0.0", port=8012, reload=False)
