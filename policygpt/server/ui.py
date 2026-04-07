from pathlib import Path


class WebUIRenderer:
    def __init__(self, web_dir: Path) -> None:
        self.web_dir = web_dir

    def _asset_url(self, asset_name: str) -> str:
        asset_path = (self.web_dir / asset_name).resolve()
        version = int(asset_path.stat().st_mtime) if asset_path.is_file() else 0
        return f"/static/{asset_name}?v={version}"

    def render_index(self) -> str:
        html = (self.web_dir / "index.html").read_text(encoding="utf-8")
        html = html.replace('/static/styles.css', self._asset_url("styles.css"))
        html = html.replace('/static/usage-widget.css', self._asset_url("usage-widget.css"))
        html = html.replace('/static/app.js', self._asset_url("app.js"))
        html = html.replace('/static/usage-widget.js', self._asset_url("usage-widget.js"))
        return html
