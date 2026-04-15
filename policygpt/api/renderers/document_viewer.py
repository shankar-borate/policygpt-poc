from html import escape
from pathlib import Path


class DocumentViewerRenderer:
    def __init__(self, web_dir: Path | None = None) -> None:
        self.web_dir = web_dir or Path(__file__).resolve().parents[3] / "web"

    def _asset_url(self, asset_name: str) -> str:
        asset_path = (self.web_dir / asset_name).resolve()
        version = int(asset_path.stat().st_mtime) if asset_path.is_file() else 0
        return f"/static/{asset_name}?v={version}"

    def render(
        self,
        *,
        document_title: str,
        source_path: str,
        open_url: str,
        iframe_url: str,
        target_section_title: str,
    ) -> str:
        title = escape(document_title or Path(source_path).stem or "Document")
        file_name = escape(Path(source_path).name)
        safe_open_url = escape(open_url, quote=True)
        safe_iframe_url = escape(iframe_url, quote=True)
        safe_target_title = escape(target_section_title or "")

        matched_html = (
            f'<p class="document-viewer-matched">Matched section: <strong>{safe_target_title}</strong></p>'
            if safe_target_title
            else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Policy GPT Reference</title>
    <link rel="stylesheet" href="{self._asset_url('document-viewer.css')}">
</head>
<body class="document-viewer-body">
    <div class="document-viewer-shell">
        <header class="document-viewer-header">
            <div class="document-viewer-header-text">
                <p class="document-viewer-eyebrow">Reference Viewer</p>
                <h1>{title}</h1>
                <p class="document-viewer-subtitle">{file_name}</p>
            </div>
            <div class="document-viewer-header-actions">
                {matched_html}
                <a class="document-viewer-open-link" href="{safe_open_url}" target="_blank" rel="noreferrer noopener">Open in new tab</a>
            </div>
        </header>
        <div class="document-viewer-frame-wrap">
            <iframe
                class="document-viewer-iframe"
                src="{safe_iframe_url}"
                title="Original policy document: {escape(file_name)}"
                sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
            ></iframe>
        </div>
    </div>
    <script src="{self._asset_url('document-viewer.js')}"></script>
</body>
</html>
"""
