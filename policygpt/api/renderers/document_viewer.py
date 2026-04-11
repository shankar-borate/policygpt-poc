import json
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
        sections: list[dict[str, object]],
        target_section_id: str,
        target_section_title: str,
        open_url: str,
    ) -> str:
        title = escape(document_title or Path(source_path).stem or "Document")
        file_name = escape(Path(source_path).name)
        safe_source_path = escape(source_path)
        safe_open_url = escape(open_url, quote=True)
        safe_target_title = escape(target_section_title or "Matched section")
        target_payload = json.dumps({"id": target_section_id})
        empty_text_html = '<p class="document-viewer-empty">No extracted text available.</p>'

        toc_items: list[str] = []
        section_items: list[str] = []
        for index, section in enumerate(sections):
            section_dom_id = escape(str(section["dom_id"]))
            section_title = escape(str(section["title"]))
            body_html = self._render_section_body(str(section["text"]))
            target_class = " document-viewer-section-target" if section["dom_id"] == target_section_id else ""
            toc_items.append(
                (
                    f'<a class="document-viewer-toc-link" href="#{section_dom_id}" '
                    f'data-section-link-id="{section_dom_id}">'
                    f"<span>{index + 1}.</span> {section_title}"
                    "</a>"
                )
            )
            section_items.append(
                (
                    f'<section id="{section_dom_id}" class="document-viewer-section{target_class}" tabindex="-1">'
                    f'<p class="document-viewer-section-kicker">Section {index + 1}</p>'
                    f"<h2>{section_title}</h2>"
                    f"{body_html or empty_text_html}"
                    "</section>"
                )
            )

        if not toc_items:
            toc_items.append('<p class="document-viewer-empty">No indexed sections available.</p>')
        if not section_items:
            section_items.append('<p class="document-viewer-empty">No extracted text was available for this file.</p>')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Policy GPT Reference</title>
    <link rel="stylesheet" href="{self._asset_url('document-viewer.css')}">
</head>
<body class="document-viewer-body" data-target='{escape(target_payload, quote=True)}'>
    <div class="document-viewer-shell">
        <header class="document-viewer-header">
            <div>
                <p class="document-viewer-eyebrow">Reference Viewer</p>
                <h1>{title}</h1>
                <p class="document-viewer-subtitle">{file_name}</p>
            </div>
            <a class="document-viewer-open-link" href="{safe_open_url}" target="_blank" rel="noreferrer noopener">Open original file</a>
        </header>
        <section class="document-viewer-banner">
            <strong>Jump target:</strong> {safe_target_title}
            <span>Showing extracted, indexed article sections so the reference can land on the matched spot reliably.</span>
        </section>
        <main class="document-viewer-content">
            <aside class="document-viewer-toc">
                <p class="document-viewer-toc-title">Sections</p>
                {''.join(toc_items)}
            </aside>
            <article class="document-viewer-article" aria-label="Document content">
                <div class="document-viewer-meta">
                    <span>Source path</span>
                    <code>{safe_source_path}</code>
                </div>
                {''.join(section_items)}
            </article>
        </main>
    </div>
    <script src="{self._asset_url('document-viewer.js')}"></script>
</body>
</html>
"""

    @staticmethod
    def _render_section_body(text: str) -> str:
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        html_parts: list[str] = []
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            if all(
                line.startswith(("- ", "* ")) or (line[:1].isdigit() and ". " in line[:4])
                for line in lines
            ):
                items = []
                for line in lines:
                    cleaned = line
                    if cleaned.startswith(("- ", "* ")):
                        cleaned = cleaned[2:].strip()
                    else:
                        cleaned = cleaned.split(". ", 1)[-1].strip()
                    items.append(f"<li>{escape(cleaned)}</li>")
                html_parts.append(f"<ul>{''.join(items)}</ul>")
                continue

            paragraph = "<br>".join(escape(line) for line in lines)
            html_parts.append(f"<p>{paragraph}</p>")
        return "".join(html_parts)
