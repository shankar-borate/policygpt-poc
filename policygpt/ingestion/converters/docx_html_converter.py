"""DocxToHtmlConverter — converts DOCX/DOC files to clean, structured HTML.

Improvements over flat extraction
-----------------------------------
- Run-level formatting: bold → <strong>, italic → <em>.
- Heading 1–4 → <h2>–<h5> (document <h1> is the title).
- Consecutive list items wrapped in <ul> or <ol>.
- Tables use <thead>/<tbody> and scope="col" on header cells.
- Empty table cells shown as — (em-dash) rather than blank.
- Document core-properties (author, title) added as a metadata block.

Caching
-------
Converted HTML is saved to {output_dir}/{stem}.html and reused on subsequent
runs as long as the HTML file is newer than the source file.
"""

from __future__ import annotations

import html as _html_lib
import logging
from pathlib import Path

from policygpt.ingestion.converters.base import HtmlConverter

logger = logging.getLogger(__name__)


# ── Module-level helpers (static, no class dependency) ────────────────────────

def _runs_to_html(para) -> str:
    """Build an HTML inline string from a paragraph's runs, preserving bold/italic."""
    parts: list[str] = []
    for run in para.runs:
        if not run.text:
            continue
        text = _html_lib.escape(run.text)
        if run.bold and run.italic:
            text = f"<strong><em>{text}</em></strong>"
        elif run.bold:
            text = f"<strong>{text}</strong>"
        elif run.italic:
            text = f"<em>{text}</em>"
        parts.append(text)
    return "".join(parts)


def _para_to_html(para) -> str:
    """Convert a single paragraph to an HTML element string."""
    if not para.text.strip():
        return ""

    style_name = (para.style.name or "").lower()
    content = _runs_to_html(para) or _html_lib.escape(para.text.strip())

    if "heading 1" in style_name:
        return f"<h2>{content}</h2>"
    if "heading 2" in style_name:
        return f"<h3>{content}</h3>"
    if "heading 3" in style_name:
        return f"<h4>{content}</h4>"
    if "heading 4" in style_name or "heading 5" in style_name:
        return f"<h5>{content}</h5>"
    if "list number" in style_name:
        return f'<li data-list="ol">{content}</li>'
    if "list" in style_name or "bullet" in style_name:
        return f"<li>{content}</li>"
    return f"<p>{content}</p>"


def _table_to_html(table) -> str:
    """Convert a python-docx Table to HTML with <thead>/<tbody>."""
    rows = list(table.rows)
    if not rows:
        return ""

    header_cells = "".join(
        f'<th scope="col">{_html_lib.escape(cell.text.strip()) or "&#8212;"}</th>'
        for cell in rows[0].cells
    )
    thead = f"<thead><tr>{header_cells}</tr></thead>"

    body_rows: list[str] = []
    for row in rows[1:]:
        cells = "".join(
            f"<td>{_html_lib.escape(cell.text.strip()) or '&#8212;'}</td>"
            for cell in row.cells
        )
        body_rows.append(f"<tr>{cells}</tr>")
    tbody = ("<tbody>\n" + "\n".join(body_rows) + "\n</tbody>") if body_rows else ""

    return f"<table>\n{thead}\n{tbody}\n</table>"


def _wrap_lists(parts: list[str]) -> list[str]:
    """Wrap consecutive <li> fragments in <ul> or <ol> blocks."""
    result: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i].startswith('<li data-list="ol">'):
            group: list[str] = []
            while i < len(parts) and parts[i].startswith('<li data-list="ol">'):
                group.append(parts[i].replace(' data-list="ol"', ""))
                i += 1
            result.append("<ol>\n" + "\n".join(group) + "\n</ol>")
        elif parts[i].startswith("<li>"):
            group = []
            while i < len(parts) and parts[i].startswith("<li>"):
                group.append(parts[i])
                i += 1
            result.append("<ul>\n" + "\n".join(group) + "\n</ul>")
        else:
            result.append(parts[i])
            i += 1
    return result


# ── Converter class ────────────────────────────────────────────────────────────

class DocxToHtmlConverter(HtmlConverter):
    """Converts DOCX/DOC files to structured HTML."""

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({"docx", "doc"})

    def _convert_to_html(self, path: Path) -> str:
        try:
            import docx
        except ImportError as exc:
            raise ImportError(
                "DOCX-to-HTML conversion requires 'python-docx'. "
                "Install it with: pip install python-docx"
            ) from exc

        title = self._title_from_stem(path.stem)
        document = docx.Document(str(path))
        parts: list[str] = []

        # Iterate document body elements in document order.
        for block in document.element.body:
            tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

            if tag == "p":
                from docx.text.paragraph import Paragraph as _Paragraph
                para = _Paragraph(block, document)
                html = _para_to_html(para)
                if html:
                    parts.append(html)

            elif tag == "tbl":
                from docx.table import Table as _Table
                table = _Table(block, document)
                parts.append(_table_to_html(table))

        # Wrap consecutive list items.
        parts = _wrap_lists(parts)

        # Build metadata block from core properties.
        props = document.core_properties
        meta_items: list[str] = [
            f"<dt>Source</dt><dd>{_html_lib.escape(path.name)}</dd>",
            f"<dt>Type</dt><dd>Word Document</dd>",
        ]
        if props.author:
            meta_items.append(f"<dt>Author</dt><dd>{_html_lib.escape(props.author)}</dd>")
        if props.modified:
            try:
                meta_items.append(
                    f"<dt>Modified</dt><dd>{props.modified.strftime('%d-%b-%Y')}</dd>"
                )
            except Exception:
                pass
        meta = '<dl class="doc-meta">\n' + "\n".join(meta_items) + "\n</dl>"

        body = meta + "\n" + "\n".join(parts)
        return self._wrap_html(title, body)
