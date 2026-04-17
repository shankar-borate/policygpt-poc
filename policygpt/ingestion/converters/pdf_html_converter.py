"""PdfToHtmlConverter — converts PDF documents to clean HTML for ingestion.

Why HTML instead of raw text
-----------------------------
Contest PDFs (reward slabs, eligibility tables) embed critical data in tables.
Converting to HTML lets the existing table-aware HtmlExtractor parse rows and
columns correctly — dates, thresholds, and reward amounts stay attached to
their column headers rather than being merged into flat text.

Output format
-------------
One <div class="page"> per PDF page.
Tables are emitted as <table> / <tr> / <th> / <td>.
Text lines are emitted as <h2> (detected headings) or <p>.
No CSS styling is added — the PolicyRewriter or browser default handles that.

Caching
-------
The converted HTML is saved to {output_dir}/{stem}.html and reused on
subsequent runs as long as the HTML file is newer than the source PDF.
Pass skip_if_cached=False to force re-conversion.
"""

from __future__ import annotations

import contextlib
import html as _html_lib
import logging
import re
from pathlib import Path

from policygpt.ingestion.converters.base import HtmlConverter

logger = logging.getLogger(__name__)

# pdfminer emits warnings like "Cannot set gray non-stroke color because
# /'P762' is an invalid float value" for PDFs that use spot/custom color
# spaces.  These are rendering artefacts — text and table extraction is
# unaffected.  Silence them so they don't pollute ingestion logs.
logging.getLogger("pdfminer").setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_pdfminer_warnings():
    """Temporarily raise pdfminer log level to ERROR inside a with-block."""
    pdfminer_logger = logging.getLogger("pdfminer")
    original_level = pdfminer_logger.level
    pdfminer_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        pdfminer_logger.setLevel(original_level)


def _wrap_bullet_lists(parts: list[str]) -> list[str]:
    """Wrap consecutive <li> fragments in a <ul> block."""
    result: list[str] = []
    i = 0
    while i < len(parts):
        if parts[i].startswith("<li>"):
            group: list[str] = []
            while i < len(parts) and parts[i].startswith("<li>"):
                group.append(parts[i])
                i += 1
            result.append("<ul>\n" + "\n".join(group) + "\n</ul>")
        else:
            result.append(parts[i])
            i += 1
    return result


class PdfToHtmlConverter(HtmlConverter):
    """Converts PDF files to clean, table-preserving HTML."""

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({"pdf"})

    # ── Core conversion ───────────────────────────────────────────────────────

    def _convert_to_html(self, path: Path) -> str:
        try:
            import pdfplumber
        except ImportError as exc:
            raise ImportError(
                "PDF-to-HTML conversion requires 'pdfplumber'. "
                "Install it with: pip install pdfplumber"
            ) from exc

        title = self._title_from_stem(path.stem)
        page_parts: list[str] = []

        with _suppress_pdfminer_warnings(), pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_html = self._convert_page(page, page_num)
                except Exception as exc:
                    logger.warning(
                        "PdfToHtmlConverter: skipping page %d of %s — %s",
                        page_num, path.name, exc,
                    )
                    continue
                if page_html.strip():
                    page_parts.append(page_html)

        meta = (
            f'<dl class="doc-meta">'
            f"<dt>Source</dt><dd>{_html_lib.escape(path.name)}</dd>"
            f"<dt>Type</dt><dd>PDF</dd>"
            f"<dt>Pages</dt><dd>{page_count}</dd>"
            f"</dl>"
        )
        body = meta + "\n" + "\n".join(page_parts)
        return self._wrap_html(title, body)

    def _convert_page(self, page, page_num: int) -> str:
        """Convert one pdfplumber Page to an HTML fragment."""
        parts: list[str] = []

        # Find all tables on this page, sorted top-to-bottom.
        tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])

        if not tables:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            for line in text.splitlines():
                line = line.strip()
                if line:
                    parts.append(self._line_to_html(line))
        else:
            prev_bottom = 0.0
            for table in tables:
                x0, y0, x1, y1 = table.bbox

                if y0 > prev_bottom:
                    crop = page.crop((0, prev_bottom, page.width, y0))
                    text = crop.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            parts.append(self._line_to_html(line))

                data = table.extract()
                table_html = self._table_data_to_html(data)
                if table_html:
                    parts.append(table_html)

                prev_bottom = y1

            if prev_bottom < page.height:
                crop = page.crop((0, prev_bottom, page.width, page.height))
                text = crop.extract_text(x_tolerance=3, y_tolerance=3) or ""
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        parts.append(self._line_to_html(line))

        if not parts:
            return ""
        inner = "\n".join(_wrap_bullet_lists(parts))
        return f'<div class="page" data-page="{page_num}">\n{inner}\n</div>'

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _line_to_html(text: str) -> str:
        escaped = _html_lib.escape(text)

        # Numbered section headings: "1. Title" / "1.2.3 Title" / "Section 2:"
        if re.match(r"^\d+(\.\d+)*[\.\)]\s+[A-Z]", text) or re.match(
            r"^(Section|Chapter|Part|Appendix)\s+\d+", text, re.I
        ):
            return f"<h2>{escaped}</h2>"

        # Short ALL-CAPS lines (heading, not an acronym sentence)
        if re.match(r"^[A-Z][A-Z\s\d/&:,\-]{4,}$", text) and len(text) < 120:
            return f"<h2>{escaped}</h2>"

        # Title-case short line that doesn't end mid-sentence → sub-heading
        words = text.split()
        if (
            2 <= len(words) <= 10
            and not text.endswith((".", ",", ";"))
            and all(w[0].isupper() for w in words if w and w[0].isalpha())
        ):
            return f"<h3>{escaped}</h3>"

        # Label ending with colon (short) → sub-heading
        if text.endswith(":") and len(text) < 80:
            return f"<h3>{escaped}</h3>"

        # Bullet / dash list items
        m = re.match(r"^([•\u2022\u2023\u25e6\-\*])\s+(.*)", text, re.S)
        if m:
            return f"<li>{_html_lib.escape(m.group(2).strip())}</li>"

        return f"<p>{escaped}</p>"

    @staticmethod
    def _table_data_to_html(data: list[list] | None) -> str:
        """Convert pdfplumber table data to an HTML table with thead/tbody."""
        if not data:
            return ""

        # Drop rows where every cell is empty or None.
        rows = [row for row in data if any(cell for cell in row if cell)]
        if not rows:
            return ""

        # Header row → <thead>
        header_cells = "".join(
            f'<th scope="col">{_html_lib.escape(str(c or "").strip())}</th>'
            for c in rows[0]
        )
        thead = f"<thead><tr>{header_cells}</tr></thead>"

        # Body rows → <tbody>
        body_rows: list[str] = []
        for row in rows[1:]:
            cells = "".join(
                f"<td>{_html_lib.escape(str(c or '').strip()) or '&#8212;'}</td>"
                for c in row
            )
            body_rows.append(f"<tr>{cells}</tr>")
        tbody = ("<tbody>\n" + "\n".join(body_rows) + "\n</tbody>") if body_rows else ""

        return f"<table>\n{thead}\n{tbody}\n</table>"
