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
Image-only pages are handled by a configurable VisionDescriber (LLM-based) or
an OcrExtractor fallback; skipped when neither is provided.
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
import io
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from policygpt.ingestion.converters.base import HtmlConverter

if TYPE_CHECKING:
    from policygpt.ingestion.converters.vision import VisionDescriber
    from policygpt.ingestion.extraction.ocr import OcrExtractor

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
    """Converts PDF files to clean, table-preserving HTML.

    Text-based pages are converted using pdfplumber's text/table extraction.
    Image-based pages (no extractable text) are handled in priority order:
      1. VisionDescriber (LLM) — detailed semantic HTML description
      2. OcrExtractor (OCR)    — plain text extracted and wrapped in HTML
      3. Skipped               — empty page div omitted from output

    Parameters
    ----------
    output_dir:
        Directory where converted HTML files are written.
    skip_if_cached:
        Reuse existing HTML when newer than the source document.
    vision_describer:
        Optional LLM-based vision model for image-only pages.
        When provided, image pages are sent to the model for a rich HTML
        description before falling back to OCR.
    ocr:
        Optional OCR extractor used as a fallback when vision is unavailable
        or returns empty.  Must implement ``extract_from_bytes()``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        skip_if_cached: bool = True,
        vision_describer: "VisionDescriber | None" = None,
        ocr: "OcrExtractor | None" = None,
    ) -> None:
        super().__init__(output_dir=output_dir, skip_if_cached=skip_if_cached)
        self._vision = vision_describer
        self._ocr = ocr

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
        stats = {"text": 0, "vision": 0, "ocr": 0, "skipped": 0}

        with _suppress_pdfminer_warnings(), pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            logger.info("PdfToHtmlConverter: opened %s — %d page(s)", path.name, page_count)
            print(f"    [PDF] {path.name} — {page_count} page(s)", flush=True)

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_html, page_type = self._convert_page(page, page_num)
                except Exception as exc:
                    logger.warning(
                        "PdfToHtmlConverter: skipping page %d/%d of %s — %s",
                        page_num, page_count, path.name, exc,
                    )
                    stats["skipped"] += 1
                    continue
                if page_html.strip():
                    page_parts.append(page_html)
                    stats[page_type] = stats.get(page_type, 0) + 1
                else:
                    stats["skipped"] += 1

        logger.info(
            "PdfToHtmlConverter: %s done — text=%d vision=%d ocr=%d skipped=%d",
            path.name, stats["text"], stats["vision"], stats["ocr"], stats["skipped"],
        )
        print(
            f"    [PDF] {path.name} — done: "
            f"{stats['text']} text, {stats['vision']} vision, "
            f"{stats['ocr']} ocr, {stats['skipped']} skipped",
            flush=True,
        )

        meta = (
            f'<dl class="doc-meta">'
            f"<dt>Source</dt><dd>{_html_lib.escape(path.name)}</dd>"
            f"<dt>Type</dt><dd>PDF</dd>"
            f"<dt>Pages</dt><dd>{page_count}</dd>"
            f"</dl>"
        )
        body = meta + "\n" + "\n".join(page_parts)
        return self._wrap_html(title, body)

    def _convert_page(self, page, page_num: int) -> tuple[str, str]:
        """Convert one pdfplumber Page to an (html_fragment, page_type) tuple.

        page_type is one of: "text" | "vision" | "ocr" | "skipped"
        Routes to text-based or image-based conversion depending on whether
        pdfplumber can extract any text from the page.
        """
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        has_text = bool(text.strip())
        tables = page.find_tables()
        has_tables = bool(tables)

        if has_text or has_tables:
            logger.info(
                "PdfToHtmlConverter: page %d — text-based (%d chars, %d table(s)), extracting with pdfplumber",
                page_num, len(text.strip()), len(tables),
            )
            print(
                f"    [PDF] page {page_num} — text ({len(text.strip())} chars, {len(tables)} table(s)) → pdfplumber",
                flush=True,
            )
            return self._convert_text_page(page, page_num, text, tables), "text"

        # No extractable text — treat as image-based page
        logger.info(
            "PdfToHtmlConverter: page %d — no extractable text, routing to image pipeline",
            page_num,
        )
        print(f"    [PDF] page {page_num} — image-based, using vision/OCR", flush=True)
        return self._convert_image_page(page, page_num)

    # ── Text-based page conversion ────────────────────────────────────────────

    def _convert_text_page(self, page, page_num: int, plain_text: str, tables: list) -> str:
        """Convert a text/table page to HTML (original logic)."""
        parts: list[str] = []

        if not tables:
            for line in plain_text.splitlines():
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

    # ── Image-based page conversion ───────────────────────────────────────────

    def _convert_image_page(self, page, page_num: int) -> tuple[str, str]:
        """Convert an image-only PDF page via vision model or OCR fallback.

        Returns (html_fragment, page_type) where page_type is
        "vision" | "ocr" | "skipped".
        """
        logger.info("PdfToHtmlConverter: page %d — rendering to image …", page_num)
        image_bytes, mime_type = self._render_page_as_image(page)

        if not image_bytes:
            logger.warning(
                "PdfToHtmlConverter: page %d — render failed (Pillow installed?), skipping",
                page_num,
            )
            print(f"    [PDF] page {page_num} — render failed, skipping", flush=True)
            return "", "skipped"

        size_kb = len(image_bytes) / 1024
        logger.info(
            "PdfToHtmlConverter: page %d — rendered %.1f KB (%s)", page_num, size_kb, mime_type
        )

        # 1. Vision model (LLM) — rich semantic HTML description
        if self._vision is not None:
            logger.info(
                "PdfToHtmlConverter: page %d — calling vision model [%s] …",
                page_num, self._vision.provider_name,
            )
            print(
                f"    [PDF] page {page_num} — vision [{self._vision.provider_name}] …",
                flush=True,
            )
            html_fragment = self._vision.describe_page(image_bytes, mime_type)
            if html_fragment.strip():
                logger.info(
                    "PdfToHtmlConverter: page %d — vision OK (%d chars)",
                    page_num, len(html_fragment),
                )
                print(
                    f"    [PDF] page {page_num} — vision OK ({len(html_fragment)} chars)",
                    flush=True,
                )
                return (
                    f'<div class="page" data-page="{page_num}" '
                    f'data-source="vision:{self._vision.provider_name}">\n'
                    f"{html_fragment}\n"
                    f"</div>",
                    "vision",
                )
            logger.warning(
                "PdfToHtmlConverter: page %d — vision returned empty, falling back to OCR",
                page_num,
            )
            print(f"    [PDF] page {page_num} — vision empty, trying OCR …", flush=True)

        # 2. OCR fallback — plain text wrapped in HTML
        if self._ocr is not None:
            logger.info("PdfToHtmlConverter: page %d — calling OCR …", page_num)
            print(f"    [PDF] page {page_num} — OCR …", flush=True)
            ocr_text = self._ocr.extract_from_bytes(image_bytes, mime_type)
            if ocr_text.strip():
                parts = [
                    self._line_to_html(line)
                    for line in ocr_text.splitlines()
                    if line.strip()
                ]
                inner = "\n".join(_wrap_bullet_lists(parts))
                logger.info(
                    "PdfToHtmlConverter: page %d — OCR OK (%d chars)", page_num, len(ocr_text)
                )
                print(
                    f"    [PDF] page {page_num} — OCR OK ({len(ocr_text)} chars)", flush=True
                )
                return (
                    f'<div class="page" data-page="{page_num}" data-source="ocr">\n'
                    f"{inner}\n"
                    f"</div>",
                    "ocr",
                )
            logger.warning(
                "PdfToHtmlConverter: page %d — OCR returned empty text", page_num
            )

        logger.warning(
            "PdfToHtmlConverter: page %d — no vision/OCR produced output, page skipped",
            page_num,
        )
        print(f"    [PDF] page {page_num} — skipped (no output from vision/OCR)", flush=True)
        return "", "skipped"

    # ── Page rendering ────────────────────────────────────────────────────────

    @staticmethod
    def _render_page_as_image(page) -> tuple[bytes, str]:
        """Render a pdfplumber Page to PNG bytes.

        Uses pdfplumber's ``to_image()`` which requires Pillow.
        Returns ``(b"", "")`` if rendering fails (e.g. Pillow not installed).
        """
        try:
            buf = io.BytesIO()
            page.to_image(resolution=150).original.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        except Exception as exc:
            logger.warning("PdfToHtmlConverter: page render failed — %s", exc)
            return b"", ""

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
