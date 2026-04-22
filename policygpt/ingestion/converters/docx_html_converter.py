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
from typing import TYPE_CHECKING

from policygpt.ingestion.converters.base import HtmlConverter

if TYPE_CHECKING:
    from policygpt.ingestion.converters.vision import VisionDescriber
    from policygpt.ingestion.extraction.ocr import OcrExtractor
    from policygpt.ingestion.explainers.factory import ExplainerFactory

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
    """Converts DOCX/DOC files to structured HTML.

    Parameters
    ----------
    output_dir:
        Directory where converted HTML files are written.
    skip_if_cached:
        Reuse existing HTML when newer than the source file.
    vision_describer:
        Optional LLM vision model for embedded images/diagrams.
    ocr:
        Optional OCR extractor fallback for embedded images.
    """

    def __init__(
        self,
        output_dir: str | Path,
        skip_if_cached: bool = True,
        vision_describer: "VisionDescriber | None" = None,
        ocr: "OcrExtractor | None" = None,
        explainer: "ExplainerFactory | None" = None,
    ) -> None:
        super().__init__(output_dir=output_dir, skip_if_cached=skip_if_cached)
        self._vision = vision_describer
        self._ocr = ocr
        self._explainer = explainer

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

        logger.info("DocxToHtmlConverter: %s", path.name)
        print(f"    [DOCX] {path.name}", flush=True)

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

        # Process embedded images via vision / OCR.
        image_parts = self._process_images(document, path.name)
        if image_parts:
            parts.extend(image_parts)

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

    def _process_images(self, document, filename: str) -> list[str]:
        """Describe all embedded images in the document via vision → OCR → skip."""
        parts: list[str] = []
        seen: set[str] = set()

        for rel in document.part.rels.values():
            if "image" not in rel.reltype:
                continue
            rId = rel.rId
            if rId in seen:
                continue
            seen.add(rId)

            try:
                image_part = rel.target_part
                image_bytes: bytes = image_part.blob
                mime_type: str = image_part.content_type or "image/png"
            except Exception as exc:
                logger.warning(
                    "DocxToHtmlConverter: %s — failed to read image %s: %s",
                    filename, rId, exc,
                )
                continue

            size_kb = len(image_bytes) / 1024
            logger.info(
                "DocxToHtmlConverter: %s image %s — %.1f KB (%s)",
                filename, rId, size_kb, mime_type,
            )
            print(
                f"    [DOCX] {filename} image {rId} — {size_kb:.1f} KB ({mime_type})",
                flush=True,
            )

            html_fragment = self._describe_image(image_bytes, mime_type, rId, filename)
            if html_fragment:
                parts.append(html_fragment)

        return parts

    def _describe_image(
        self, image_bytes: bytes, mime_type: str, rId: str, filename: str
    ) -> str:
        # 1. Vision model
        if self._vision is not None:
            logger.info(
                "DocxToHtmlConverter: %s image %s — vision [%s] …",
                filename, rId, self._vision.provider_name,
            )
            print(
                f"    [DOCX] {filename} image {rId} — vision [{self._vision.provider_name}] …",
                flush=True,
            )
            html_fragment = self._vision.describe_page(image_bytes, mime_type)
            if html_fragment.strip():
                logger.info(
                    "DocxToHtmlConverter: %s image %s — vision OK (%d chars)",
                    filename, rId, len(html_fragment),
                )
                print(
                    f"    [DOCX] {filename} image {rId} — vision OK ({len(html_fragment)} chars)",
                    flush=True,
                )
                return (
                    f'<div class="doc-image" '
                    f'data-source="vision:{self._vision.provider_name}">\n'
                    f"{html_fragment}\n"
                    f"</div>"
                )
            logger.warning(
                "DocxToHtmlConverter: %s image %s — vision empty, trying OCR", filename, rId
            )
            print(
                f"    [DOCX] {filename} image {rId} — vision empty, trying OCR …", flush=True
            )

        # 2. OCR fallback
        if self._ocr is not None:
            logger.info(
                "DocxToHtmlConverter: %s image %s — OCR …", filename, rId
            )
            print(f"    [DOCX] {filename} image {rId} — OCR …", flush=True)
            ocr_text = self._ocr.extract_from_bytes(image_bytes, mime_type)
            if ocr_text.strip():
                lines = [
                    f"<p>{_html_lib.escape(line)}</p>"
                    for line in ocr_text.splitlines()
                    if line.strip()
                ]
                logger.info(
                    "DocxToHtmlConverter: %s image %s — OCR OK (%d chars)",
                    filename, rId, len(ocr_text),
                )
                print(
                    f"    [DOCX] {filename} image {rId} — OCR OK ({len(ocr_text)} chars)",
                    flush=True,
                )
                return (
                    f'<div class="doc-image" data-source="ocr">\n'
                    f'{chr(10).join(lines)}\n'
                    f"</div>"
                )

        logger.warning(
            "DocxToHtmlConverter: %s image %s — no vision/OCR output, skipped", filename, rId
        )
        print(
            f"    [DOCX] {filename} image {rId} — skipped (no vision/OCR output)", flush=True
        )
        return ""
