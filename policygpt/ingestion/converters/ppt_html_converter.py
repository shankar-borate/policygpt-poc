"""PptToHtmlConverter — converts PPTX/PPT files to HTML for ingestion.

Each slide becomes a <section class="slide"> element.
Text frames (title, content, bullet lists) are emitted as headings/paragraphs.
Tables are preserved as <table>/<tr>/<th>/<td>.
Images are skipped at conversion time — the OCR pipeline handles them later.

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


class PptToHtmlConverter(HtmlConverter):
    """Converts PPTX/PPT files to HTML."""

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({"pptx", "ppt"})

    def _convert_to_html(self, path: Path) -> str:
        try:
            from pptx import Presentation
            from pptx.util import Pt
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError as exc:
            raise ImportError(
                "PPTX-to-HTML conversion requires 'python-pptx'. "
                "Install it with: pip install python-pptx"
            ) from exc

        title = self._title_from_stem(path.stem)
        prs = Presentation(str(path))
        slide_parts: list[str] = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_html = self._convert_slide(slide, slide_num)
            if slide_html.strip():
                slide_parts.append(slide_html)

        slide_count = len(prs.slides)
        meta = (
            f'<dl class="doc-meta">'
            f"<dt>Source</dt><dd>{_html_lib.escape(path.name)}</dd>"
            f"<dt>Type</dt><dd>PowerPoint</dd>"
            f"<dt>Slides</dt><dd>{slide_count}</dd>"
            f"</dl>"
        )
        body = meta + "\n" + "\n".join(slide_parts)
        return self._wrap_html(title, body)

    def _convert_slide(self, slide, slide_num: int) -> str:
        """Convert one slide to an HTML <section> fragment."""
        parts: list[str] = []

        # Sort shapes top-to-bottom, left-to-right for reading order.
        shapes = sorted(slide.shapes, key=lambda s: (s.top or 0, s.left or 0))

        for shape in shapes:
            if shape.has_table:
                parts.append(self._table_to_html(shape.table))
            elif shape.has_text_frame:
                text_html = self._text_frame_to_html(shape)
                if text_html:
                    parts.append(text_html)
            # Images are intentionally skipped — the OCR pipeline handles them.

        if not parts:
            return ""
        inner = "\n".join(parts)
        return f'<section class="slide" data-slide="{slide_num}">\n{inner}\n</section>'

    @staticmethod
    def _text_frame_to_html(shape) -> str:
        """Convert a shape's text frame to HTML paragraphs/headings."""
        frame = shape.text_frame
        parts: list[str] = []

        for para_idx, para in enumerate(frame.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            escaped = _html_lib.escape(text)

            # Heuristic: first paragraph in a title/subtitle placeholder → h2;
            # bold run or large font → h3; everything else → p or li.
            is_title_ph = shape.shape_type in (13, 14, 15)  # TITLE, CENTER_TITLE, SUBTITLE
            if para_idx == 0 and is_title_ph:
                parts.append(f"<h2>{escaped}</h2>")
            elif para.runs and any(
                r.font.bold for r in para.runs if r.font and r.font.bold is not None
            ):
                parts.append(f"<h3>{escaped}</h3>")
            elif para.level and para.level > 0:
                indent = "  " * para.level
                parts.append(f"{indent}<li>{escaped}</li>")
            else:
                parts.append(f"<p>{escaped}</p>")

        return "\n".join(parts)

    @staticmethod
    def _table_to_html(table) -> str:
        """Convert a python-pptx Table to an HTML table with thead/tbody."""
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
