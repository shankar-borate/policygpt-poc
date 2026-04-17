"""HtmlConverter — abstract base class for all document-to-HTML converters.

Each converter handles one or more source content types (pdf, pptx, docx, …)
and produces a self-contained HTML string that can feed the rest of the
ingestion pipeline unchanged.

Caching
-------
Converted HTML is written to {output_dir}/{stem}.html and reused on
subsequent runs when the output file is newer than the source document.
Pass skip_if_cached=False to force re-conversion.
"""

from __future__ import annotations

import html as _html_lib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class HtmlConverter(ABC):
    """Abstract base class for format-specific document-to-HTML converters.

    Parameters
    ----------
    output_dir:
        Directory where converted HTML files are written.
        Typically {debug_log_dir}/html/
    skip_if_cached:
        When True (default), skip conversion if the HTML file already exists
        and is newer than the source document.
    """

    def __init__(
        self,
        output_dir: str | Path,
        skip_if_cached: bool = True,
    ) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._skip_if_cached = skip_if_cached

    # ── Contract ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def supported_content_types(self) -> frozenset[str]:
        """Content type strings this converter handles (e.g. {"pdf", "pptx"})."""

    @abstractmethod
    def _convert_to_html(self, path: Path) -> str:
        """Convert the document at *path* to an HTML string.

        Subclasses implement format-specific logic here.
        The returned string must be a complete HTML document.
        """

    # ── Public API ────────────────────────────────────────────────────────────

    def convert_all(self, source_path: str) -> list[tuple[str, str]]:
        """Convert *source_path* to one or more HTML files.

        Most formats produce a single file.  Override in subclasses where a
        single source document should become multiple independent documents
        (e.g. one HTML file per Excel worksheet).

        Returns a list of (html_path, html_content) tuples.
        """
        return [self.convert(source_path)]

    def convert(self, source_path: str) -> tuple[str, str]:
        """Convert the document at *source_path* to HTML.

        Returns
        -------
        (html_path, html_content)
            html_path    — path to the saved HTML file.
            html_content — the HTML string.
            On any error both fall back to an empty-body HTML so the pipeline
            can continue without crashing.
        """
        src = Path(source_path)
        out_path = self._out / (src.stem + ".html")
        converter_name = type(self).__name__

        # Cache hit — reuse existing HTML when it is newer than the source.
        if self._skip_if_cached and out_path.exists():
            if out_path.stat().st_mtime >= src.stat().st_mtime:
                print(f"  [Convert] {src.name} — cached, skipping", flush=True)
                logger.debug("%s: cache hit %s", converter_name, src.name)
                return str(out_path), out_path.read_text(encoding="utf-8", errors="ignore")

        print(f"  [Convert] {src.name} — converting with {converter_name} …", flush=True)
        t0 = time.perf_counter()
        try:
            html_content = self._convert_to_html(src)
            elapsed = time.perf_counter() - t0
            out_path.write_text(html_content, encoding="utf-8")
            print(f"  [Convert] {src.name} — done in {elapsed:.1f}s", flush=True)
            logger.info("%s: saved %s (%.1fs)", converter_name, out_path, elapsed)
            return str(out_path), html_content
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [Convert] {src.name} — FAILED after {elapsed:.1f}s: {exc}", flush=True)
            logger.warning(
                "%s: failed for %s — %s", converter_name, src.name, exc, exc_info=True
            )
            fallback = (
                f"<html><body>"
                f"<p>Conversion failed: {_html_lib.escape(str(exc))}</p>"
                f"</body></html>"
            )
            return source_path, fallback

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _title_from_stem(stem: str) -> str:
        """Turn a file stem into a readable title."""
        import re
        stem = re.sub(r"[_\-]v[\d.]+$", "", stem, flags=re.I)
        return re.sub(r"[_\-]+", " ", stem).strip()

    @staticmethod
    def _wrap_html(title: str, body: str) -> str:
        """Wrap *body* in a minimal HTML document skeleton."""
        escaped_title = _html_lib.escape(title)
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            '  <meta charset="utf-8">\n'
            f"  <title>{escaped_title}</title>\n"
            "</head>\n"
            "<body>\n"
            f"<h1>{escaped_title}</h1>\n"
            f"{body}\n"
            "</body>\n"
            "</html>"
        )
