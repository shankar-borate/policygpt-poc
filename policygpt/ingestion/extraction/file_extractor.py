"""FileExtractor — thin dispatcher that routes files to the correct strategy.

Concrete extraction logic lives in the ``extractors`` sub-package:

* :class:`~policygpt.ingestion.extraction.parsers.HtmlExtractor`  — ``.html`` / ``.htm``
* :class:`~policygpt.ingestion.extraction.parsers.PdfExtractor`   — ``.pdf``
* :class:`~policygpt.ingestion.extraction.parsers.PlainTextExtractor` — ``.txt``

``FileExtractor`` keeps the three ``extract_from_*`` convenience methods so
existing call-sites that used the old monolithic class continue to work.
"""

from __future__ import annotations

from pathlib import Path

from policygpt.config import Config
from policygpt.ingestion.extraction.parsers.base import BaseExtractor
from policygpt.ingestion.extraction.parsers.html_extractor import HtmlExtractor
from policygpt.ingestion.extraction.parsers.pdf_extractor import PdfExtractor
from policygpt.ingestion.extraction.parsers.text_extractor import PlainTextExtractor
from policygpt.models.extraction import ExtractedSection


class FileExtractor:
    """Routes extraction to the appropriate strategy based on file extension."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._html = HtmlExtractor(config)
        self._pdf = PdfExtractor(config)
        self._text = PlainTextExtractor(config)
        self._strategies = [self._html, self._pdf, self._text]

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    def extract(self, path: str) -> tuple[str, list[ExtractedSection]]:
        suffix = Path(path).suffix.lower()
        for strategy in self._strategies:
            if suffix in strategy.SUPPORTED_EXTENSIONS:
                return strategy.extract(path)
        return Path(path).stem, []

    # ── Per-type convenience methods (backward compatibility) ──────────────────

    def extract_from_html(self, path: str) -> tuple[str, list[ExtractedSection]]:
        return self._html.extract_from_html(path)

    def extract_from_pdf(self, path: str) -> tuple[str, list[ExtractedSection]]:
        return self._pdf.extract_from_pdf(path)

    def extract_from_plain_text(self, path: str) -> tuple[str, list[ExtractedSection]]:
        return self._text.extract_from_plain_text(path)

    # ── Static helpers (backward compatibility) ────────────────────────────────

    @staticmethod
    def read_text_file(path: str) -> str:
        return BaseExtractor.read_text_file(path)

    @staticmethod
    def clean_whitespace(text: str) -> str:
        return BaseExtractor.clean_whitespace(text)
