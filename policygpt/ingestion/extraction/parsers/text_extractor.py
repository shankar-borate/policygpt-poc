"""PlainTextExtractor — extracts sections from plain-text files."""

from __future__ import annotations

import re
from pathlib import Path

from policygpt.constants import FileExtension
from policygpt.ingestion.extraction.parsers.base import BaseExtractor
from policygpt.models.extraction import ExtractedSection


class PlainTextExtractor(BaseExtractor):
    """Extracts document title and sections from ``.txt`` files."""

    SUPPORTED_EXTENSIONS: frozenset[FileExtension] = frozenset({FileExtension.TXT})

    def extract(self, path: str) -> tuple[str, list[ExtractedSection]]:
        return self.extract_from_plain_text(path)

    def extract_from_plain_text(self, path: str) -> tuple[str, list[ExtractedSection]]:
        text = self.clean_whitespace(self.read_text_file(path))
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        doc_title = Path(path).stem

        if lines and len(lines[0]) < 150:
            doc_title = lines[0]

        units: list[tuple[str, str]] = []
        buffer: list[str] = []

        def flush_paragraph() -> None:
            nonlocal buffer
            if buffer:
                paragraph = " ".join(buffer).strip()
                if paragraph:
                    units.append(("p", paragraph))
                buffer = []

        for line in lines:
            is_heading = self._looks_like_heading_line(line)
            is_bullet = re.match(r"^[-*\u2022]\s+.+", line) or re.match(r"^\d+[\.\)]\s+.+", line)

            if is_heading:
                flush_paragraph()
                units.append(("h2", line))
            elif is_bullet:
                flush_paragraph()
                units.append(("li", line))
            else:
                buffer.append(line)

        flush_paragraph()
        return doc_title, self._group_units_into_sections(units)
