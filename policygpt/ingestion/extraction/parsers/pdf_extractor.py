"""PdfExtractor — extracts sections from PDF files using pypdf."""

from __future__ import annotations

from pathlib import Path

from policygpt.constants import FileExtension
from policygpt.ingestion.extraction.parsers.base import BaseExtractor
from policygpt.models.extraction import ExtractedSection


class PdfExtractor(BaseExtractor):
    """Extracts document title and sections from ``.pdf`` files."""

    SUPPORTED_EXTENSIONS: frozenset[FileExtension] = frozenset({FileExtension.PDF})

    def extract(self, path: str) -> tuple[str, list[ExtractedSection]]:
        return self.extract_from_pdf(path)

    def extract_from_pdf(self, path: str) -> tuple[str, list[ExtractedSection]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF ingestion requires the 'pypdf' package. "
                "Install dependencies from requirements.txt."
            ) from exc

        reader = PdfReader(path)
        metadata_title = ""
        if reader.metadata and reader.metadata.title:
            metadata_title = self.clean_whitespace(str(reader.metadata.title))

        page_texts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append(extracted)

        combined_text = self.clean_whitespace("\n\n".join(page_texts))
        if not combined_text:
            return Path(path).stem, []

        lines = [line.strip() for line in combined_text.split("\n") if line.strip()]
        units = self._build_text_units(lines)
        title = self._select_text_document_title(
            path=path,
            lines=lines,
            metadata_title=metadata_title,
        )
        return title, self._group_units_into_sections(units)
