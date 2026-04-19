"""FolderReader — reads documents from a local directory.

Supported file types
--------------------
html / htm   Full HTML extraction (implemented).
txt          Plain text extraction (implemented).
pdf          PDF extraction (placeholder — extractor raises NotImplementedError).
ppt / pptx   PowerPoint extraction (placeholder).
jpg/jpeg/png Image OCR extraction (placeholder).

Adding a new file type requires only:
  1. Add the extension → content_type entry to _EXTENSION_MAP.
  2. Implement the corresponding Extractor in ingestion/extractors/.
  No changes needed here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from policygpt.constants import ContentType, FileExtension
from policygpt.ingestion.readers.base import IngestMessage, Reader

logger = logging.getLogger(__name__)

# Maps file extension → logical content_type used by ExtractorRegistry and
# HtmlConverterRegistry.  Non-HTML formats that can be converted to HTML are
# listed with their own content_type; after conversion the type becomes "html".
_EXTENSION_MAP: dict[FileExtension, ContentType] = {
    FileExtension.HTML: ContentType.HTML,
    FileExtension.HTM:  ContentType.HTML,
    FileExtension.TXT:  ContentType.TEXT,
    FileExtension.PDF:  ContentType.PDF,
    FileExtension.PPTX: ContentType.PPTX,
    FileExtension.PPT:  ContentType.PPT,
    FileExtension.DOCX: ContentType.DOCX,
    FileExtension.DOC:  ContentType.DOC,
    FileExtension.XLSX: ContentType.XLSX,
    FileExtension.XLS:  ContentType.XLS,
    FileExtension.JPG:  ContentType.JPG,
    FileExtension.JPEG: ContentType.JPEG,
    FileExtension.PNG:  ContentType.PNG,
    FileExtension.GIF:  ContentType.GIF,
    FileExtension.BMP:  ContentType.BMP,
    FileExtension.TIFF: ContentType.TIFF,
    FileExtension.TIF:  ContentType.TIF,
    FileExtension.WEBP: ContentType.WEBP,
}

# File-name substrings that mark generated artefacts — skip them so the
# pipeline does not re-ingest its own debug output.
_EXCLUDED_SUFFIXES: tuple[str, ...] = ("_summary", "_faq", "_entities", "_section")


class FolderReader(Reader):
    """Scans a local folder and yields one IngestMessage per document file.

    Parameters
    ----------
    folder_path:
        Absolute or relative path to the directory to scan.
    domain:
        Domain name forwarded onto every IngestMessage (e.g. "policy").
    user_ids:
        Access-control list forwarded onto every IngestMessage.
    excluded_name_parts:
        Additional file-name substrings to skip (merged with built-in list).
    """

    def __init__(
        self,
        folder_path: str,
        domain: str,
        user_ids: list[str],
        excluded_name_parts: tuple[str, ...] = (),
    ) -> None:
        self.folder_path = Path(folder_path).resolve()
        self.domain = domain
        self.user_ids = user_ids
        self._excluded = _EXCLUDED_SUFFIXES + tuple(excluded_name_parts)

    # ── Public API ─────────────────────────────────────────────────────────────

    def read(self) -> Iterator[IngestMessage]:
        """Yield one IngestMessage per supported file found in folder_path."""
        files = self._scan()
        if not files:
            logger.warning("FolderReader: no supported files found in %s", self.folder_path)
            return

        for path in files:
            content_type = _EXTENSION_MAP[FileExtension(path.suffix.lower())]
            try:
                content = self._read_file(path, content_type)
            except Exception as exc:
                logger.warning("FolderReader: skipping %s — %s: %s", path.name, type(exc).__name__, exc)
                continue

            yield IngestMessage(
                content=content,
                content_type=content_type,
                file_name=path.name,
                source_path=str(path),
                domain=self.domain,
                user_ids=self.user_ids,
                metadata={"folder": str(self.folder_path)},
            )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _scan(self) -> list[Path]:
        """Return sorted list of supported, non-excluded files."""
        found: list[Path] = []
        for suffix in _EXTENSION_MAP:
            found.extend(self.folder_path.glob(f"*{suffix}"))
            found.extend(self.folder_path.glob(f"*{suffix.upper()}"))

        result: list[Path] = []
        for path in sorted(set(found)):
            name_lower = path.name.lower()
            if any(part in name_lower for part in self._excluded):
                continue
            result.append(path)

        logger.debug("FolderReader: found %d files in %s", len(result), self.folder_path)
        return result

    @staticmethod
    def _read_file(path: Path, content_type: ContentType) -> bytes | str:
        """Read file bytes for binary types, decoded text for text types."""
        if content_type in {ContentType.HTML, ContentType.TEXT}:
            return path.read_text(encoding="utf-8", errors="ignore")
        # All other formats (pdf, pptx, docx, xlsx, images …) are binary.
        # The converter pipeline reads from source_path directly, so the
        # content bytes are passed through but not deeply parsed here.
        return path.read_bytes()
