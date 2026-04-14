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

from policygpt.ingestion.readers.base import IngestMessage, Reader

logger = logging.getLogger(__name__)

# Maps lowercase suffix → content_type string understood by ExtractorRegistry.
_EXTENSION_MAP: dict[str, str] = {
    ".html": "html",
    ".htm":  "html",
    ".txt":  "text",
    ".pdf":  "pdf",
    ".ppt":  "ppt",
    ".pptx": "ppt",
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".tiff": "image",
    ".tif":  "image",
    ".bmp":  "image",
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
            content_type = _EXTENSION_MAP[path.suffix.lower()]
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
    def _read_file(path: Path, content_type: str) -> bytes | str:
        """Read file bytes for binary types, decoded text for text types."""
        if content_type in {"html", "text"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        return path.read_bytes()
