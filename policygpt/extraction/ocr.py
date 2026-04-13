"""AWS Textract OCR helper for extracting text from images embedded in documents.

Only imported when ocr_enabled=True in Config, so Textract / boto3 are not
required for deployments that do not need OCR.

Supports:
  - Local image files (JPEG, PNG, TIFF, BMP)
  - Base64-encoded data URIs (e.g. <img src="data:image/png;base64,...">)
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
_DATA_URI_PATTERN = re.compile(
    r"^data:image/(?P<mime>[a-zA-Z0-9+/.-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)


class TextractOCR:
    """Thin wrapper around AWS Textract detect_document_text.

    Textract returns individual word/line blocks with confidence scores.
    We collect LINE blocks above min_confidence and join them with newlines,
    preserving the natural reading order Textract infers from the layout.
    """

    def __init__(self, region: str, min_confidence: float = 80.0) -> None:
        self.region = region
        self.min_confidence = min_confidence
        self._client = None  # lazy — only created on first use

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("textract", region_name=self.region)
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_from_path(self, image_path: str) -> str:
        """Return OCR text for a local image file, or '' on any error."""
        path = Path(image_path)
        if not path.is_file():
            return ""
        if path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            return ""
        try:
            image_bytes = path.read_bytes()
            return self._call_textract(image_bytes)
        except Exception:
            return ""

    def extract_from_data_uri(self, data_uri: str) -> str:
        """Return OCR text for a base64 data URI, or '' on any error."""
        match = _DATA_URI_PATTERN.match((data_uri or "").strip())
        if not match:
            return ""
        try:
            image_bytes = base64.b64decode(match.group("data"))
            return self._call_textract(image_bytes)
        except Exception:
            return ""

    def extract_from_src(self, src: str, html_file_dir: Path) -> str:
        """Resolve an <img src> and return OCR text.

        Handles:
          - data: URIs (inline base64)
          - Absolute file paths
          - Relative paths resolved against the HTML file's directory
        """
        src = (src or "").strip()
        if not src:
            return ""

        if src.startswith("data:"):
            return self.extract_from_data_uri(src)

        # Treat as file path — resolve relative to HTML dir
        candidate = Path(src)
        if not candidate.is_absolute():
            candidate = html_file_dir / candidate
        return self.extract_from_path(str(candidate.resolve()))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _call_textract(self, image_bytes: bytes) -> str:
        response = self.client.detect_document_text(
            Document={"Bytes": image_bytes}
        )
        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE"
            and float(block.get("Confidence", 0)) >= self.min_confidence
        ]
        return "\n".join(lines).strip()
