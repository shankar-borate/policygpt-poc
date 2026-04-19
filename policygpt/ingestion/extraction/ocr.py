"""OCR helpers for extracting text from images embedded in documents.

Three OCR backends are supported:
  - TextractOCR     — AWS Textract (``ocr_provider="textract"``)
  - ClaudeVisionOCR — Anthropic Claude vision (``ocr_provider="claude"``)

Use ``build_ocr_extractor(provider, region, confidence)`` to instantiate
the right one from config.  Returns None when OCR is disabled.

Image resolution
----------------
``ImageFetcher`` is the single entry point for turning any ``<img src>``
value into raw bytes + MIME type.  It handles:

  - Embedded data URIs  (``data:image/png;base64,...``)
  - Local file paths    (relative or absolute)
  - S3 URLs             (``s3://bucket/key`` or any ``*.amazonaws.com`` form)
  - All other HTTP/HTTPS URLs are skipped.

Use ``ImageFetcher().fetch(src, html_file_dir)`` or the module-level
convenience wrapper ``resolve_image_bytes_and_mime(src, html_file_dir)``.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"}

_MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".gif": "image/gif",
    ".bmp": "image/bmp",  ".webp": "image/webp",
    ".tiff": "image/tiff", ".tif": "image/tiff",
}

# Maximum raw image bytes we will process / store (1 MB).
MAX_IMAGE_BYTES = 1 * 1024 * 1024

_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>image/[a-zA-Z0-9+/.-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)

# S3 URL patterns:
#   s3://bucket/key
#   https://bucket.s3.amazonaws.com/key
#   https://s3.amazonaws.com/bucket/key
#   https://s3.region.amazonaws.com/bucket/key
#   https://bucket.s3.region.amazonaws.com/key
_S3_PROTOCOL_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")
_S3_VHOST_RE = re.compile(
    r"^https?://(?P<bucket>[a-z0-9][a-z0-9\-]{1,61}[a-z0-9])"
    r"\.s3(?:\.[a-z0-9-]+)?\.amazonaws\.com/(?P<key>.+)$"
)
_S3_PATH_RE = re.compile(
    r"^https?://s3(?:\.[a-z0-9-]+)?\.amazonaws\.com/(?P<bucket>[^/]+)/(?P<key>.+)$"
)


# ── ImageFetcher ──────────────────────────────────────────────────────────────

class ImageFetcher:
    """Resolves any ``<img src>`` value to raw image bytes + MIME type.

    Source types handled (tried in order):
      1. Embedded data URI  — decoded from base64 inline
      2. S3 URL             — downloaded via boto3 (lazy client)
      3. Local file path    — read from disk (relative paths resolved against
                              the directory of the HTML file)
      4. Other HTTP/HTTPS   — skipped (returns ``(None, "")``)

    Images larger than ``MAX_IMAGE_BYTES`` (1 MB) are skipped at every stage.

    Usage
    -----
    ``ImageFetcher`` is cheap to instantiate (boto3 client is lazy) and safe
    to share as a module-level singleton.  ``FileExtractor`` calls
    ``fetch(src, html_file_dir)`` for every ``<img>`` tag it encounters.
    """

    def __init__(self, s3_region: str | None = None, max_bytes: int = MAX_IMAGE_BYTES) -> None:
        self._s3_region = s3_region or os.environ.get("AWS_DEFAULT_REGION", "")
        self._max_bytes = max_bytes
        self._s3_client = None  # lazy

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(self, src: str, html_file_dir: Path) -> tuple[bytes | None, str]:
        """Resolve *src* to ``(image_bytes, mime_type)``.

        Parameters
        ----------
        src:
            The ``src`` (or ``data-src``) attribute value from an ``<img>`` tag.
        html_file_dir:
            Directory of the containing HTML file — used to resolve relative
            local paths.

        Returns
        -------
        ``(image_bytes, mime_type)`` where ``image_bytes`` is ``None`` when
        the image cannot be resolved or exceeds ``MAX_IMAGE_BYTES``.
        """
        src = (src or "").strip()
        if not src:
            return None, ""

        # 1. Embedded data URI
        if src.startswith("data:"):
            return self._fetch_data_uri(src)

        # 2. S3 URL
        bucket, key = self._parse_s3_url(src)
        if bucket is not None:
            return self._fetch_s3(bucket, key, src)

        # 3. Other external URL — skip
        if src.startswith("http://") or src.startswith("https://"):
            return None, ""

        # 4. Local file path
        return self._fetch_local(src, html_file_dir)

    # ── Internal: data URI ────────────────────────────────────────────────────

    def _fetch_data_uri(self, src: str) -> tuple[bytes | None, str]:
        m = _DATA_URI_RE.match(src)
        if not m:
            return None, ""
        try:
            raw = base64.b64decode(m.group("data"))
            if len(raw) > self._max_bytes:
                return None, ""
            return raw, m.group("mime")
        except Exception:
            return None, ""

    # ── Internal: S3 ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_s3_url(src: str) -> tuple[str, str] | tuple[None, None]:
        for pattern in (_S3_PROTOCOL_RE, _S3_VHOST_RE, _S3_PATH_RE):
            m = pattern.match(src)
            if m:
                return m.group("bucket"), m.group("key")
        return None, None

    def _fetch_s3(self, bucket: str, key: str, original_src: str) -> tuple[bytes | None, str]:
        try:
            resp = self._s3.get_object(Bucket=bucket, Key=key)

            content_length = int(resp.get("ContentLength") or 0)
            if content_length > self._max_bytes:
                logger.debug("ImageFetcher: skipping S3 image %s — %d bytes exceeds limit", original_src, content_length)
                return None, ""

            raw = resp["Body"].read()
            if len(raw) > self._max_bytes:
                return None, ""

            # Prefer S3 ContentType; fall back to key suffix
            mime = (resp.get("ContentType") or "").split(";")[0].strip()
            if not mime or not mime.startswith("image/"):
                suffix = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
                mime = _MIME_MAP.get(suffix, "image/png")

            return raw, mime

        except Exception as exc:
            logger.warning("ImageFetcher: failed to fetch S3 image %s — %s", original_src, exc)
            return None, ""

    @property
    def _s3(self):
        if self._s3_client is None:
            import boto3
            kwargs = {"region_name": self._s3_region} if self._s3_region else {}
            self._s3_client = boto3.client("s3", **kwargs)
        return self._s3_client

    # ── Internal: local file ─────────────────────────────────────────────────

    def _fetch_local(self, src: str, html_file_dir: Path) -> tuple[bytes | None, str]:
        candidate = Path(src)
        if not candidate.is_absolute():
            candidate = html_file_dir / candidate
        candidate = candidate.resolve()

        if not candidate.is_file():
            return None, ""
        if candidate.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            return None, ""

        try:
            raw = candidate.read_bytes()
            if len(raw) > self._max_bytes:
                return None, ""
            return raw, _MIME_MAP.get(candidate.suffix.lower(), "image/png")
        except Exception:
            return None, ""


# Module-level singleton — shared across all FileExtractor instances.
_image_fetcher = ImageFetcher()


# ── Module-level convenience wrappers ─────────────────────────────────────────

def resolve_image_bytes_and_mime(
    src: str,
    html_file_dir: Path,
) -> tuple[bytes | None, str]:
    """Resolve an ``<img src>`` to raw bytes + MIME type.

    Thin wrapper around ``ImageFetcher().fetch()`` kept for backward compat.
    """
    return _image_fetcher.fetch(src, html_file_dir)


def image_bytes_to_data_uri(image_bytes: bytes, mime_type: str) -> str:
    """Encode raw image bytes as a base64 data URI string."""
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


# ── Textract OCR ──────────────────────────────────────────────────────────────

class TextractOCR:
    """Thin wrapper around AWS Textract detect_document_text.

    Textract returns individual word/line blocks with confidence scores.
    We collect LINE blocks above min_confidence and join them with newlines,
    preserving the natural reading order Textract infers from the layout.
    """

    def __init__(self, region: str, min_confidence: float = 80.0) -> None:
        self.region = region
        self.min_confidence = min_confidence
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("textract", region_name=self.region)
        return self._client

    def extract_from_path(self, image_path: str) -> str:
        """Return OCR text for a local image file, or '' on any error."""
        path = Path(image_path)
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            return ""
        try:
            return self._call_textract(path.read_bytes())
        except Exception:
            return ""

    def extract_from_data_uri(self, data_uri: str) -> str:
        """Return OCR text for a base64 data URI, or '' on any error."""
        m = _DATA_URI_RE.match((data_uri or "").strip())
        if not m:
            return ""
        try:
            return self._call_textract(base64.b64decode(m.group("data")))
        except Exception:
            return ""

    def extract_from_src(self, src: str, html_file_dir: Path) -> str:
        """Resolve an <img src> and return OCR text."""
        image_bytes, _ = resolve_image_bytes_and_mime(src, html_file_dir)
        if not image_bytes:
            return ""
        return self._call_textract(image_bytes)

    def extract_from_bytes(self, image_bytes: bytes, mime_type: str = "") -> str:
        """Return OCR text for raw image bytes."""
        return self._call_textract(image_bytes)

    def _call_textract(self, image_bytes: bytes) -> str:
        response = self.client.detect_document_text(Document={"Bytes": image_bytes})
        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block.get("BlockType") == "LINE"
            and float(block.get("Confidence", 0)) >= self.min_confidence
        ]
        return "\n".join(lines).strip()


# ── Claude Vision OCR ─────────────────────────────────────────────────────────

class ClaudeVisionOCR:
    """OCR via Claude's vision capability (``ocr_provider="claude"``).

    Uses the Anthropic SDK directly — requires ``ANTHROPIC_API_KEY`` in the
    environment.  Uses claude-haiku for speed and cost efficiency.
    """

    _MODEL = "claude-haiku-4-5-20251001"

    _PROMPT = (
        "Extract all text from this image exactly as it appears. "
        "Preserve the original structure — use newlines between rows/sections "
        "and keep numbers, symbols, and punctuation intact. "
        "If there is no text, respond with an empty string."
    )

    def __init__(self) -> None:
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            self._client = anthropic.Anthropic(api_key=api_key or None)
        return self._client

    def extract_from_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Extract text from raw image bytes using Claude vision."""
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        try:
            msg = self.client.messages.create(
                model=self._MODEL,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": self._PROMPT},
                    ],
                }],
            )
            return msg.content[0].text.strip()
        except Exception:
            return ""

    def extract_from_src(self, src: str, html_file_dir: Path) -> str:
        """Resolve an <img src> and extract text via Claude vision."""
        image_bytes, mime_type = resolve_image_bytes_and_mime(src, html_file_dir)
        if not image_bytes:
            return ""
        return self.extract_from_bytes(image_bytes, mime_type or "image/png")


# ── OcrExtractor ABC ──────────────────────────────────────────────────────────

class OcrExtractor(ABC):
    """Provider-agnostic interface for OCR on raw image bytes.

    Both TextractOCR and ClaudeVisionOCR satisfy this interface.
    PdfToHtmlConverter depends on this type, not a concrete class.
    """

    @abstractmethod
    def extract_from_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Return OCR text for raw image bytes, or '' on any error."""


# Register existing classes as virtual subclasses so isinstance() works
OcrExtractor.register(TextractOCR)
OcrExtractor.register(ClaudeVisionOCR)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_ocr_extractor(
    provider: str,
    region: str = "",
    min_confidence: float = 80.0,
) -> OcrExtractor | None:
    """Instantiate the correct OCR backend from config.

    Parameters
    ----------
    provider:
        ``"textract"`` | ``"claude"`` | ``""`` (disabled).
    region:
        AWS region — required for Textract.
    min_confidence:
        Textract confidence floor (0–100).

    Returns
    -------
    An OcrExtractor instance, or None when OCR is disabled.
    """
    provider = (provider or "").strip().lower()
    if not provider:
        return None
    if provider == "textract":
        if not region:
            raise ValueError("build_ocr_extractor: region is required for Textract OCR")
        logger.info("OCR backend: Textract (region=%s, min_confidence=%.0f)", region, min_confidence)
        return TextractOCR(region=region, min_confidence=min_confidence)
    if provider == "claude":
        logger.info("OCR backend: Claude Vision")
        return ClaudeVisionOCR()
    raise ValueError(
        f"Unknown ocr_provider {provider!r}. Supported: 'textract', 'claude'."
    )
