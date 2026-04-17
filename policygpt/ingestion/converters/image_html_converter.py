"""ImageToHtmlConverter — wraps images in minimal HTML for the OCR pipeline.

Instead of extracting text here (which would duplicate the OCR extractor's
work), this converter wraps the image in a trivial HTML document containing
a single <img> tag.  The existing HtmlExtractor + OCR pipeline then picks
it up, extracts text via AWS Textract, and indexes it normally.

Supported formats: PNG, JPG/JPEG, GIF, BMP, TIFF, WEBP.

Caching
-------
The wrapper HTML is saved to {output_dir}/{stem}.html and reused on subsequent
runs as long as the HTML file is newer than the source image.
"""

from __future__ import annotations

import html as _html_lib
import logging
from pathlib import Path

from policygpt.ingestion.converters.base import HtmlConverter

logger = logging.getLogger(__name__)

# Map of file extensions to MIME types for the <img> src data URI.
_IMAGE_MIME: dict[str, str] = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
    ".webp": "image/webp",
}


class ImageToHtmlConverter(HtmlConverter):
    """Wraps image files in an HTML document for the OCR pipeline."""

    @property
    def supported_content_types(self) -> frozenset[str]:
        # Include the generic "image" content_type used by SQS/API readers as
        # well as the extension-based types emitted by the folder reader.
        return frozenset({
            "image",
            "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp",
        })

    def _convert_to_html(self, path: Path) -> str:
        import base64

        title = self._title_from_stem(path.stem)
        mime_type = _IMAGE_MIME.get(path.suffix.lower(), "image/png")

        # Embed image as a base64 data URI so the HTML is self-contained and
        # the OCR extractor can re-read it without needing the original path.
        image_bytes = path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"

        escaped_title = _html_lib.escape(title)
        escaped_path = _html_lib.escape(str(path))
        body = (
            f'<figure>\n'
            f'  <img src="{data_uri}" alt="{escaped_title}" '
            f'data-source-path="{escaped_path}">\n'
            f'  <figcaption>{escaped_title}</figcaption>\n'
            f'</figure>'
        )
        return self._wrap_html(title, body)
