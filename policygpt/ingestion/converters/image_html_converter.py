"""ImageToHtmlConverter — converts image files to HTML via vision model or OCR.

Instead of just wrapping the image in an <img> tag, this converter sends the
image to a configurable vision LLM for a rich, structured HTML description —
the same pipeline used for image-based PDF pages.

Fallback chain (same as PdfToHtmlConverter._convert_image_page):
  1. VisionDescriber (LLM)  — detailed semantic HTML
  2. OcrExtractor           — plain text wrapped in HTML
  3. <img> embed fallback   — bare image wrapped in HTML (no text extraction)

Supported formats: PNG, JPG/JPEG, GIF, BMP, TIFF, WEBP.

Caching
-------
The converted HTML is saved to {output_dir}/{stem}.html and reused on
subsequent runs as long as the HTML file is newer than the source image.
"""

from __future__ import annotations

import base64
import html as _html_lib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from policygpt.ingestion.converters.base import HtmlConverter

if TYPE_CHECKING:
    from policygpt.ingestion.converters.vision import VisionDescriber
    from policygpt.ingestion.extraction.ocr import OcrExtractor
    from policygpt.ingestion.explainers.factory import ExplainerFactory

logger = logging.getLogger(__name__)

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
    """Converts image files to structured HTML via vision model or OCR.

    Parameters
    ----------
    output_dir:
        Directory where converted HTML files are written.
    skip_if_cached:
        Reuse existing HTML when newer than the source image.
    vision_describer:
        Optional LLM-based vision model for rich HTML description.
    ocr:
        Optional OCR extractor used as fallback when vision is unavailable
        or returns empty.
    """

    def __init__(
        self,
        output_dir: str | Path,
        skip_if_cached: bool = True,
        vision_describer: "VisionDescriber | None" = None,
        ocr: "OcrExtractor | None" = None,
        explainer: "ExplainerFactory | None" = None,
    ) -> None:
        super().__init__(output_dir=output_dir, skip_if_cached=skip_if_cached)
        self._vision = vision_describer
        self._ocr = ocr
        self._explainer = explainer

    @property
    def supported_content_types(self) -> frozenset[str]:
        return frozenset({
            "image",
            "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp",
        })

    def _convert_to_html(self, path: Path) -> str:
        title = self._title_from_stem(path.stem)
        mime_type = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
        image_bytes = path.read_bytes()

        logger.info(
            "ImageToHtmlConverter: %s — %.1f KB (%s)",
            path.name, len(image_bytes) / 1024, mime_type,
        )
        print(f"    [IMG] {path.name} — {len(image_bytes)/1024:.1f} KB ({mime_type})", flush=True)

        body = self._describe_image(image_bytes, mime_type, path)

        # Images are always explained — no doc context needed (single-unit file)
        if self._explainer is not None:
            import re as _re
            from policygpt.ingestion.explainers.base import UnitContent
            plain = _re.sub(r"<[^>]+>", " ", body)
            plain = _re.sub(r"\s+", " ", plain).strip()
            unit = UnitContent(
                unit_index=1,
                unit_label="image",
                text=plain,
            )
            explanation = self._explainer.explain_unit(unit, ctx=None)
            if explanation:
                body = body + "\n" + explanation

        return self._wrap_html(title, body)

    def _describe_image(self, image_bytes: bytes, mime_type: str, path: Path) -> str:
        # 1. Vision model — rich semantic HTML
        if self._vision is not None:
            logger.info(
                "ImageToHtmlConverter: calling vision [%s] for %s …",
                self._vision.provider_name, path.name,
            )
            print(
                f"    [IMG] {path.name} — vision [{self._vision.provider_name}] …",
                flush=True,
            )
            html_fragment = self._vision.describe_page(image_bytes, mime_type)
            if html_fragment.strip():
                logger.info(
                    "ImageToHtmlConverter: vision OK — %d chars (%s)",
                    len(html_fragment), path.name,
                )
                print(
                    f"    [IMG] {path.name} — vision OK ({len(html_fragment)} chars)",
                    flush=True,
                )
                return (
                    f'<div class="image-content" '
                    f'data-source="vision:{self._vision.provider_name}">\n'
                    f"{html_fragment}\n"
                    f"</div>"
                )
            logger.warning(
                "ImageToHtmlConverter: vision returned empty for %s, trying OCR", path.name
            )
            print(f"    [IMG] {path.name} — vision empty, trying OCR …", flush=True)

        # 2. OCR fallback — plain text wrapped in HTML
        if self._ocr is not None:
            logger.info("ImageToHtmlConverter: calling OCR for %s …", path.name)
            print(f"    [IMG] {path.name} — OCR …", flush=True)
            ocr_text = self._ocr.extract_from_bytes(image_bytes, mime_type)
            if ocr_text.strip():
                lines = [
                    f"<p>{_html_lib.escape(line)}</p>"
                    for line in ocr_text.splitlines()
                    if line.strip()
                ]
                inner = "\n".join(lines)
                logger.info(
                    "ImageToHtmlConverter: OCR OK — %d chars (%s)", len(ocr_text), path.name
                )
                print(
                    f"    [IMG] {path.name} — OCR OK ({len(ocr_text)} chars)", flush=True
                )
                return f'<div class="image-content" data-source="ocr">\n{inner}\n</div>'
            logger.warning("ImageToHtmlConverter: OCR returned empty for %s", path.name)

        # 3. Bare image embed fallback — at least preserve the visual
        logger.warning(
            "ImageToHtmlConverter: no vision/OCR output for %s — embedding raw image", path.name
        )
        print(f"    [IMG] {path.name} — fallback: embedding image", flush=True)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"
        escaped_title = _html_lib.escape(path.stem)
        escaped_path = _html_lib.escape(str(path))
        return (
            f'<figure>\n'
            f'  <img src="{data_uri}" alt="{escaped_title}" '
            f'data-source-path="{escaped_path}">\n'
            f'  <figcaption>{escaped_title}</figcaption>\n'
            f'</figure>'
        )
