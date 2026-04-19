"""HtmlConverterRegistry — maps content_type → HtmlConverter instance.

Usage
-----
registry = HtmlConverterRegistry(output_dir=html_dir)
if registry.supports("pdf"):
    html_path, html_content = registry.convert("pdf", source_path)

Registering a new format
-------------------------
1. Subclass HtmlConverter, implement _convert_to_html() and supported_content_types.
2. Import the class below and add it to _CONVERTER_CLASSES.
Done — nothing else changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Type

from policygpt.ingestion.converters.base import HtmlConverter
from policygpt.ingestion.converters.pdf_html_converter import PdfToHtmlConverter
from policygpt.ingestion.converters.ppt_html_converter import PptToHtmlConverter
from policygpt.ingestion.converters.docx_html_converter import DocxToHtmlConverter
from policygpt.ingestion.converters.excel_html_converter import ExcelToHtmlConverter
from policygpt.ingestion.converters.image_html_converter import ImageToHtmlConverter

if TYPE_CHECKING:
    from policygpt.ingestion.converters.vision import VisionDescriber
    from policygpt.ingestion.extraction.ocr import OcrExtractor

logger = logging.getLogger(__name__)

# Ordered list of converter classes to register at startup.
# Add new classes here — their supported_content_types drive the mapping.
_CONVERTER_CLASSES: list[Type[HtmlConverter]] = [
    PdfToHtmlConverter,
    PptToHtmlConverter,
    DocxToHtmlConverter,
    ExcelToHtmlConverter,
    ImageToHtmlConverter,
]


class HtmlConverterRegistry:
    """Central registry that selects the right HtmlConverter for a content type.

    All converters are instantiated once at construction time, sharing the
    same output directory.

    Parameters
    ----------
    output_dir:
        Directory where converted HTML files are written.
        Typically {debug_log_dir}/html/
    skip_if_cached:
        When True (default), skip conversion when the HTML file is newer than
        the source document.
    skip_content_types:
        Set of content-type strings to exclude from this registry.
        Use this to disable individual format converters via config flags
        (e.g. ``{"xlsx", "xls"}`` to skip Excel).
    vision_describer:
        Optional vision model for images across all formats (PDF, PPT, DOCX, standalone).
        Forwarded to PdfToHtmlConverter, PptToHtmlConverter, DocxToHtmlConverter,
        and ImageToHtmlConverter.
    ocr:
        Optional OCR extractor fallback for image content in all supported formats.
    """

    def __init__(
        self,
        output_dir: str | Path,
        skip_if_cached: bool = True,
        skip_content_types: frozenset[str] = frozenset(),
        vision_describer: "VisionDescriber | None" = None,
        ocr: "OcrExtractor | None" = None,
    ) -> None:
        self._registry: dict[str, HtmlConverter] = {}
        _vision_converters = (
            PdfToHtmlConverter,
            PptToHtmlConverter,
            DocxToHtmlConverter,
            ImageToHtmlConverter,
        )
        for cls in _CONVERTER_CLASSES:
            if cls in _vision_converters:
                instance: HtmlConverter = cls(
                    output_dir=output_dir,
                    skip_if_cached=skip_if_cached,
                    vision_describer=vision_describer,
                    ocr=ocr,
                )
            else:
                instance = cls(output_dir=output_dir, skip_if_cached=skip_if_cached)
            for ct in instance.supported_content_types:
                if ct in skip_content_types:
                    logger.info("HtmlConverterRegistry: skipping %r (disabled by config)", ct)
                    continue
                if ct in self._registry:
                    logger.warning(
                        "Content-type %r already registered by %s — overriding with %s",
                        ct,
                        type(self._registry[ct]).__name__,
                        cls.__name__,
                    )
                self._registry[ct] = instance
        logger.debug(
            "HtmlConverterRegistry ready: %s", sorted(self._registry.keys())
        )

    def supports(self, content_type: str) -> bool:
        """Return True if a converter is registered for *content_type*."""
        return content_type in self._registry

    def convert(self, content_type: str, source_path: str) -> tuple[str, str]:
        """Convert the document to HTML using the registered converter.

        Returns
        -------
        (html_path, html_content)

        Raises
        ------
        KeyError
            If no converter is registered for *content_type*.
        """
        try:
            converter = self._registry[content_type]
        except KeyError:
            supported = sorted(self._registry.keys())
            raise KeyError(
                f"No HTML converter registered for content_type={content_type!r}. "
                f"Supported: {supported}"
            ) from None
        return converter.convert(source_path)

    def convert_all(self, content_type: str, source_path: str) -> list[tuple[str, str]]:
        """Convert the document to one or more HTML files.

        Most formats produce a single file.  Excel workbooks with multiple
        sheets produce one file per sheet.

        Returns
        -------
        list of (html_path, html_content) — at least one entry on success.

        Raises
        ------
        KeyError
            If no converter is registered for *content_type*.
        """
        try:
            converter = self._registry[content_type]
        except KeyError:
            supported = sorted(self._registry.keys())
            raise KeyError(
                f"No HTML converter registered for content_type={content_type!r}. "
                f"Supported: {supported}"
            ) from None
        return converter.convert_all(source_path)

    @property
    def supported_content_types(self) -> frozenset[str]:
        """All content types this registry can convert."""
        return frozenset(self._registry.keys())
