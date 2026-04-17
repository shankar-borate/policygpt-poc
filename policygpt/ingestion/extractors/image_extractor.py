"""ImageExtractor — extracts text from standalone image files via OCR.

Delegates to TextractOCR (AWS Textract) when config.ocr_enabled=True.
When OCR is disabled the document is ingested with empty sections (best-effort).

Supported image formats: JPEG, PNG, TIFF, BMP.

Implementation notes for future enhancement
------------------------------------------
- Local images → TextractOCR.extract_from_path()
- The entire image is treated as a single section titled "Image Text".
- Title is derived from the file name (stem, humanized).
- For multi-page TIFFs, split pages and OCR each as a separate section.
"""

from __future__ import annotations

import logging
from pathlib import Path

from policygpt.config import Config
from policygpt.ingestion.extractors.base import ExtractedDocument, Extractor
from policygpt.ingestion.readers.base import IngestMessage

logger = logging.getLogger(__name__)

_HUMANIZE_CHARS = str.maketrans("_-", "  ")


def _humanize_stem(stem: str) -> str:
    return stem.translate(_HUMANIZE_CHARS).title()


class ImageExtractor(Extractor):
    """Extracts text from image files using AWS Textract OCR.

    When ocr_enabled=False in config the document is returned with no sections
    (zero-section documents are skipped by IngestionPipeline).

    Parameters
    ----------
    config:
        Application config — drives ocr_enabled, ocr_min_confidence, aws_region.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._ocr = None  # lazy init — only when ocr_enabled=True

    @property
    def supported_content_types(self) -> frozenset[str]:
        # "image" — legacy generic type used by SQS/API readers.
        # Individual extension types — emitted by FolderReader for raw images.
        return frozenset({
            "image",
            "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp",
        })

    def extract(self, message: IngestMessage) -> ExtractedDocument:
        if message.content_type not in self.supported_content_types:
            raise ValueError(
                f"ImageExtractor does not handle content_type={message.content_type!r}"
            )
        title = _humanize_stem(Path(message.file_name).stem)

        if not self._config.ocr_enabled:
            logger.debug(
                "OCR disabled — skipping image %s", message.source_path
            )
            return ExtractedDocument(title=title, sections=[])

        ocr = self._get_ocr()
        ocr_text = ocr.extract_from_path(message.source_path)
        if not ocr_text.strip():
            logger.debug("No OCR text extracted from %s", message.source_path)
            return ExtractedDocument(title=title, sections=[])

        return ExtractedDocument(
            title=title,
            sections=[("Image Text", ocr_text)],
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_ocr(self):
        if self._ocr is None:
            from policygpt.extraction.ocr import TextractOCR

            self._ocr = TextractOCR(
                region=self._config.aws_region,
                min_confidence=self._config.ocr_min_confidence,
            )
        return self._ocr
