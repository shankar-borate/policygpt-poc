"""PptExtractor — placeholder for PowerPoint content extraction.

STATUS: Not yet implemented.

Implementation notes
--------------------
Use python-pptx (https://python-pptx.readthedocs.io) to:
  1. Iterate over slides — each slide becomes a section.
  2. Collect text from shape.text_frame for text/title placeholders.
  3. Use the presentation title (from core_properties) or the first text box
     on slide 1 as the document title.
  4. For embedded images, optionally run TextractOCR on each shape's image
     blob when config.ocr_enabled=True.
  5. Return ExtractedDocument(title, [(slide_title, slide_text), ...]).

Dependencies to add to requirements.txt when implementing:
  python-pptx>=1.0.0
"""

from __future__ import annotations

import logging

from policygpt.config import Config
from policygpt.ingestion.pipeline_extractors.base import ExtractedDocument, Extractor
from policygpt.ingestion.readers.base import IngestMessage

logger = logging.getLogger(__name__)


class PptExtractor(Extractor):
    """Extracts title and sections from PowerPoint (.ppt/.pptx) documents.

    Each slide is treated as one section; the slide title shape (if present)
    becomes the section title, and all other text shapes are joined as the
    section body.

    Not yet implemented — raises NotImplementedError.
    """

    def __init__(self, config: Config) -> None:  # noqa: ARG002
        pass  # config not yet used; stored when implementation is added

    @property
    def supported_content_types(self) -> frozenset[str]:
        # "ppt" — legacy generic type; "pptx" — OOXML format from FolderReader.
        return frozenset({"ppt", "pptx"})

    def extract(self, message: IngestMessage) -> ExtractedDocument:
        if message.content_type not in self.supported_content_types:
            raise ValueError(
                f"PptExtractor does not handle content_type={message.content_type!r}"
            )
        raise NotImplementedError(
            "PptExtractor is not yet implemented. "
            "Install python-pptx and implement slide-by-slide extraction. "
            f"source_path={message.source_path!r}"
        )
