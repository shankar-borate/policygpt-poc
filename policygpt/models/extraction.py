"""Extraction result models shared between extraction/ and ingestion/extractors/."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractedSection:
    """One logical section extracted from a document.

    Attributes
    ----------
    title:
        Section heading (derived from <hN> tag, slide title, etc.).
    text:
        Plain text content of the section — no HTML, no markdown.
    images:
        Base64 data URIs (``data:image/png;base64,...``) for images found
        within this section.  Empty for text-only sections.
    """

    title: str
    text: str
    images: list[str] = field(default_factory=list)
