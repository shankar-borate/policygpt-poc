"""Explainer base types — DocumentContext, UnitContent, PageExplainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DocumentContext:
    """Document-level context built during pass 1, fed into pass 2 explanations."""

    title: str
    doc_type: str    # "PowerPoint" | "PDF" | "Excel" | "Image"
    summary: str     # 2-3 sentence LLM-generated summary of the whole doc
    total_units: int # total slides / pages / sheets


@dataclass
class UnitContent:
    """One slide, page, sheet, or image file — input to the explainer."""

    unit_index: int         # 1-based
    unit_label: str         # "slide" | "page" | "sheet" | "image"
    text: str               # extracted text from the unit
    image_bytes: bytes = field(default=b"")  # raw bytes if unit has a picture
    mime_type: str = ""
    prev_explanation: str = ""  # rolling context from the previous unit

    @property
    def is_image(self) -> bool:
        """True when the unit is image-dominant (has image bytes and very little text)."""
        return bool(self.image_bytes) and len(self.text.strip()) < 50

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


class PageExplainer(ABC):
    """Abstract base for all explainer implementations."""

    @abstractmethod
    def explain(self, unit: UnitContent, ctx: DocumentContext | None) -> str:
        """Return an HTML fragment (unit-explanation div) or empty string."""
