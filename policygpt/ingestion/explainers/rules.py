"""ExplainRules — decides whether and how to explain a unit."""

from __future__ import annotations

from policygpt.ingestion.explainers.base import UnitContent


class ExplainRules:
    """Stateless rule set for the explainability feature.

    Parameters
    ----------
    min_chars:
        Units with fewer extracted characters than this threshold are
        considered sparse and will be explained via text LLM.
        Default: 250 chars (~50 words).
    """

    def __init__(self, min_chars: int = 250) -> None:
        self.min_chars = min_chars

    def should_explain(self, unit: UnitContent) -> bool:
        """Return True if this unit needs an explanation."""
        return unit.is_image or unit.char_count < self.min_chars

    def explainer_mode(self, unit: UnitContent) -> str:
        """Return 'vision' for image-dominant units, 'text' for sparse-text units."""
        return "vision" if unit.is_image else "text"
