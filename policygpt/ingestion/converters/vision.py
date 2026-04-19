"""Vision-based page description for image-heavy PDF pages.

When a PDF page yields no extractable text, a VisionDescriber renders it to
a PNG and sends it to a vision-capable LLM, which returns a detailed HTML
fragment preserving structure, tables, and semantic content.

Providers
---------
  ClaudeVisionDescriber  — Anthropic SDK (ANTHROPIC_API_KEY)
  OpenAIVisionDescriber  — OpenAI SDK   (OPENAI_API_KEY)

Use ``build_vision_describer(provider, model)`` to instantiate the right one
from config.  Returns None when provider is empty (vision disabled).
"""

from __future__ import annotations

import base64
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_PAGE_DESCRIPTION_PROMPT = """\
You are analyzing a page or image from a business, insurance, or banking policy document.
Your output will be used to answer FAQs and detailed questions, so be thorough and precise.

Produce a complete, structured HTML fragment with TWO sections:

--- SECTION 1: CONTENT EXTRACTION ---
Extract everything visible on the page/image exactly as written:

1. Extract ALL text exactly as written — headings, labels, body paragraphs, footnotes.
2. Tables — reproduce every row and column as an HTML <table> with <thead>/<tbody>.
   Include all values, units, thresholds, dates, and conditions without omission.
3. Charts / diagrams — describe the title, axes, legend, and ALL key data points in a <p>.
4. Stamps, signatures, logos, watermarks — note their presence in a <p class="annotation">.
5. Preserve heading hierarchy using <h2>, <h3>, <h4>.
6. Use <ul>/<li> for bullet or numbered lists.
7. Use <p> for plain paragraphs.

--- SECTION 2: DETAILED EXPLANATION ---
After the extracted content, add a <div class="ai-explanation"> block with a plain-language
explanation written for someone who needs to understand and act on this content:

- Summarize what this page/image is about in 2–3 sentences.
- Explain any tables: what each column means, what the data represents, and how to read it.
- Explain any charts or diagrams: what trend or insight they show.
- Call out key facts, limits, thresholds, eligibility conditions, deadlines, or amounts
  that a customer or agent would typically ask about.
- If this is a process/flow diagram, describe each step and its outcome.
- Write in clear, simple language suitable for FAQ answers and chatbot responses.

Output ONLY the inner HTML fragment — no <html>, <head>, or <body> tags.
If the page is blank or unreadable, output exactly: <p class="empty-page"></p>
"""


# ── Abstract base ─────────────────────────────────────────────────────────────

class VisionDescriber(ABC):
    """Converts a rendered page image to a structured HTML fragment.

    Subclasses implement provider-specific API calls.  All implementations
    are expected to be cheap to construct (lazy client initialisation) and
    safe to share as a long-lived singleton per pipeline run.
    """

    @abstractmethod
    def describe_page(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Return an HTML fragment describing the page.

        Parameters
        ----------
        image_bytes:
            Raw PNG (or JPEG/WEBP) bytes of the rendered page.
        mime_type:
            MIME type of the image (e.g. ``"image/png"``).

        Returns
        -------
        An HTML fragment ready to insert inside a ``<div class="page">``.
        Empty string on any error.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier used in log messages."""


# ── Claude ────────────────────────────────────────────────────────────────────

class ClaudeVisionDescriber(VisionDescriber):
    """Page description via Anthropic Claude vision.

    Requires the ``anthropic`` SDK (``pip install anthropic``) and
    ``ANTHROPIC_API_KEY`` set in the environment.

    Parameters
    ----------
    model:
        Claude model ID.  Defaults to ``claude-haiku-4-5-20251001`` for
        speed and cost efficiency.
    max_tokens:
        Maximum tokens in the model response.
    """

    _DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, model: str = "", max_tokens: int = 4096) -> None:
        self._model = model.strip() or self._DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._client = None  # lazy

    @property
    def provider_name(self) -> str:
        return f"claude/{self._model}"

    @property
    def _api_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "ClaudeVisionDescriber requires the 'anthropic' package. "
                    "Install it with: pip install anthropic"
                ) from exc
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            self._client = anthropic.Anthropic(api_key=api_key or None)
        return self._client

    def describe_page(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        if not image_bytes:
            return ""
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        logger.info(
            "ClaudeVisionDescriber: sending %.1f KB image to %s …",
            len(image_bytes) / 1024, self._model,
        )
        try:
            msg = self._api_client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
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
                        {"type": "text", "text": _PAGE_DESCRIPTION_PROMPT},
                    ],
                }],
            )
            result = (msg.content[0].text or "").strip()
            logger.info(
                "ClaudeVisionDescriber: received %d chars (model=%s, input_tokens=%s, output_tokens=%s)",
                len(result), self._model,
                getattr(msg.usage, "input_tokens", "?"),
                getattr(msg.usage, "output_tokens", "?"),
            )
            return result
        except Exception as exc:
            logger.warning("ClaudeVisionDescriber.describe_page failed: %s", exc)
            return ""


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIVisionDescriber(VisionDescriber):
    """Page description via OpenAI GPT-4o vision.

    Requires the ``openai`` SDK (``pip install openai>=1.0``) and
    ``OPENAI_API_KEY`` set in the environment.

    Parameters
    ----------
    model:
        OpenAI model ID.  Defaults to ``gpt-4o-mini`` for cost efficiency.
    max_tokens:
        Maximum tokens in the model response.
    """

    _DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str = "", max_tokens: int = 4096) -> None:
        self._model = model.strip() or self._DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._client = None  # lazy

    @property
    def provider_name(self) -> str:
        return f"openai/{self._model}"

    @property
    def _api_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "OpenAIVisionDescriber requires the 'openai' package. "
                    "Install it with: pip install openai"
                ) from exc
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            self._client = openai.OpenAI(api_key=api_key or None)
        return self._client

    def describe_page(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        if not image_bytes:
            return ""
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"
        logger.info(
            "OpenAIVisionDescriber: sending %.1f KB image to %s …",
            len(image_bytes) / 1024, self._model,
        )
        try:
            response = self._api_client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                        {"type": "text", "text": _PAGE_DESCRIPTION_PROMPT},
                    ],
                }],
            )
            result = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            logger.info(
                "OpenAIVisionDescriber: received %d chars (model=%s, prompt_tokens=%s, completion_tokens=%s)",
                len(result), self._model,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
            )
            return result
        except Exception as exc:
            logger.warning("OpenAIVisionDescriber.describe_page failed: %s", exc)
            return ""


# ── Factory ───────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[VisionDescriber]] = {
    "claude": ClaudeVisionDescriber,
    "openai": OpenAIVisionDescriber,
}


def build_vision_describer(provider: str, model: str = "") -> VisionDescriber | None:
    """Instantiate a VisionDescriber from config values.

    Parameters
    ----------
    provider:
        ``"claude"`` | ``"openai"`` | ``""`` (disabled).
    model:
        Optional model override.  Each provider falls back to its default
        when this is empty.

    Returns
    -------
    A VisionDescriber instance, or None when provider is empty/unknown.
    """
    provider = (provider or "").strip().lower()
    if not provider:
        return None
    cls = _PROVIDERS.get(provider)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown vision_provider {provider!r}. Supported: {supported}."
        )
    instance = cls(model=model)
    logger.info("VisionDescriber: using %s", instance.provider_name)
    return instance
