"""Backward-compatibility shim — import from canonical location."""
from policygpt.observability.pricing.pricing_loader import (
    ModelPricingLoader,
    FALLBACK_PRICING_BY_MODEL,
    OPENAI_MODEL_PRICING_URLS,
    AWS_BEDROCK_PRICING_URL,
    AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL,
)

__all__ = [
    "ModelPricingLoader",
    "FALLBACK_PRICING_BY_MODEL",
    "OPENAI_MODEL_PRICING_URLS",
    "AWS_BEDROCK_PRICING_URL",
    "AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL",
]
