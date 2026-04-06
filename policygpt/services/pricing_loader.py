from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from policygpt.config import Config
from policygpt.services.usage_metrics import ModelPricingSnapshot


OPENAI_GPT41_PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-4.1"
AWS_BEDROCK_PRICING_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/index.json"
AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/index.json"
)


@dataclass(frozen=True)
class _BedrockLookupRule:
    display_name: str
    source_url: str
    offer_url: str


FALLBACK_PRICING_BY_MODEL: dict[str, dict[str, Any]] = {
    "gpt-4.1": {
        "display_name": "OpenAI GPT-4.1",
        "input_price_per_million_usd": 2.0,
        "output_price_per_million_usd": 8.0,
        "source_url": OPENAI_GPT41_PRICING_URL,
    },
    "openai.gpt-oss-20b-1:0": {
        "display_name": "OpenAI GPT OSS 20B via Bedrock",
        "input_price_per_million_usd": 0.08,
        "output_price_per_million_usd": 0.35,
        "source_url": AWS_BEDROCK_PRICING_URL,
    },
    "openai.gpt-oss-120b-1:0": {
        "display_name": "OpenAI GPT OSS 120B via Bedrock",
        "input_price_per_million_usd": 0.18,
        "output_price_per_million_usd": 0.71,
        "source_url": AWS_BEDROCK_PRICING_URL,
    },
    "global.anthropic.claude-sonnet-4-6": {
        "display_name": "Claude Sonnet 4.6 via Bedrock",
        "input_price_per_million_usd": 3.0,
        "output_price_per_million_usd": 15.0,
        "source_url": AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL,
    },
    "global.anthropic.claude-opus-4-6-v1": {
        "display_name": "Claude Opus 4.6 via Bedrock",
        "input_price_per_million_usd": 5.0,
        "output_price_per_million_usd": 25.0,
        "source_url": AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL,
    },
}


class ModelPricingLoader:
    def load_snapshot(self, config: Config) -> ModelPricingSnapshot:
        model_name = config.chat_model
        try:
            if model_name == "gpt-4.1":
                return self._load_openai_gpt41_snapshot(model_name)
            if model_name.startswith("openai.gpt-oss-20b"):
                return self._load_bedrock_gpt_oss_snapshot(config, model_name, model_slug="gpt-oss-20b")
            if model_name.startswith("openai.gpt-oss-120b"):
                return self._load_bedrock_gpt_oss_snapshot(config, model_name, model_slug="gpt-oss-120b")
            if "claude-sonnet-4-6" in model_name:
                return self._load_bedrock_anthropic_snapshot(
                    config=config,
                    model_name=model_name,
                    service_name="Claude Sonnet 4.6 (Amazon Bedrock Edition)",
                    display_name="Claude Sonnet 4.6 via Bedrock",
                )
            if "claude-opus-4-6" in model_name:
                return self._load_bedrock_anthropic_snapshot(
                    config=config,
                    model_name=model_name,
                    service_name="Claude Opus 4.6 (Amazon Bedrock Edition)",
                    display_name="Claude Opus 4.6 via Bedrock",
                )
        except Exception:
            fallback = self._fallback_snapshot(model_name)
            if fallback is not None:
                return fallback
            raise

        fallback = self._fallback_snapshot(model_name)
        return fallback or ModelPricingSnapshot(
            model_name=model_name,
            display_name=model_name,
            source_status="unavailable",
        )

    def _load_openai_gpt41_snapshot(self, model_name: str) -> ModelPricingSnapshot:
        html = self._fetch_text(OPENAI_GPT41_PRICING_URL)
        input_price, output_price = self._parse_openai_gpt41_pricing(html)
        return ModelPricingSnapshot(
            model_name=model_name,
            display_name="OpenAI GPT-4.1",
            input_price_per_million_usd=input_price,
            output_price_per_million_usd=output_price,
            source_url=OPENAI_GPT41_PRICING_URL,
            source_status="live",
        )

    def _load_bedrock_gpt_oss_snapshot(self, config: Config, model_name: str, model_slug: str) -> ModelPricingSnapshot:
        offer_index = self._fetch_json(AWS_BEDROCK_PRICING_URL)
        input_price = self._extract_bedrock_offer_price(
            offer_index=offer_index,
            region_code=config.bedrock_region,
            matcher=lambda attrs: attrs.get("model") == model_slug and self._is_standard_input_tokens(attrs),
        )
        output_price = self._extract_bedrock_offer_price(
            offer_index=offer_index,
            region_code=config.bedrock_region,
            matcher=lambda attrs: attrs.get("model") == model_slug and self._is_standard_output_tokens(attrs),
        )
        display_name = "OpenAI GPT OSS 20B via Bedrock" if model_slug.endswith("20b") else "OpenAI GPT OSS 120B via Bedrock"
        return ModelPricingSnapshot(
            model_name=model_name,
            display_name=display_name,
            input_price_per_million_usd=input_price,
            output_price_per_million_usd=output_price,
            source_url=AWS_BEDROCK_PRICING_URL,
            source_status="live",
        )

    def _load_bedrock_anthropic_snapshot(
        self,
        *,
        config: Config,
        model_name: str,
        service_name: str,
        display_name: str,
    ) -> ModelPricingSnapshot:
        offer_index = self._fetch_json(AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL)
        is_global = model_name.startswith("global.")
        input_price = self._extract_bedrock_offer_price(
            offer_index=offer_index,
            region_code=config.bedrock_region,
            matcher=lambda attrs: self._matches_anthropic_usage(attrs, service_name=service_name, global_expected=is_global, usage_kind="input"),
        )
        output_price = self._extract_bedrock_offer_price(
            offer_index=offer_index,
            region_code=config.bedrock_region,
            matcher=lambda attrs: self._matches_anthropic_usage(attrs, service_name=service_name, global_expected=is_global, usage_kind="output"),
        )
        return ModelPricingSnapshot(
            model_name=model_name,
            display_name=display_name,
            input_price_per_million_usd=input_price,
            output_price_per_million_usd=output_price,
            source_url=AWS_BEDROCK_FOUNDATION_MODELS_PRICING_URL,
            source_status="live",
        )

    @staticmethod
    def _parse_openai_gpt41_pricing(html: str) -> tuple[float, float]:
        input_match = re.search(
            r"<div>\s*Input\s*</div>\s*<div[^>]*>\$([0-9]+(?:\.[0-9]+)?)</div>",
            html,
            flags=re.IGNORECASE,
        )
        output_match = re.search(
            r"<div>\s*Output\s*</div>\s*<div[^>]*>\$([0-9]+(?:\.[0-9]+)?)</div>",
            html,
            flags=re.IGNORECASE,
        )
        if not input_match or not output_match:
            raise ValueError("Could not parse GPT-4.1 pricing from the OpenAI model page.")
        return float(input_match.group(1)), float(output_match.group(1))

    @staticmethod
    def _is_standard_input_tokens(attributes: dict[str, Any]) -> bool:
        return ModelPricingLoader._is_standard_token_type(attributes, token_type="input")

    @staticmethod
    def _is_standard_output_tokens(attributes: dict[str, Any]) -> bool:
        return ModelPricingLoader._is_standard_token_type(attributes, token_type="output")

    @staticmethod
    def _is_standard_token_type(attributes: dict[str, Any], token_type: str) -> bool:
        inference_type = (attributes.get("inferenceType") or "").strip().lower()
        if inference_type != f"{token_type} tokens":
            return False
        feature = (attributes.get("feature") or "").strip().lower()
        service_tier = (attributes.get("service_tier") or "").strip().lower()
        return feature == "on-demand inference" or service_tier == "standard"

    @staticmethod
    def _matches_anthropic_usage(
        attributes: dict[str, Any],
        *,
        service_name: str,
        global_expected: bool,
        usage_kind: str,
    ) -> bool:
        if attributes.get("servicename") != service_name:
            return False

        usage_type = attributes.get("usagetype") or ""
        if global_expected != ("Global" in usage_type):
            return False

        # The widget is meant for standard runtime tokens, not reserved,
        # batch, cache, or long-context variants.
        if any(marker in usage_type for marker in ("Batch", "Reserved", "Cache", "LCtx")):
            return False

        expected_marker = "InputTokenCount" if usage_kind == "input" else "OutputTokenCount"
        return expected_marker in usage_type

    @staticmethod
    def _extract_bedrock_offer_price(
        *,
        offer_index: dict[str, Any],
        region_code: str,
        matcher,
    ) -> float:
        products = offer_index.get("products") or {}
        terms = ((offer_index.get("terms") or {}).get("OnDemand") or {})
        candidates: list[tuple[int, float]] = []

        for sku, product in products.items():
            attributes = product.get("attributes") or {}
            if attributes.get("regionCode") != region_code:
                continue
            if not matcher(attributes):
                continue

            term_block = next(iter((terms.get(sku) or {}).values()), None)
            if not term_block:
                continue
            price_dimension = next(iter((term_block.get("priceDimensions") or {}).values()), None)
            if not price_dimension:
                continue

            usd_value = price_dimension.get("pricePerUnit", {}).get("USD")
            if usd_value in (None, ""):
                continue

            description = price_dimension.get("description", "")
            normalized_price = ModelPricingLoader._normalize_price_per_million(float(usd_value), description=description)
            score = 0
            if (attributes.get("service_tier") or "").strip().lower() == "standard":
                score += 3
            if (attributes.get("feature") or "").strip().lower() == "on-demand inference":
                score += 2
            candidates.append((score, normalized_price))

        if not candidates:
            raise ValueError(f"Could not resolve Bedrock pricing for region {region_code}.")

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _normalize_price_per_million(raw_price: float, *, description: str) -> float:
        normalized_description = (description or "").lower()
        if "per 1k token" in normalized_description or "per 1k input token" in normalized_description or "per 1k output token" in normalized_description:
            return raw_price * 1000.0
        if "million" in normalized_description:
            return raw_price
        # Default Bedrock token prices to per-million if the description is
        # missing the unit marker.
        return raw_price

    @staticmethod
    def _fetch_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="ignore")

    @staticmethod
    def _fetch_json(url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    @staticmethod
    def _fallback_snapshot(model_name: str) -> ModelPricingSnapshot | None:
        fallback = FALLBACK_PRICING_BY_MODEL.get(model_name)
        if fallback is None:
            return None
        return ModelPricingSnapshot(
            model_name=model_name,
            display_name=fallback["display_name"],
            input_price_per_million_usd=fallback["input_price_per_million_usd"],
            output_price_per_million_usd=fallback["output_price_per_million_usd"],
            source_url=fallback["source_url"],
            source_status="fallback",
        )
