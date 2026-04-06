from __future__ import annotations

import threading
from dataclasses import dataclass, field

from policygpt.models import utc_now_iso


def estimate_text_tokens(text: str, chars_per_token: int = 4, tokens_per_word: float = 1.3) -> int:
    compact = " ".join((text or "").split())
    if not compact:
        return 0

    word_count = len(compact.split())
    char_based = max(1, (len(compact) + max(1, chars_per_token) - 1) // max(1, chars_per_token))
    word_based = max(1, int(word_count * max(tokens_per_word, 0.1)))
    return max(char_based, word_based)


@dataclass(frozen=True)
class ModelPricingSnapshot:
    model_name: str
    display_name: str
    input_price_per_million_usd: float | None = None
    output_price_per_million_usd: float | None = None
    source_url: str = ""
    source_status: str = "unavailable"  # live | fallback | unavailable
    loaded_at: str = field(default_factory=utc_now_iso)


class LLMUsageTracker:
    def __init__(self, model_name: str) -> None:
        self._lock = threading.RLock()
        self._pricing_snapshot = ModelPricingSnapshot(
            model_name=model_name,
            display_name=model_name,
        )
        self._input_tokens = 0
        self._output_tokens = 0
        self._call_count = 0
        self._updated_at = utc_now_iso()

    def reset(self, model_name: str) -> None:
        with self._lock:
            self._pricing_snapshot = ModelPricingSnapshot(
                model_name=model_name,
                display_name=model_name,
            )
            self._input_tokens = 0
            self._output_tokens = 0
            self._call_count = 0
            self._updated_at = utc_now_iso()

    def set_pricing_snapshot(self, snapshot: ModelPricingSnapshot) -> None:
        with self._lock:
            self._pricing_snapshot = snapshot
            self._updated_at = utc_now_iso()

    def record_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        model_name: str | None = None,
    ) -> None:
        with self._lock:
            if model_name:
                self._pricing_snapshot = ModelPricingSnapshot(
                    model_name=model_name,
                    display_name=self._pricing_snapshot.display_name,
                    input_price_per_million_usd=self._pricing_snapshot.input_price_per_million_usd,
                    output_price_per_million_usd=self._pricing_snapshot.output_price_per_million_usd,
                    source_url=self._pricing_snapshot.source_url,
                    source_status=self._pricing_snapshot.source_status,
                    loaded_at=self._pricing_snapshot.loaded_at,
                )
            self._input_tokens += max(0, int(input_tokens))
            self._output_tokens += max(0, int(output_tokens))
            self._call_count += 1
            self._updated_at = utc_now_iso()

    def snapshot(self) -> dict:
        with self._lock:
            pricing = self._pricing_snapshot
            input_cost = self._cost_for_tokens(self._input_tokens, pricing.input_price_per_million_usd)
            output_cost = self._cost_for_tokens(self._output_tokens, pricing.output_price_per_million_usd)
            total_cost = None if input_cost is None and output_cost is None else (input_cost or 0.0) + (output_cost or 0.0)

            return {
                "model_name": pricing.model_name,
                "display_name": pricing.display_name,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
                "call_count": self._call_count,
                "input_price_per_million_usd": pricing.input_price_per_million_usd,
                "output_price_per_million_usd": pricing.output_price_per_million_usd,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "source_url": pricing.source_url,
                "source_status": pricing.source_status,
                "pricing_loaded_at": pricing.loaded_at,
                "updated_at": self._updated_at,
            }

    @staticmethod
    def _cost_for_tokens(tokens: int, price_per_million_usd: float | None) -> float | None:
        if price_per_million_usd is None:
            return None
        return (max(0, tokens) / 1_000_000.0) * price_per_million_usd
