from __future__ import annotations

import threading
import uuid
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


@dataclass(frozen=True)
class UsageHistoryEntry:
    request_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None
    input_cost_inr: float | None
    output_cost_inr: float | None
    total_cost_inr: float | None
    duration_ms: int
    created_at: str = field(default_factory=utc_now_iso)


class LLMUsageTracker:
    def __init__(self, model_name: str, usd_to_inr_exchange_rate: float = 93.0) -> None:
        self._lock = threading.RLock()
        self._pricing_snapshot = ModelPricingSnapshot(
            model_name=model_name,
            display_name=model_name,
        )
        self._usd_to_inr_exchange_rate = max(0.0, float(usd_to_inr_exchange_rate))
        self._input_tokens = 0
        self._output_tokens = 0
        self._call_count = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_duration_ms = 0
        self._last_request_id = ""
        self._updated_at = utc_now_iso()
        self._history: list[UsageHistoryEntry] = []
        self._max_history_entries = 100

    def reset(self, model_name: str, usd_to_inr_exchange_rate: float | None = None) -> None:
        with self._lock:
            self._pricing_snapshot = ModelPricingSnapshot(
                model_name=model_name,
                display_name=model_name,
            )
            if usd_to_inr_exchange_rate is not None:
                self._usd_to_inr_exchange_rate = max(0.0, float(usd_to_inr_exchange_rate))
            self._input_tokens = 0
            self._output_tokens = 0
            self._call_count = 0
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            self._last_duration_ms = 0
            self._last_request_id = ""
            self._updated_at = utc_now_iso()
            self._history = []

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
        request_id: str | None = None,
        duration_ms: int | float | None = None,
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
            pricing = self._pricing_snapshot
            normalized_input_tokens = max(0, int(input_tokens))
            normalized_output_tokens = max(0, int(output_tokens))
            normalized_duration_ms = max(0, int(round(float(duration_ms or 0))))
            normalized_request_id = (request_id or uuid.uuid4().hex[:12]).strip() or uuid.uuid4().hex[:12]

            self._input_tokens += normalized_input_tokens
            self._output_tokens += normalized_output_tokens
            self._call_count += 1
            self._last_input_tokens = normalized_input_tokens
            self._last_output_tokens = normalized_output_tokens
            self._last_duration_ms = normalized_duration_ms
            self._last_request_id = normalized_request_id
            input_cost_usd = self._cost_for_tokens(normalized_input_tokens, pricing.input_price_per_million_usd)
            output_cost_usd = self._cost_for_tokens(normalized_output_tokens, pricing.output_price_per_million_usd)
            total_cost_usd = None if input_cost_usd is None and output_cost_usd is None else (input_cost_usd or 0.0) + (output_cost_usd or 0.0)
            self._history.append(
                UsageHistoryEntry(
                    request_id=normalized_request_id,
                    model_name=pricing.model_name,
                    input_tokens=normalized_input_tokens,
                    output_tokens=normalized_output_tokens,
                    input_cost_usd=input_cost_usd,
                    output_cost_usd=output_cost_usd,
                    total_cost_usd=total_cost_usd,
                    input_cost_inr=self._convert_to_inr(input_cost_usd),
                    output_cost_inr=self._convert_to_inr(output_cost_usd),
                    total_cost_inr=self._convert_to_inr(total_cost_usd),
                    duration_ms=normalized_duration_ms,
                )
            )
            if len(self._history) > self._max_history_entries:
                self._history = self._history[-self._max_history_entries :]
            self._updated_at = utc_now_iso()

    def snapshot(self) -> dict:
        with self._lock:
            pricing = self._pricing_snapshot
            input_cost = self._cost_for_tokens(self._input_tokens, pricing.input_price_per_million_usd)
            output_cost = self._cost_for_tokens(self._output_tokens, pricing.output_price_per_million_usd)
            total_cost = None if input_cost is None and output_cost is None else (input_cost or 0.0) + (output_cost or 0.0)
            last_input_cost = self._cost_for_tokens(self._last_input_tokens, pricing.input_price_per_million_usd)
            last_output_cost = self._cost_for_tokens(self._last_output_tokens, pricing.output_price_per_million_usd)
            last_total_cost = None if last_input_cost is None and last_output_cost is None else (last_input_cost or 0.0) + (last_output_cost or 0.0)

            return {
                "model_name": pricing.model_name,
                "display_name": pricing.display_name,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "last_input_tokens": self._last_input_tokens,
                "last_output_tokens": self._last_output_tokens,
                "last_request_id": self._last_request_id,
                "last_duration_ms": self._last_duration_ms,
                "total_tokens": self._input_tokens + self._output_tokens,
                "call_count": self._call_count,
                "input_price_per_million_usd": pricing.input_price_per_million_usd,
                "output_price_per_million_usd": pricing.output_price_per_million_usd,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "last_input_cost_usd": last_input_cost,
                "last_output_cost_usd": last_output_cost,
                "last_total_cost_usd": last_total_cost,
                "exchange_rate_usd_to_inr": self._usd_to_inr_exchange_rate,
                "input_cost_inr": self._convert_to_inr(input_cost),
                "output_cost_inr": self._convert_to_inr(output_cost),
                "total_cost_inr": self._convert_to_inr(total_cost),
                "last_input_cost_inr": self._convert_to_inr(last_input_cost),
                "last_output_cost_inr": self._convert_to_inr(last_output_cost),
                "last_total_cost_inr": self._convert_to_inr(last_total_cost),
                "history": [
                    {
                        "request_id": entry.request_id,
                        "model_name": entry.model_name,
                        "input_tokens": entry.input_tokens,
                        "output_tokens": entry.output_tokens,
                        "input_cost_usd": entry.input_cost_usd,
                        "output_cost_usd": entry.output_cost_usd,
                        "total_cost_usd": entry.total_cost_usd,
                        "input_cost_inr": entry.input_cost_inr,
                        "output_cost_inr": entry.output_cost_inr,
                        "total_cost_inr": entry.total_cost_inr,
                        "duration_ms": entry.duration_ms,
                        "created_at": entry.created_at,
                    }
                    for entry in reversed(self._history)
                ],
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

    def _convert_to_inr(self, usd_value: float | None) -> float | None:
        if usd_value is None:
            return None
        return usd_value * self._usd_to_inr_exchange_rate
