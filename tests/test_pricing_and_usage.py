import unittest
from unittest.mock import patch

from policygpt.core.ai.providers.openai_provider import OpenAIService
from policygpt.observability.pricing.pricing_loader import ModelPricingLoader
from policygpt.observability.usage_metrics import LLMUsageTracker, ModelPricingSnapshot


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeOpenAIMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeOpenAIMessage(content)


class _FakeOpenAIResponse:
    def __init__(self, *, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_FakeOpenAIChoice(content)]
        self.usage = _FakeOpenAIUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _FakeOpenAICompletions:
    def __init__(self, response: _FakeOpenAIResponse) -> None:
        self.response = response

    def create(self, **_kwargs):
        return self.response


class _FakeOpenAIChat:
    def __init__(self, response: _FakeOpenAIResponse) -> None:
        self.completions = _FakeOpenAICompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response: _FakeOpenAIResponse) -> None:
        self.chat = _FakeOpenAIChat(response)


class PricingLoaderTests(unittest.TestCase):
    def test_parse_openai_model_pricing(self) -> None:
        html = """
        <div class="pricing-grid">
            <div>Input</div><div class="text-2xl font-semibold">$2.00</div>
            <div>Cached input</div><div class="text-2xl font-semibold">$0.50</div>
            <div>Output</div><div class="text-2xl font-semibold">$8.00</div>
        </div>
        """

        input_price, output_price = ModelPricingLoader._parse_openai_model_pricing(html)

        self.assertEqual(input_price, 2.0)
        self.assertEqual(output_price, 8.0)

    def test_gpt_5_4_fallback_pricing_exists(self) -> None:
        snapshot = ModelPricingLoader._fallback_snapshot("gpt-5.4")

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.model_name, "gpt-5.4")
        self.assertEqual(snapshot.display_name, "OpenAI GPT-5.4")
        self.assertEqual(snapshot.input_price_per_million_usd, 2.5)
        self.assertEqual(snapshot.output_price_per_million_usd, 15.0)

    def test_load_snapshot_supports_gpt_5_4_live_pricing(self) -> None:
        from policygpt.config import Config

        html = """
        <div class="pricing-grid">
            <div>Input</div><div class="text-2xl font-semibold">$2.50</div>
            <div>Cached input</div><div class="text-2xl font-semibold">$0.25</div>
            <div>Output</div><div class="text-2xl font-semibold">$15.00</div>
        </div>
        """
        loader = ModelPricingLoader()

        with patch.object(ModelPricingLoader, "_fetch_text", return_value=html):
            snapshot = loader.load_snapshot(Config(chat_model="gpt-5.4"))

        self.assertEqual(snapshot.model_name, "gpt-5.4")
        self.assertEqual(snapshot.display_name, "OpenAI GPT-5.4")
        self.assertEqual(snapshot.input_price_per_million_usd, 2.5)
        self.assertEqual(snapshot.output_price_per_million_usd, 15.0)
        self.assertEqual(snapshot.source_status, "live")

    def test_load_snapshot_labels_bedrock_gpt_oss_120b_correctly(self) -> None:
        from policygpt.config import Config

        offer_index = {
            "products": {
                "sku-input": {
                    "attributes": {
                        "regionCode": "ap-south-1",
                        "model": "gpt-oss-120b",
                        "inferenceType": "Input tokens",
                        "feature": "On-demand Inference",
                    }
                },
                "sku-output": {
                    "attributes": {
                        "regionCode": "ap-south-1",
                        "model": "gpt-oss-120b",
                        "inferenceType": "Output tokens",
                        "feature": "On-demand Inference",
                    }
                },
            },
            "terms": {
                "OnDemand": {
                    "sku-input": {
                        "term-1": {
                            "priceDimensions": {
                                "dim-1": {
                                    "description": "$0.00018 per 1K input tokens",
                                    "pricePerUnit": {"USD": "0.0001800000"},
                                }
                            }
                        }
                    },
                    "sku-output": {
                        "term-1": {
                            "priceDimensions": {
                                "dim-1": {
                                    "description": "$0.00071 per 1K output tokens",
                                    "pricePerUnit": {"USD": "0.0007100000"},
                                }
                            }
                        }
                    },
                }
            },
        }
        loader = ModelPricingLoader()

        with patch.object(ModelPricingLoader, "_fetch_json", return_value=offer_index):
            snapshot = loader.load_snapshot(Config(chat_model="openai.gpt-oss-120b-1:0"))

        self.assertEqual(snapshot.model_name, "openai.gpt-oss-120b-1:0")
        self.assertEqual(snapshot.display_name, "OpenAI GPT OSS 120B via Bedrock")
        self.assertAlmostEqual(snapshot.input_price_per_million_usd, 0.18)
        self.assertAlmostEqual(snapshot.output_price_per_million_usd, 0.71)

    def test_extract_bedrock_offer_price_normalizes_thousand_token_prices(self) -> None:
        offer_index = {
            "products": {
                "sku-input": {
                    "attributes": {
                        "regionCode": "ap-south-1",
                        "model": "gpt-oss-20b",
                        "inferenceType": "Input tokens",
                        "feature": "On-demand Inference",
                    }
                }
            },
            "terms": {
                "OnDemand": {
                    "sku-input": {
                        "term-1": {
                            "priceDimensions": {
                                "dim-1": {
                                    "description": "$0.00008 per 1K input tokens for gpt-oss-20b in Asia Pacific (Mumbai)",
                                    "pricePerUnit": {"USD": "0.0000800000"},
                                }
                            }
                        }
                    }
                }
            },
        }

        price = ModelPricingLoader._extract_bedrock_offer_price(
            offer_index=offer_index,
            region_code="ap-south-1",
            matcher=lambda attrs: attrs.get("model") == "gpt-oss-20b",
        )

        self.assertAlmostEqual(price, 0.08)


class UsageTrackerTests(unittest.TestCase):
    def test_usage_tracker_accumulates_tokens_and_cost(self) -> None:
        tracker = LLMUsageTracker("gpt-4.1", usd_to_inr_exchange_rate=93.0)
        tracker.set_pricing_snapshot(
            ModelPricingSnapshot(
                model_name="gpt-4.1",
                display_name="OpenAI GPT-4.1",
                input_price_per_million_usd=2.0,
                output_price_per_million_usd=8.0,
                source_status="live",
            )
        )
        tracker.record_call(model_name="gpt-4.1", input_tokens=1200, output_tokens=300)

        payload = tracker.snapshot()

        self.assertEqual(payload["input_tokens"], 1200)
        self.assertEqual(payload["output_tokens"], 300)
        self.assertEqual(payload["last_input_tokens"], 1200)
        self.assertEqual(payload["last_output_tokens"], 300)
        self.assertAlmostEqual(payload["input_cost_usd"], 0.0024)
        self.assertAlmostEqual(payload["output_cost_usd"], 0.0024)
        self.assertAlmostEqual(payload["total_cost_usd"], 0.0048)
        self.assertAlmostEqual(payload["input_cost_inr"], 0.2232)
        self.assertAlmostEqual(payload["output_cost_inr"], 0.2232)
        self.assertAlmostEqual(payload["total_cost_inr"], 0.4464)
        self.assertAlmostEqual(payload["last_input_cost_inr"], 0.2232)
        self.assertAlmostEqual(payload["last_output_cost_inr"], 0.2232)
        self.assertAlmostEqual(payload["last_total_cost_inr"], 0.4464)

    def test_usage_tracker_records_history_with_request_id_and_duration(self) -> None:
        tracker = LLMUsageTracker("gpt-4.1", usd_to_inr_exchange_rate=100.0)
        tracker.set_pricing_snapshot(
            ModelPricingSnapshot(
                model_name="gpt-4.1",
                display_name="OpenAI GPT-4.1",
                input_price_per_million_usd=2.0,
                output_price_per_million_usd=8.0,
                source_status="live",
            )
        )
        tracker.record_call(
            model_name="gpt-4.1",
            input_tokens=1000,
            output_tokens=250,
            request_id="abc123xyz789",
            duration_ms=1450,
        )

        payload = tracker.snapshot()

        self.assertEqual(payload["last_request_id"], "abc123xyz789")
        self.assertEqual(payload["last_duration_ms"], 1450)
        self.assertEqual(len(payload["history"]), 1)
        self.assertEqual(payload["history"][0]["request_id"], "abc123xyz789")
        self.assertEqual(payload["history"][0]["duration_ms"], 1450)
        self.assertAlmostEqual(payload["history"][0]["input_cost_inr"], 0.2)
        self.assertAlmostEqual(payload["history"][0]["output_cost_inr"], 0.2)
        self.assertAlmostEqual(payload["history"][0]["total_cost_inr"], 0.4)

    def test_usage_tracker_reset_clears_history(self) -> None:
        tracker = LLMUsageTracker("gpt-4.1")
        tracker.record_call(input_tokens=10, output_tokens=5, request_id="to-clear", duration_ms=20)

        tracker.reset("gpt-4.1")
        payload = tracker.snapshot()

        self.assertEqual(payload["history"], [])
        self.assertEqual(payload["last_request_id"], "")
        self.assertEqual(payload["last_duration_ms"], 0)

    def test_openai_service_records_usage_into_tracker(self) -> None:
        tracker = LLMUsageTracker("gpt-4.1")
        response = _FakeOpenAIResponse(content="Hello back", prompt_tokens=42, completion_tokens=11)
        service = OpenAIService(
            chat_model="gpt-4.1",
            embedding_model="text-embedding-3-large",
            usage_tracker=tracker,
            client=_FakeOpenAIClient(response),
        )

        text = service.llm_text(system_prompt="system", user_prompt="user", max_output_tokens=100)
        payload = tracker.snapshot()

        self.assertEqual(text, "Hello back")
        self.assertEqual(payload["model_name"], "gpt-4.1")
        self.assertEqual(payload["input_tokens"], 42)
        self.assertEqual(payload["output_tokens"], 11)
        self.assertEqual(len(payload["history"]), 1)
        self.assertTrue(payload["history"][0]["request_id"])

    def test_openai_service_strips_reasoning_tags_from_answer(self) -> None:
        response = _FakeOpenAIResponse(
            content="<think>internal reasoning</think>\nCustomer-facing answer",
            prompt_tokens=42,
            completion_tokens=11,
        )
        service = OpenAIService(
            chat_model="gpt-4.1",
            embedding_model="text-embedding-3-large",
            client=_FakeOpenAIClient(response),
        )

        text = service.llm_text(system_prompt="system", user_prompt="user", max_output_tokens=100)

        self.assertEqual(text, "Customer-facing answer")

    def test_config_supports_exchange_rate_override(self) -> None:
        from unittest.mock import patch
        from policygpt.config import Config

        with patch.dict("os.environ", {"POLICY_GPT_USD_TO_INR_RATE": "95.5"}, clear=False):
            config = Config.from_env()

        self.assertEqual(config.usd_to_inr_exchange_rate, 95.5)


if __name__ == "__main__":
    unittest.main()
