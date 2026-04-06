import unittest

from policygpt.services.openai_service import OpenAIService
from policygpt.services.pricing_loader import ModelPricingLoader
from policygpt.services.usage_metrics import LLMUsageTracker, ModelPricingSnapshot


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
    def test_parse_openai_gpt41_pricing(self) -> None:
        html = """
        <div class="pricing-grid">
            <div>Input</div><div class="text-2xl font-semibold">$2.00</div>
            <div>Cached input</div><div class="text-2xl font-semibold">$0.50</div>
            <div>Output</div><div class="text-2xl font-semibold">$8.00</div>
        </div>
        """

        input_price, output_price = ModelPricingLoader._parse_openai_gpt41_pricing(html)

        self.assertEqual(input_price, 2.0)
        self.assertEqual(output_price, 8.0)

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
        tracker = LLMUsageTracker("gpt-4.1")
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
        self.assertAlmostEqual(payload["input_cost_usd"], 0.0024)
        self.assertAlmostEqual(payload["output_cost_usd"], 0.0024)
        self.assertAlmostEqual(payload["total_cost_usd"], 0.0048)

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


if __name__ == "__main__":
    unittest.main()
