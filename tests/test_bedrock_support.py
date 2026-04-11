import json
import unittest

from policygpt.config import Config
from policygpt.core.ai.providers.bedrock_provider import BedrockService
from policygpt.observability.usage_metrics import LLMUsageTracker


class _FakeStreamingBody:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload


class _FakeBedrockClient:
    def __init__(self, *, invoke_response: dict | None = None, converse_response: dict | None = None) -> None:
        self.invoke_response = invoke_response or {}
        self.converse_response = converse_response or {}
        self.invoke_calls: list[dict] = []
        self.converse_calls: list[dict] = []

    def invoke_model(self, **kwargs):
        self.invoke_calls.append(kwargs)
        return {"body": _FakeStreamingBody(self.invoke_response)}

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        return self.converse_response


class ConfigProfileTests(unittest.TestCase):
    def test_claude_bedrock_profiles_resolve_to_expected_models(self) -> None:
        cases = {
            "bedrock-claude-sonnet-4-6": "global.anthropic.claude-sonnet-4-6",
            "bedrock-claude-opus-4-6": "global.anthropic.claude-opus-4-6-v1",
        }

        for ai_profile, expected_model in cases.items():
            with self.subTest(ai_profile=ai_profile):
                config = Config(ai_profile=ai_profile)
                self.assertEqual(config.ai_provider, "bedrock")
                self.assertEqual(config.chat_model, expected_model)
                self.assertEqual(config.embedding_model, "amazon.titan-embed-text-v2:0")


class BedrockServiceRoutingTests(unittest.TestCase):
    def test_gpt_oss_profiles_continue_using_invoke_model(self) -> None:
        client = _FakeBedrockClient(
            invoke_response={
                "choices": [{"message": {"content": "invoke-model-answer"}}],
                "usage": {"prompt_tokens": 31, "completion_tokens": 9},
            }
        )
        tracker = LLMUsageTracker("openai.gpt-oss-20b-1:0")
        service = BedrockService(
            chat_model="openai.gpt-oss-20b-1:0",
            embedding_model="amazon.titan-embed-text-v2:0",
            region_name="ap-south-1",
            usage_tracker=tracker,
            client=client,
        )

        response_text = service.llm_text(
            system_prompt="You are helpful.",
            user_prompt="Hello",
            max_output_tokens=123,
        )

        self.assertEqual(response_text, "invoke-model-answer")
        self.assertEqual(len(client.invoke_calls), 1)
        self.assertEqual(len(client.converse_calls), 0)

        request_body = json.loads(client.invoke_calls[0]["body"])
        self.assertEqual(client.invoke_calls[0]["modelId"], "openai.gpt-oss-20b-1:0")
        self.assertEqual(request_body["max_completion_tokens"], 123)
        self.assertEqual(request_body["messages"][0]["role"], "system")
        self.assertEqual(request_body["messages"][1]["role"], "user")
        self.assertEqual(tracker.snapshot()["input_tokens"], 31)
        self.assertEqual(tracker.snapshot()["output_tokens"], 9)

    def test_gpt_oss_reasoning_tags_are_removed_from_visible_answer(self) -> None:
        client = _FakeBedrockClient(
            invoke_response={
                "choices": [
                    {
                        "message": {
                            "content": "<reasoning>internal chain of thought</reasoning>\nFinal visible answer"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 7},
            }
        )
        service = BedrockService(
            chat_model="openai.gpt-oss-20b-1:0",
            embedding_model="amazon.titan-embed-text-v2:0",
            region_name="ap-south-1",
            client=client,
        )

        response_text = service.llm_text(
            system_prompt="You are helpful.",
            user_prompt="Hello",
            max_output_tokens=100,
        )

        self.assertEqual(response_text, "Final visible answer")

    def test_claude_profiles_use_converse(self) -> None:
        client = _FakeBedrockClient(
            converse_response={
                "usage": {
                    "inputTokens": 44,
                    "outputTokens": 12,
                },
                "output": {
                    "message": {
                        "content": [
                            {"text": "claude-answer"},
                        ]
                    }
                }
            }
        )
        tracker = LLMUsageTracker("global.anthropic.claude-sonnet-4-6")
        service = BedrockService(
            chat_model="global.anthropic.claude-sonnet-4-6",
            embedding_model="amazon.titan-embed-text-v2:0",
            region_name="ap-south-1",
            usage_tracker=tracker,
            client=client,
        )

        response_text = service.llm_text(
            system_prompt="You are helpful.",
            user_prompt="Hello",
            max_output_tokens=456,
        )

        self.assertEqual(response_text, "claude-answer")
        self.assertEqual(len(client.invoke_calls), 0)
        self.assertEqual(len(client.converse_calls), 1)

        request = client.converse_calls[0]
        self.assertEqual(request["modelId"], "global.anthropic.claude-sonnet-4-6")
        self.assertEqual(request["system"], [{"text": "You are helpful."}])
        self.assertEqual(request["messages"], [{"role": "user", "content": [{"text": "Hello"}]}])
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 456})
        self.assertEqual(tracker.snapshot()["input_tokens"], 44)
        self.assertEqual(tracker.snapshot()["output_tokens"], 12)

    def test_converse_reasoning_blocks_are_filtered_out(self) -> None:
        client = _FakeBedrockClient(
            converse_response={
                "usage": {
                    "inputTokens": 44,
                    "outputTokens": 12,
                },
                "output": {
                    "message": {
                        "content": [
                            {"type": "reasoning", "text": "hidden thoughts"},
                            {"type": "text", "text": "Visible answer"},
                        ]
                    }
                }
            }
        )
        service = BedrockService(
            chat_model="global.anthropic.claude-sonnet-4-6",
            embedding_model="amazon.titan-embed-text-v2:0",
            region_name="ap-south-1",
            client=client,
        )

        response_text = service.llm_text(
            system_prompt="You are helpful.",
            user_prompt="Hello",
            max_output_tokens=456,
        )

        self.assertEqual(response_text, "Visible answer")


if __name__ == "__main__":
    unittest.main()
