import json
import re
import time
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

from policygpt.core.ai.base import AIRequestTooLargeError
from policygpt.observability.usage_metrics import LLMUsageTracker, estimate_text_tokens


class BedrockService:
    def __init__(
        self,
        chat_model: str,
        embedding_model: str,
        region_name: str,
        rate_limit_retries: int = 2,
        rate_limit_backoff_seconds: float = 8.0,
        usage_tracker: LLMUsageTracker | None = None,
        client: Any | None = None,
    ) -> None:
        self.client = client or boto3.client("bedrock-runtime", region_name=region_name)
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.region_name = region_name
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_backoff_seconds = max(0.0, rate_limit_backoff_seconds)
        self.usage_tracker = usage_tracker

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for text in texts:
            request_body = json.dumps({"inputText": text})
            response = self._run_with_retries(
                lambda request_body=request_body: self.client.invoke_model(
                    modelId=self.embedding_model,
                    body=request_body,
                )
            )
            payload = json.loads(response["body"].read())
            embedding = payload.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("Bedrock embedding response did not include an embedding vector.")
            vectors.append(np.array(embedding, dtype=np.float32))
        return vectors

    def llm_text(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        started_at = time.perf_counter()
        if self._uses_converse_api():
            converse_request = {
                "modelId": self.chat_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                "inferenceConfig": {"maxTokens": max_output_tokens},
            }
            if system_prompt.strip():
                converse_request["system"] = [{"text": system_prompt}]

            response = self._run_with_retries(lambda: self.client.converse(**converse_request))
            response_text = self._extract_converse_text(response)
            duration_ms = int(round((time.perf_counter() - started_at) * 1000))
            self._record_usage(
                usage=self._extract_converse_usage(response),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response_text,
                duration_ms=duration_ms,
            )
            return response_text

        native_request = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": max_output_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        response = self._run_with_retries(
            lambda: self.client.invoke_model(
                modelId=self.chat_model,
                body=json.dumps(native_request),
            )
        )
        payload = json.loads(response["body"].read())
        response_text = self._extract_chat_text(payload)
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        self._record_usage(
            usage=self._extract_payload_usage(payload),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            duration_ms=duration_ms,
        )
        return response_text

    def _uses_converse_api(self) -> bool:
        normalized_model = (self.chat_model or "").strip().lower()
        return normalized_model.startswith("anthropic.") or ".anthropic." in normalized_model

    def _run_with_retries(self, operation):
        attempts = self.rate_limit_retries + 1
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:
                if self.is_request_too_large_error(exc):
                    raise AIRequestTooLargeError(str(exc)) from exc

                if self.is_retryable_rate_limit_error(exc) and attempt < attempts - 1:
                    delay_seconds = self.rate_limit_backoff_seconds * (attempt + 1)
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue

                raise

        raise RuntimeError("Bedrock operation failed after retries.")

    @staticmethod
    def _extract_chat_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return BedrockService._strip_reasoning_content(content)
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    cleaned = BedrockService._strip_reasoning_content(item)
                    if cleaned:
                        text_parts.append(cleaned)
                elif isinstance(item, dict):
                    item_type = str(item.get("type") or item.get("content_type") or "").strip().lower()
                    if "reasoning" in item_type or "thinking" in item_type:
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        cleaned = BedrockService._strip_reasoning_content(text)
                        if cleaned:
                            text_parts.append(cleaned)
            return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
        return BedrockService._strip_reasoning_content(str(content))

    @staticmethod
    def _extract_converse_text(payload: dict[str, Any]) -> str:
        output = payload.get("output") or {}
        message = output.get("message") or {}
        content = message.get("content") or []
        if isinstance(content, str):
            return BedrockService._strip_reasoning_content(content)
        if not isinstance(content, list):
            return BedrockService._strip_reasoning_content(str(content))

        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                cleaned = BedrockService._strip_reasoning_content(item)
                if cleaned:
                    text_parts.append(cleaned)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or item.get("content_type") or "").strip().lower()
                if "reasoning" in item_type or "thinking" in item_type:
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    cleaned = BedrockService._strip_reasoning_content(text)
                    if cleaned:
                        text_parts.append(cleaned)
        return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()

    @staticmethod
    def _strip_reasoning_content(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        patterns = (
            r"<reasoning\b[^>]*>[\s\S]*?</reasoning>",
            r"<thinking\b[^>]*>[\s\S]*?</thinking>",
            r"<think\b[^>]*>[\s\S]*?</think>",
            r"<analysis\b[^>]*>[\s\S]*?</analysis>",
        )
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"</?(?:reasoning|thinking|think|analysis)\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _extract_payload_usage(payload: dict[str, Any]) -> dict[str, int]:
        usage = payload.get("usage") or {}
        return {
            "input_tokens": int(
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("inputTokens")
                or 0
            ),
            "output_tokens": int(
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("outputTokens")
                or 0
            ),
        }

    @staticmethod
    def _extract_converse_usage(payload: dict[str, Any]) -> dict[str, int]:
        usage = payload.get("usage") or {}
        return {
            "input_tokens": int(usage.get("inputTokens") or usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("outputTokens") or usage.get("output_tokens") or 0),
        }

    def _record_usage(
        self,
        *,
        usage: dict[str, int],
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        duration_ms: int = 0,
    ) -> None:
        if self.usage_tracker is None:
            return

        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)

        if input_tokens <= 0:
            input_tokens = estimate_text_tokens(system_prompt) + estimate_text_tokens(user_prompt)
        if output_tokens <= 0:
            output_tokens = estimate_text_tokens(response_text)

        self.usage_tracker.record_call(
            model_name=self.chat_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

    @staticmethod
    def is_request_too_large_error(exc: Exception) -> bool:
        message = str(exc).lower()
        if any(
            marker in message
            for marker in {
                "request too large",
                "input or output tokens must be reduced",
                "maximum context length",
                "context length exceeded",
                "too many tokens",
                "prompt is too long",
            }
        ):
            return True

        if isinstance(exc, ClientError):
            error_code = (exc.response.get("Error", {}).get("Code") or "").lower()
            if error_code == "validationexception":
                return "token" in message or "too large" in message or "too long" in message

        return False

    @staticmethod
    def is_retryable_rate_limit_error(exc: Exception) -> bool:
        if BedrockService.is_request_too_large_error(exc):
            return False

        if isinstance(exc, ClientError):
            error_code = (exc.response.get("Error", {}).get("Code") or "").lower()
            if error_code in {
                "throttlingexception",
                "toomanyrequestsexception",
                "servicequotaexceededexception",
                "modelnotreadyexception",
                "internalserverexception",
            }:
                return True
            return False

        if isinstance(exc, (ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError, BotoCoreError)):
            return True

        message = str(exc).lower()
        return any(
            marker in message
            for marker in {
                "rate limit",
                "rate exceeded",
                "throttl",
                "temporarily unavailable",
            }
        )
