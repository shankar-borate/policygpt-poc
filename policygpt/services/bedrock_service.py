import json
import time
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

from policygpt.services.base import AIRequestTooLargeError


class BedrockService:
    def __init__(
        self,
        chat_model: str,
        embedding_model: str,
        region_name: str,
        rate_limit_retries: int = 2,
        rate_limit_backoff_seconds: float = 8.0,
        client: Any | None = None,
    ) -> None:
        self.client = client or boto3.client("bedrock-runtime", region_name=region_name)
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.region_name = region_name
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_backoff_seconds = max(0.0, rate_limit_backoff_seconds)

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
            return self._extract_converse_text(response)

        native_request = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": max_output_tokens,
            "stream": False,
        }
        response = self._run_with_retries(
            lambda: self.client.invoke_model(
                modelId=self.chat_model,
                body=json.dumps(native_request),
            )
        )
        payload = json.loads(response["body"].read())
        return self._extract_chat_text(payload)

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
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()
        return str(content).strip()

    @staticmethod
    def _extract_converse_text(payload: dict[str, Any]) -> str:
        output = payload.get("output") or {}
        message = output.get("message") or {}
        content = message.get("content") or []
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip()

        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()

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
