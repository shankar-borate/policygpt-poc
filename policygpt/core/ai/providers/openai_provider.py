import re
import time

import numpy as np
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from policygpt.core.ai.base import AIRequestTooLargeError
from policygpt.observability.usage_metrics import LLMUsageTracker, estimate_text_tokens


class OpenAIService:
    def __init__(
        self,
        chat_model: str,
        embedding_model: str,
        rate_limit_retries: int = 2,
        rate_limit_backoff_seconds: float = 8.0,
        usage_tracker: LLMUsageTracker | None = None,
        client=None,
    ):
        self.client = client or OpenAI()
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_backoff_seconds = max(0.0, rate_limit_backoff_seconds)
        self.usage_tracker = usage_tracker

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        response = self._run_with_retries(
            lambda: self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def llm_text(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        started_at = time.perf_counter()
        response = self._run_with_retries(
            lambda: self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_output_tokens,
            )
        )
        content = response.choices[0].message.content
        response_text = self._strip_reasoning_content((content or "").strip())
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        self._record_usage(
            response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            duration_ms=duration_ms,
        )
        return response_text

    def _record_usage(self, response, *, system_prompt: str, user_prompt: str, response_text: str, duration_ms: int = 0) -> None:
        if self.usage_tracker is None:
            return

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None

        if input_tokens is None:
            input_tokens = estimate_text_tokens(system_prompt) + estimate_text_tokens(user_prompt)
        if output_tokens is None:
            output_tokens = estimate_text_tokens(response_text)

        self.usage_tracker.record_call(
            model_name=self.chat_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

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

        raise RuntimeError("OpenAI operation failed after retries.")

    @staticmethod
    def is_request_too_large_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in {
                "request too large",
                "input or output tokens must be reduced",
                "maximum context length",
                "context length exceeded",
                "too many tokens",
            }
        )

    @staticmethod
    def is_retryable_rate_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        if OpenAIService.is_request_too_large_error(exc):
            return False

        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True

        return "rate limit" in message or "temporarily unavailable" in message


# Backward-compat alias
OpenAIRequestTooLargeError = AIRequestTooLargeError
