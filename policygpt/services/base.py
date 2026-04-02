from typing import Protocol

import numpy as np


class AIRequestTooLargeError(RuntimeError):
    pass


class AIService(Protocol):
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def llm_text(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        ...
