import numpy as np
from openai import OpenAI


class OpenAIService:
    def __init__(self, chat_model: str, embedding_model: str):
        self.client = OpenAI()
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def llm_text(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_output_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
