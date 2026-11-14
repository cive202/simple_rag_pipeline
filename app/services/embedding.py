from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI


class AzureOpenAIEmbedder:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        dimension: int = 1536,
        batch_size: int = 16,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(model=self._model, input=batch)
            for item in sorted(response.data, key=lambda d: d.index):
                if len(item.embedding) != self._dimension:
                    raise ValueError(
                        f"Expected {self._dimension} dimensions, "
                        f"got {len(item.embedding)}"
                    )
                vectors.append(item.embedding)
        return vectors


if __name__ == "__main__":
    from app.core.client import embedding_client
    from app.core.config import settings

    embedder = AzureOpenAIEmbedder(
        client=embedding_client, model=settings.azure_openai_embedding_deployment
    )
    vector = embedder.embed_text("Artificial intelligence is transforming healthcare.")
    print("Dimensions:", len(vector))
    print("First 5 values:", vector[:5])
