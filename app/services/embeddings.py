from functools import lru_cache
from collections.abc import Sequence

from app.core.client import embedding_client
from app.core.config import settings
from app.services.embedding import AzureOpenAIEmbedder


@lru_cache
def _embedder() -> AzureOpenAIEmbedder:
    return AzureOpenAIEmbedder(
        client=embedding_client,
        model=settings.azure_openai_embedding_deployment,
        dimension=settings.embedding_dimension,
    )


def embed_text(text: str) -> list[float]:
    return _embedder().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return _embedder().embed_texts(texts)
