from urllib.parse import urlparse

from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.config import settings


def _build_openai_client(
    endpoint: str,
    api_key: str,
    api_version: str | None = None,
) -> OpenAI:
    cleaned_endpoint = endpoint.rstrip("/")

    # Accept full endpoint values copied from Foundry like:
    #   https://<resource>.services.ai.azure.com/openai/v1/responses
    # and normalize them to resource root before building base_url.
    marker = "/openai/v1"
    lowered = cleaned_endpoint.lower()
    marker_index = lowered.find(marker)
    if marker_index != -1:
        cleaned_endpoint = cleaned_endpoint[:marker_index]

    host = (urlparse(cleaned_endpoint).hostname or "").lower()
    is_foundry_services_endpoint = host.endswith("services.ai.azure.com")

    kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": f"{cleaned_endpoint}/openai/v1/",
    }
    if api_version and not is_foundry_services_endpoint:
        kwargs["default_query"] = {"api-version": api_version}
    return OpenAI(**kwargs)

embedding_client = _build_openai_client(
    endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key.get_secret_value(),
    api_version=settings.openai_api_version,
)

_chat_endpoint = settings.azure_openai_chat_endpoint
_chat_api_key = (
    settings.azure_openai_chat_api_key.get_secret_value()
    if settings.azure_openai_chat_api_key
    else None
)
_chat_api_version = settings.azure_openai_chat_api_version or settings.openai_api_version

if _chat_api_key:
    _resolved_chat_endpoint = _chat_endpoint or settings.azure_openai_endpoint
    _resolved_chat_api_key = _chat_api_key
else:
    _resolved_chat_endpoint = settings.azure_openai_endpoint
    _resolved_chat_api_key = settings.azure_openai_api_key.get_secret_value()

chat_client = _build_openai_client(
    endpoint=_resolved_chat_endpoint,
    api_key=_resolved_chat_api_key,
    api_version=_chat_api_version,
)

# Backward compatibility for any modules importing `client`.
client = chat_client

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key.get_secret_value()
    if settings.qdrant_api_key
    else None,
    timeout=settings.qdrant_timeout_seconds,
)
