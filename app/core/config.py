from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str
    qdrant_url: str
    qdrant_collection: str = "documents"
    qdrant_timeout_seconds: int = 10
    embedding_dimension: int = 1536
    redis_url: str
    database_url: str
    session_ttl_seconds: int
    azure_openai_endpoint: str
    azure_openai_api_key: SecretStr
    azure_openai_embedding_deployment: str
    azure_openai_chat_endpoint: str | None = None
    azure_openai_chat_api_key: SecretStr | None = None
    azure_openai_chat_api_version: str | None = None
    openai_api_version: str | None = None
    azure_openai_chat_deployment: str = "gpt-4o"
    rag_top_k: int = 5
    chat_temperature: float = 0.2
    qdrant_api_key: SecretStr | None = None

    @field_validator(
        "qdrant_url", "azure_openai_endpoint", "azure_openai_chat_endpoint", mode="before"
    )
    @classmethod
    def normalize_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        cleaned = value.strip().strip('"').strip("'")
        return cleaned.rstrip("/") if cleaned else None

    @field_validator("azure_openai_chat_api_version", "openai_api_version", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        cleaned = value.strip().strip('"').strip("'")
        return cleaned or None

    @field_validator("qdrant_api_key", "azure_openai_chat_api_key", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        cleaned = value.strip().strip('"').strip("'")
        return cleaned or None


settings = Settings()
