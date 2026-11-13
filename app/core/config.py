from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str
    qdrant_url: str
    redis_url: str
    database_url: str
    session_ttl_seconds: int
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_embedding_deployment: str


def get_settings():
    return Settings()
