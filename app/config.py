from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Support Operations API"
    app_version: str = "1.0.0"
    environment: str = "development"
    database_url: str = "sqlite:///./support.db"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "ministral-3:3b"

    api_prefix: str = "/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()