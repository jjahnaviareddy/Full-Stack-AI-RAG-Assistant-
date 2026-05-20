"""
Application configuration using Pydantic Settings.
Reads from environment variables or .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "RAG Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = Field(default="dev-secret-key-change-in-production")

    # API
    api_prefix: str = "/api/v1"
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Vector Store
    faiss_index_path: str = "./data/faiss_indexes"
    embedding_dimension: int = 1536

    # Documents
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RAG
    default_top_k: int = 4
    max_top_k: int = 10
    temperature: float = 0.1
    max_tokens: int = 1000

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        Path(self.faiss_index_path).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
