from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Chroma (persistent, local)
    chroma_path: Path = Path("chroma_db")
    chroma_collection: str = "ecg_knowledge"

    # LLM (OpenAI-compatible)
    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # Embeddings: "local" (ONNX MiniLM, no key) or "openai" (any OpenAI-compatible endpoint)
    embedding_provider: str = "local"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # ECG model
    ecg_model_path: Path = Path("models/ecg_lstm.keras")
    ecg_meta_path: Path = Path("models/ecg_lstm_meta.json")

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
