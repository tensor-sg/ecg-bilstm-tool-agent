from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Providers reachable through the OpenAI Python client (all speak the Chat Completions API with tool calling).
PROVIDERS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key": "openai_api_key"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash", "key": "gemini_api_key"},
    "custom": {"base_url": None, "model": None, "key": "llm_api_key"},
}


@dataclass(frozen=True)
class LLMEndpoint:
    provider: str
    api_key: str
    base_url: str
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Chroma (persistent, local)
    chroma_path: Path = Path("chroma_db")
    chroma_collection: str = "ecg_knowledge"

    # LLM agent: "openai" (OPENAI_API_KEY), "gemini" (GEMINI_API_KEY, via Google's OpenAI-compatible endpoint)
    # or "custom" (LLM_API_KEY + LLM_BASE_URL, e.g. Ollama, vLLM, LM Studio).
    llm_provider: str = "openai"
    llm_model: str | None = None                 # default per provider: gpt-4o-mini / gemini-2.5-flash
    llm_fallback_provider: str | None = None     # used when the main provider fails, e.g. "gemini"
    llm_fallback_model: str | None = None
    llm_temperature: float | None = None         # omitted from requests when unset
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    # Embeddings: "local" (ONNX MiniLM, no key) or "openai" (any OpenAI-compatible endpoint)
    embedding_provider: str = "local"
    embedding_api_key: str | None = None         # falls back to OPENAI_API_KEY
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

    def llm_endpoint(self, provider: str | None = None, model: str | None = None) -> LLMEndpoint:
        """Credentials, base URL and model for a provider (default: LLM_PROVIDER)."""
        provider = (provider or self.llm_provider).lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{provider}' (use one of {', '.join(PROVIDERS)})")
        spec = PROVIDERS[provider]
        key = getattr(self, spec["key"])
        base_url = (self.llm_base_url if provider == "custom" else None) or spec["base_url"]
        model = model or (self.llm_model if provider == self.llm_provider.lower() else None) or spec["model"]
        missing = [name for name, v in ((spec["key"].upper(), key), ("LLM_BASE_URL", base_url), ("LLM_MODEL", model)) if not v]
        if missing:
            raise ValueError(f"LLM provider '{provider}' needs {', '.join(missing)} in .env")
        return LLMEndpoint(provider, key, base_url, model)

    def llm_fallback_endpoint(self) -> LLMEndpoint | None:
        if not self.llm_fallback_provider or self.llm_fallback_provider.lower() == "none":
            return None
        return self.llm_endpoint(self.llm_fallback_provider, self.llm_fallback_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
