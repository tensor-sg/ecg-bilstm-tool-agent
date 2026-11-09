"""Text embeddings: local ONNX MiniLM (default, no API key) or any OpenAI-compatible endpoint."""

from __future__ import annotations

from app.config import Settings


class Embedder:
    def __init__(self, settings: Settings):
        self.provider = settings.embedding_provider
        if self.provider == "local":
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            # all-MiniLM-L6-v2 via onnxruntime; downloaded once (~80 MB) to ~/.cache/chroma.
            self._local = DefaultEmbeddingFunction()
            self.name = "local:all-MiniLM-L6-v2"
        elif self.provider == "openai":
            from openai import OpenAI

            api_key = settings.embedding_api_key or settings.openai_api_key
            if not api_key:
                raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
            self.client = OpenAI(api_key=api_key, base_url=settings.embedding_base_url)
            self.model = settings.embedding_model
            self.name = f"openai:{self.model}"
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER '{self.provider}' (use 'local' or 'openai')")

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self.provider == "local":
                vectors.extend([float(x) for x in v] for v in self._local(batch))
            else:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                vectors.extend(d.embedding for d in resp.data)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
