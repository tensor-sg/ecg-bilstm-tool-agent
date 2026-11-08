"""Persistent local vector store backed by ChromaDB."""

from __future__ import annotations

import logging

import chromadb

from app.config import Settings
from app.embeddings import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings: Settings, embedder: Embedder):
        path = settings.resolve(settings.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
        self.collection = settings.chroma_collection
        self.embedder = embedder
        # Embeddings are always passed explicitly, so Chroma's own embedding function is disabled.
        self._col = self.client.get_or_create_collection(
            self.collection,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedder.name},
            embedding_function=None,
        )
        stored_model = (self._col.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedder.name:
            raise RuntimeError(
                f"Collection '{self.collection}' was built with embeddings '{stored_model}' but the app is configured "
                f"for '{embedder.name}'. Use a new CHROMA_COLLECTION or delete {path}."
            )
        logger.info("Chroma collection %s at %s (%d chunks)", self.collection, path, self._col.count())

    def add_chunks(self, source: str, chunks: list[str]) -> int:
        self._col.upsert(
            # Deterministic ids: re-ingesting the same file overwrites instead of duplicating.
            ids=[f"{source}::{i}" for i in range(len(chunks))],
            embeddings=self.embedder.embed(chunks),
            documents=chunks,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5, source: str | None = None) -> list[dict]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[self.embedder.embed_one(query)],
            n_results=min(top_k, self._col.count()),
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "score": round(1.0 - float(dist), 4),  # cosine similarity
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "text": doc,
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def delete_source(self, source: str) -> None:
        self._col.delete(where={"source": source})

    def count(self) -> int:
        return self._col.count()
