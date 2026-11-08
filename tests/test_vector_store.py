import hashlib

import numpy as np
import pytest

from app.config import Settings
from app.vector_store import VectorStore


class HashEmbedder:
    """Deterministic bag-of-words embedder so the real Chroma store can be tested offline."""

    def __init__(self, name="test:hash"):
        self.name = name

    def _vec(self, text):
        v = np.zeros(64)
        for w in text.lower().split():
            v[int(hashlib.md5(w.strip(".,?").encode()).hexdigest(), 16) % 64] += 1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_path=tmp_path / "chroma", chroma_collection="test_kb", llm_api_key="x")


def test_add_search_ranks_relevant_chunk_first(settings):
    store = VectorStore(settings, HashEmbedder())
    store.add_chunks("guide.md", ["ventricular beats have a wide qrs", "atrial beats arrive early", "heart rate is normal"])
    hits = store.search("wide qrs ventricular", top_k=2)
    assert len(hits) == 2
    assert hits[0]["text"] == "ventricular beats have a wide qrs"
    assert hits[0]["score"] > hits[1]["score"]
    assert hits[0]["source"] == "guide.md" and hits[0]["chunk_index"] == 0


def test_data_persists_across_restarts(settings):
    VectorStore(settings, HashEmbedder()).add_chunks("doc", ["persisted chunk"])
    reopened = VectorStore(settings, HashEmbedder())
    assert reopened.count() == 1
    assert reopened.search("persisted")[0]["text"] == "persisted chunk"


def test_reingest_upserts_and_source_filter_and_delete(settings):
    store = VectorStore(settings, HashEmbedder())
    store.add_chunks("a", ["alpha one", "alpha two"])
    store.add_chunks("a", ["alpha one", "alpha two"])         # same ids -> no duplicates
    store.add_chunks("b", ["beta one"])
    assert store.count() == 3
    assert {h["source"] for h in store.search("one", top_k=5, source="b")} == {"b"}
    store.delete_source("a")
    assert store.count() == 1


def test_search_empty_collection_and_large_top_k(settings):
    store = VectorStore(settings, HashEmbedder())
    assert store.search("anything") == []
    store.add_chunks("a", ["only chunk"])
    assert len(store.search("only", top_k=50)) == 1


def test_refuses_mismatched_embedding_model(settings):
    VectorStore(settings, HashEmbedder("model-a")).add_chunks("a", ["x"])
    with pytest.raises(RuntimeError, match="model-a"):
        VectorStore(settings, HashEmbedder("model-b"))
