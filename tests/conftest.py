import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Settings are required at import time; tests never contact these services.
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_FALLBACK_PROVIDER", "none")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

DATA_DIR = ROOT / "data" / "mitdb"
MODEL_PATH = ROOT / "models" / "ecg_lstm.keras"
META_PATH = ROOT / "models" / "ecg_lstm_meta.json"


# ---------------------------------------------------------------- synthetic ECG
def synthetic_ecg(fs: int = 360, seconds: float = 30.0, rr_s: float = 0.8, premature_every: int = 0, seed: int = 0):
    """Gaussian P-QRS-T complexes on baseline wander + noise. Returns (signal, true R-peak indices)."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    sig = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(n)

    peaks, pos, k = [], 0.5, 0
    while pos < seconds - 0.5:
        peaks.append(int(pos * fs))
        k += 1
        step = rr_s * (0.6 if premature_every and k % premature_every == 0 else 1.0)
        pos += step + rng.normal(0, 0.01)

    def wave(center_s, amp, width_s):
        return amp * np.exp(-0.5 * ((t - center_s) / width_s) ** 2)

    for p in peaks:
        c = p / fs
        sig += wave(c - 0.16, 0.15, 0.025) + wave(c, 1.2, 0.01) + wave(c - 0.025, -0.15, 0.008) + wave(c + 0.25, 0.3, 0.04)
    return sig.astype(np.float32), np.asarray(peaks)


@pytest.fixture
def ecg_signal():
    return synthetic_ecg()


# ---------------------------------------------------------------- fakes
class FakeStore:
    collection = "test_collection"

    def __init__(self):
        self.docs: dict[str, list[str]] = {}

    def add_chunks(self, source, chunks):
        self.docs[source] = list(chunks)
        return len(chunks)

    def delete_source(self, source):
        self.docs.pop(source, None)

    def search(self, query, top_k=5, source=None):
        hits = [
            {"score": 0.9, "source": s, "chunk_index": i, "text": c}
            for s, chunks in self.docs.items()
            if source in (None, s)
            for i, c in enumerate(chunks)
            if any(w in c.lower() for w in query.lower().split())
        ]
        return hits[:top_k]

    def count(self):
        return sum(len(c) for c in self.docs.values())


class FakeClassifier:
    classes = ["N", "S", "V", "F"]

    def __init__(self):
        self.calls = []

    def analyze(self, sig, fs, r_peaks=None):
        self.calls.append((len(sig), fs))
        return {"duration_s": round(len(sig) / fs, 2), "n_beats_classified": 10, "heart_rate_bpm": 75.0,
                "beat_counts": {"N": 8, "S": 0, "V": 2, "F": 0}}


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def llm_response(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeLLM:
    """Replays scripted chat.completions responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def make_tool_call():
    return _tool_call


@pytest.fixture
def services():
    """AppState wired with fakes (no network, no TensorFlow)."""
    from app.agent import Agent, ECGRegistry
    from app.config import get_settings
    from app.deps import AppState

    settings = get_settings()
    store, registry, clf = FakeStore(), ECGRegistry(), FakeClassifier()
    agent = Agent(settings, store, registry, clf)
    return AppState(settings=settings, store=store, registry=registry, classifier=clf, agent=agent)


@pytest.fixture
def client(services):
    from fastapi.testclient import TestClient

    from app.main import app

    app.state.services = services
    # Not used as a context manager, so the real lifespan (Chroma, embeddings, model loading) never runs.
    return TestClient(app)


# ---------------------------------------------------------------- real model / data
requires_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and META_PATH.exists() and (DATA_DIR / "100.dat").exists()),
    reason="trained model or MIT-BIH data missing (run notebooks 01 and 02)",
)


@pytest.fixture(scope="session")
def trained_classifier():
    from app.ecg.classifier import ECGClassifier

    return ECGClassifier(MODEL_PATH, META_PATH)
