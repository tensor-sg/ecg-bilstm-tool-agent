"""Integration tests: the real FastAPI app with its real lifespan, i.e. the trained LSTM, a persistent
Chroma store on disk and local MiniLM embeddings. Only the LLM is scripted, since it is an external
paid service. Uses DS2 test patients only.
"""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ecg import mitbih
from app.ecg.preprocessing import SYMBOL_TO_AAMI
from tests.conftest import DATA_DIR, ROOT, FakeLLM, llm_response, requires_model

pytestmark = [pytest.mark.model, pytest.mark.integration, requires_model]

REFERENCE_DOC = ROOT / "docs" / "ecg_reference.md"


@pytest.fixture(scope="module")
def chroma_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("chroma")


@pytest.fixture(scope="module")
def live(chroma_dir):
    """The app as `uvicorn app.main:app` starts it, with Chroma redirected to a temp folder."""
    import app.main as main

    original = main.settings
    main.settings = original.model_copy(update={"chroma_path": chroma_dir, "chroma_collection": "it_kb"})
    try:
        with TestClient(main.app) as client:      # runs the lifespan: loads embeddings, Chroma and the model
            yield client
    finally:
        main.settings = original


def record_strip(record: str, seconds: int, fs_out: int | None = None):
    sig, samples, symbols, fs = mitbih.load_record(DATA_DIR, record)
    n = int(seconds * fs)
    labels = [SYMBOL_TO_AAMI[s] for s, p in zip(symbols, samples) if p < n and s in SYMBOL_TO_AAMI]
    sig = sig[:n]
    if fs_out:
        from scipy.signal import resample

        sig, fs = resample(sig, int(len(sig) * fs_out / fs)), fs_out
    return sig, fs, {c: labels.count(c) for c in "NSVF"}


# ---------------------------------------------------------------- startup
def test_startup_loads_real_model_and_store(live):
    body = live.get("/health").json()
    assert {k: body[k] for k in ("status", "vector_collection", "ecg_model_loaded")} == \
        {"status": "ok", "vector_collection": "it_kb", "ecg_model_loaded": True}


# ---------------------------------------------------------------- RAG with real embeddings
def test_ingest_reference_document_and_retrieve_relevant_chunks(live):
    r = live.post("/documents/upload", files={"file": ("ecg_reference.md", REFERENCE_DOC.read_bytes(), "text/markdown")})
    assert r.status_code == 200 and r.json()["chunks"] >= 3
    assert live.get("/documents/count").json()["points"] == r.json()["chunks"]

    hits = live.post("/documents/search", json={"query": "What is a premature ventricular contraction?", "top_k": 3}).json()
    assert hits[0]["source"] == "ecg_reference.md"
    assert any("ventricular" in h["text"].lower() for h in hits)
    assert all(0 < h["score"] <= 1 for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_semantic_search_prefers_matching_document(live):
    live.post("/documents/text", json={"source": "cooking.txt", "text": "Knead the dough, let it rise, then bake the bread at 220 C."})
    hit = live.post("/documents/search", json={"query": "irregular heartbeat originating in the atria", "top_k": 1}).json()[0]
    assert hit["source"] == "ecg_reference.md"
    hit = live.post("/documents/search", json={"query": "how long to bake bread", "top_k": 1}).json()[0]
    assert hit["source"] == "cooking.txt"
    live.delete("/documents/cooking.txt")


def test_knowledge_base_persists_across_restart(live, chroma_dir):
    count = live.get("/documents/count").json()["points"]
    assert count > 0
    import app.main as main

    with TestClient(main.app) as restarted:      # second lifespan, same folder on disk
        assert restarted.get("/documents/count").json()["points"] == count
        assert restarted.post("/documents/search", json={"query": "ventricular", "top_k": 1}).json()


# ---------------------------------------------------------------- ECG endpoints with the real model
def test_csv_upload_at_250hz_matches_cardiologist_labels(live):
    """Record 233, first 2 min, resampled to 250 Hz and uploaded as CSV, as a device export would be."""
    sig, fs, truth = record_strip("233", 120, fs_out=250)
    csv = "mlii\n" + "\n".join(f"{v:.4f}" for v in sig)
    r = live.post("/ecg/upload-csv", files={"file": ("233.csv", csv.encode(), "text/csv")}, data={"fs": "250"})
    assert r.status_code == 200 and r.json()["duration_s"] == pytest.approx(120, abs=0.1)

    res = live.post(f"/ecg/{r.json()['ecg_id']}/analyze").json()
    total = sum(truth.values())
    assert abs(res["n_beats_classified"] - total) <= 0.03 * total
    assert abs(res["beat_counts"]["V"] - truth["V"]) <= 0.15 * truth["V"]
    assert set(res["beat_counts"]) == {"N", "S", "V"}
    assert all(b["label"] in ("S", "V") for b in res["abnormal_beats"])
    assert 0 < res["heart_rate_bpm"] < 200


def test_json_upload_with_supplied_r_peaks(live):
    sig, samples, symbols, fs = mitbih.load_record(DATA_DIR, "100")
    n = int(60 * fs)
    peaks = [int(p) for p, s in zip(samples, symbols) if p < n and s in SYMBOL_TO_AAMI]
    ecg_id = live.post("/ecg/upload", json={"signal": sig[:n].tolist(), "fs": fs, "name": "rec100", "r_peaks": peaks}).json()["ecg_id"]
    res = live.post(f"/ecg/{ecg_id}/analyze").json()
    assert abs(res["n_beats_classified"] - len(peaks)) <= 2        # only edge beats may be skipped
    assert res["beat_percentages"]["N"] >= 95
    assert any(r["name"] == "rec100" for r in live.get("/ecg/").json())


def test_analyze_response_contract(live):
    sig, fs, _ = record_strip("233", 30)
    ecg_id = live.post("/ecg/upload", json={"signal": sig.tolist(), "fs": fs}).json()["ecg_id"]
    res = live.post(f"/ecg/{ecg_id}/analyze").json()
    for key in ("duration_s", "n_beats_classified", "heart_rate_bpm", "rr_sdnn_ms", "beat_counts", "beat_percentages",
                "class_descriptions", "abnormal_beats", "signal_quality", "disclaimer"):
        assert key in res, key
    assert sum(res["beat_counts"].values()) == res["n_beats_classified"]
    assert sum(res["beat_percentages"].values()) == pytest.approx(100, abs=0.5)
    for b in res["abnormal_beats"]:
        assert 0 <= b["time_s"] <= res["duration_s"] and 0 < b["confidence"] <= 1


def test_analyze_latency(live):
    """Upload + analysis of a 5-minute strip over HTTP (measured ~0.65 s on a 2-core laptop)."""
    sig, fs, _ = record_strip("233", 300)
    payload = {"signal": sig.tolist(), "fs": fs}
    live.post(f"/ecg/{live.post('/ecg/upload', json=payload).json()['ecg_id']}/analyze")     # warm-up
    t = time.perf_counter()
    ecg_id = live.post("/ecg/upload", json=payload).json()["ecg_id"]
    assert live.post(f"/ecg/{ecg_id}/analyze").status_code == 200
    assert time.perf_counter() - t < 5


def test_noise_only_input_is_flagged_as_unreliable(live):
    """Pure noise still produces detector "beats" (found by this test); the response must flag it."""
    noise = np.random.default_rng(0).normal(0, 0.01, 360 * 20)
    ecg_id = live.post("/ecg/upload", json={"signal": noise.tolist(), "fs": 360}).json()["ecg_id"]
    r = live.post(f"/ecg/{ecg_id}/analyze")
    assert r.status_code == 200
    q = r.json()["signal_quality"]
    assert q["ok"] is False and q["warning"]


def test_real_ecg_passes_signal_quality(live):
    sig, fs, _ = record_strip("233", 60)
    ecg_id = live.post("/ecg/upload", json={"signal": sig.tolist(), "fs": fs}).json()["ecg_id"]
    q = live.post(f"/ecg/{ecg_id}/analyze").json()["signal_quality"]
    assert q["ok"] is True and q["warning"] is None


# ---------------------------------------------------------------- agent: LLM scripted, everything else real
def test_agent_uses_real_model_and_knowledge_base(live):
    sig, fs, truth = record_strip("233", 60)
    ecg_id = live.post("/ecg/upload", json={"signal": sig.tolist(), "fs": fs}).json()["ecg_id"]
    agent = live.app.state.services.agent
    original = agent.client
    agent.client = FakeLLM([
        llm_response(tool_calls=[
            _call("c1", "analyze_ecg", {"ecg_id": ecg_id}),
            _call("c2", "search_knowledge_base", {"query": "premature ventricular contraction", "top_k": 2}),
        ]),
        llm_response(content="Frequent PVCs."),
    ])
    try:
        body = live.post("/chat", json={"message": "Is this ECG abnormal?", "ecg_id": ecg_id}).json()
    finally:
        fake, agent.client = agent.client, original

    assert body["answer"] == "Frequent PVCs."
    tools = {t["tool"]: t for t in body["tool_calls"]}
    assert set(tools) == {"analyze_ecg", "search_knowledge_base"}
    assert body["sources"] and body["sources"][0]["source"] == "ecg_reference.md"

    # The real model's output is what the LLM sees in its second request.
    tool_msgs = [m for m in fake.requests[1]["messages"] if m.get("role") == "tool"]
    ecg_result = json.loads(next(m["content"] for m in tool_msgs if m["tool_call_id"] == "c1"))
    assert abs(ecg_result["beat_counts"]["V"] - truth["V"]) <= 0.15 * truth["V"]
    user_msg = next(m for m in fake.requests[0]["messages"] if m["role"] == "user")
    assert ecg_id in user_msg["content"]                                   # the LLM is told which ECG is attached


def _call(call_id, name, args):
    from tests.conftest import _tool_call

    return _tool_call(call_id, name, json.dumps(args))
