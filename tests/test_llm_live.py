"""Live end-to-end tests with real LLMs: OpenAI as the main agent, Gemini as fallback and alternative.

Nothing is faked: the real FastAPI app, trained LSTM, Chroma store, local embeddings and the provider APIs.
These tests cost API credit, so they only run when asked:

    RUN_LLM_TESTS=1 pytest -m llm

Assertions check behaviour that does not depend on wording: which tools were called with which
arguments, that cited sources exist, and that numbers in the answer match the model's output.
"""

import os
import re

import numpy as np
import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from app.ecg import mitbih
from tests.conftest import DATA_DIR, ROOT, requires_model

ENV = dotenv_values(ROOT / ".env")
pytestmark = [
    pytest.mark.llm, pytest.mark.model, requires_model,
    pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to call real LLM APIs"),
    pytest.mark.skipif(not (ENV.get("OPENAI_API_KEY") and ENV.get("GEMINI_API_KEY")), reason="OPENAI_API_KEY / GEMINI_API_KEY missing in .env"),
]


def real_settings(base, **overrides):
    """conftest replaces keys with placeholders for the offline suite; restore the real ones from .env."""
    return base.model_copy(update={
        "openai_api_key": ENV["OPENAI_API_KEY"], "gemini_api_key": ENV["GEMINI_API_KEY"],
        "llm_provider": "openai", "llm_model": "gpt-4o-mini",
        "llm_fallback_provider": "gemini", "llm_fallback_model": "gemini-2.5-flash", "llm_temperature": 0.0,
        **overrides,
    })


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    import app.main as main

    original = main.settings
    main.settings = real_settings(original, chroma_path=tmp_path_factory.mktemp("chroma"), chroma_collection="llm_kb")
    try:
        with TestClient(main.app) as client:
            doc = (ROOT / "docs" / "ecg_reference.md").read_bytes()
            assert client.post("/documents/upload", files={"file": ("ecg_reference.md", doc, "text/markdown")}).status_code == 200
            yield client
    finally:
        main.settings = original


def upload(client, record, seconds, name=None):
    sig, _, _, fs = mitbih.load_record(DATA_DIR, record)
    r = client.post("/ecg/upload", json={"signal": sig[: int(seconds * fs)].tolist(), "fs": fs, "name": name or f"mitdb_{record}"})
    return r.json()["ecg_id"]


def tools_used(body):
    return [t["tool"] for t in body["tool_calls"]]


def numbers_in(text):
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text or "")}


# ---------------------------------------------------------------- configuration
def test_health_reports_openai_main_and_gemini_fallback(live):
    assert live.get("/health").json()["llm"] == {"provider": "openai", "model": "gpt-4o-mini", "fallback": "gemini:gemini-2.5-flash"}


# ---------------------------------------------------------------- tool calling: ECG
def test_agent_analyzes_uploaded_ecg_with_the_lstm(live):
    """Record 233 (held-out patient, frequent PVCs). The answer must rest on the tool output."""
    ecg_id = upload(live, "233", 60)
    direct = live.post(f"/ecg/{ecg_id}/analyze").json()

    body = live.post("/agent/chat", json={"message": "How many ventricular beats are in this recording, and is that a lot?", "ecg_id": ecg_id}).json()

    assert body["llm"] == {"provider": "openai", "model": "gpt-4o-mini", "fallback_used": False}
    assert {"tool": "analyze_ecg", "arguments": {"ecg_id": ecg_id}} in body["tool_calls"]
    v, pct = direct["beat_counts"]["V"], direct["beat_percentages"]["V"]
    assert numbers_in(body["answer"]) & {float(v), round(pct), round(pct, 1)}, \
        f"answer does not quote the model's V count ({v}) or share ({pct}%): {body['answer']}"
    assert "diagnos" in body["answer"].lower(), "the agent must state that this is not a diagnosis"


def test_noise_recording_is_flagged_not_interpreted(live):
    noise = np.random.default_rng(0).normal(0, 0.01, 360 * 20).tolist()
    ecg_id = live.post("/ecg/upload", json={"signal": noise, "fs": 360, "name": "noise"}).json()["ecg_id"]
    body = live.post("/agent/chat", json={"message": "Analyse this ECG.", "ecg_id": ecg_id}).json()
    assert "analyze_ecg" in tools_used(body)
    assert re.search(r"nois|quality|artifact|artefact|unreliable", body["answer"], re.I), body["answer"]


def test_list_recordings_tool(live):
    upload(live, "100", 20, name="morning strip")
    upload(live, "221", 20, name="evening strip")
    body = live.post("/agent/chat", json={"message": "Which ECG recordings have been uploaded? Give their names."}).json()
    assert "list_ecg_recordings" in tools_used(body)
    assert "morning strip" in body["answer"] and "evening strip" in body["answer"]


def test_multi_turn_follow_up_uses_history(live):
    ecg_id = upload(live, "233", 60)
    v = live.post(f"/ecg/{ecg_id}/analyze").json()["beat_counts"]["V"]
    first = live.post("/agent/chat", json={"message": "Summarise this ECG in two sentences.", "ecg_id": ecg_id}).json()
    history = [{"role": "user", "content": "Summarise this ECG in two sentences."}, {"role": "assistant", "content": first["answer"]}]
    follow = live.post("/agent/chat", json={"message": "Exactly how many beats did the model label ventricular? Reply with the number.",
                                      "ecg_id": ecg_id, "history": history}).json()
    assert float(v) in numbers_in(follow["answer"]), follow["answer"]


# ---------------------------------------------------------------- RAG
def test_rag_answer_is_grounded_in_the_knowledge_base(live):
    body = live.post("/agent/chat", json={"message": "Using the knowledge base, explain what AAMI class F means and why it is hard to classify."}).json()
    assert "search_knowledge_base" in tools_used(body)
    assert body["sources"] and all(s["source"] == "ecg_reference.md" for s in body["sources"])
    assert re.search(r"fusion", body["answer"], re.I), body["answer"]


def test_combined_ecg_and_guideline_question_uses_both_tools(live):
    ecg_id = upload(live, "233", 60)
    body = live.post("/agent/chat", json={
        "message": "Analyse this recording, then use the reference documents to explain what premature ventricular contractions are.",
        "ecg_id": ecg_id}).json()
    used = tools_used(body)
    assert "analyze_ecg" in used and "search_knowledge_base" in used, used
    assert body["sources"]


# ---------------------------------------------------------------- providers
@pytest.fixture
def agent_factory(live):
    from app.agent import Agent

    services = live.app.state.services

    def make(**overrides):
        return Agent(real_settings(services.settings, **overrides), services.store, services.registry, services.classifier)

    return make


def test_gemini_as_main_agent_calls_tools(agent_factory, live):
    ecg_id = upload(live, "233", 60)
    out = agent_factory(llm_provider="gemini", llm_model=None, llm_fallback_provider="none").chat(
        "How many ventricular beats does this ECG contain?", ecg_id=ecg_id)
    assert out["llm"]["provider"] == "gemini" and not out["llm"]["fallback_used"]
    assert {"tool": "analyze_ecg", "arguments": {"ecg_id": ecg_id}} in out["tool_calls"]
    assert out["answer"]


def test_falls_back_to_gemini_when_openai_rejects_the_key(agent_factory, live):
    ecg_id = upload(live, "233", 60)
    out = agent_factory(openai_api_key="sk-invalid-key-for-test").chat("Analyse this ECG briefly.", ecg_id=ecg_id)
    assert out["llm"] == {"provider": "gemini", "model": "gemini-2.5-flash", "fallback_used": True}
    assert "analyze_ecg" in tools_used(out) and out["answer"]


# ---------------------------------------------------------------- tools API and agent trace
def test_tool_called_directly_matches_what_the_agent_received(live):
    """The /tools API and the agent run the same LSTM code path: identical beat counts."""
    ecg_id = upload(live, "233", 60)
    direct = live.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": ecg_id}}).json()["result"]
    body = live.post("/agent/chat", json={"message": "Report the exact beat counts per class for this ECG.", "ecg_id": ecg_id}).json()
    v, n = direct["beat_counts"]["V"], direct["beat_counts"]["N"]
    assert {float(v), float(n)} <= numbers_in(body["answer"]), body["answer"]


def test_agent_trace_shows_the_full_flow(live):
    ecg_id = upload(live, "233", 30)
    steps = live.post("/agent/chat", json={"message": "Is this ECG normal?", "ecg_id": ecg_id}).json()["steps"]
    kinds = [s["type"] for s in steps]
    assert kinds[0] == "llm" and steps[0]["decision"] == "call_tools" and "analyze_ecg" in steps[0]["tool_calls"]
    assert "tool" in kinds and all(s["ok"] for s in steps if s["type"] == "tool")
    assert kinds[-1] == "llm" and steps[-1]["decision"] == "answer"
    assert all(s["tokens"]["prompt"] > 0 for s in steps if s["type"] == "llm"), "token usage should be reported"
