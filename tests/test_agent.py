import json

from app.agent import TOOLS, ECGRegistry
from tests.conftest import FakeLLM, llm_response


def test_tool_schemas_are_valid():
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"search_knowledge_base", "analyze_ecg", "list_ecg_recordings"}
    for t in TOOLS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_registry_roundtrip():
    reg = ECGRegistry()
    rec = reg.add([0.0] * 720, 360, name="strip")
    assert reg.get(rec.id) is rec
    assert reg.list()[0]["duration_s"] == 2.0
    assert reg.get("missing") is None


def test_answer_without_tools(services):
    services.agent.client = FakeLLM([llm_response(content="Hello")])
    out = services.agent.chat("hi")
    steps = out.pop("steps")
    assert out == {"answer": "Hello", "tool_calls": [], "sources": [],
                   "llm": {"provider": "openai", "model": "gpt-4o-mini", "fallback_used": False}}
    assert [s["type"] for s in steps] == ["llm"] and steps[0]["decision"] == "answer"


def test_agent_runs_ecg_tool_and_feeds_result_back(services, make_tool_call):
    rec = services.registry.add([0.0] * 3600, 360, name="strip")
    fake = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": rec.id}))]),
        llm_response(content="2 ventricular beats found."),
    ])
    services.agent.client = fake

    out = services.agent.chat("analyze it", ecg_id=rec.id)

    assert out["answer"] == "2 ventricular beats found."
    assert out["tool_calls"] == [{"tool": "analyze_ecg", "arguments": {"ecg_id": rec.id}}]
    assert services.classifier.calls == [(3600, 360)]
    # the attached ecg id is given to the model, and the tool result is sent back on the second call
    assert rec.id in fake.requests[0]["messages"][-1]["content"]
    tool_msg = fake.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["beat_counts"]["V"] == 2


def test_agent_rag_tool_collects_sources(services, make_tool_call):
    services.store.add_chunks("guide.md", ["A PVC is a premature ventricular contraction."])
    services.agent.client = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "search_knowledge_base", json.dumps({"query": "pvc"}))]),
        llm_response(content="A PVC is ..."),
    ])
    out = services.agent.chat("what is a pvc")
    assert out["sources"] == [{"source": "guide.md", "chunk_index": 0, "score": 0.9}]


def test_agent_handles_unknown_ecg_and_bad_arguments(services, make_tool_call):
    fake = FakeLLM([
        llm_response(tool_calls=[
            make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": "nope"})),
            make_tool_call("c2", "list_ecg_recordings", "not json"),
            make_tool_call("c3", "does_not_exist", "{}"),
        ]),
        llm_response(content="done"),
    ])
    services.agent.client = fake
    out = services.agent.chat("go")
    assert out["answer"] == "done"
    results = [json.loads(m["content"]) for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert "Unknown ecg_id" in results[0]["error"]
    assert results[1] == {"recordings": []}
    assert "Unknown tool" in results[2]["error"]


def test_agent_without_model_reports_error(services, make_tool_call):
    services.agent.tools.classifier = None
    rec = services.registry.add([0.0] * 720, 360)
    fake = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": rec.id}))]),
        llm_response(content="model unavailable"),
    ])
    services.agent.client = fake
    services.agent.chat("go")
    assert "not loaded" in json.loads(fake.requests[1]["messages"][-1]["content"])["error"]


def test_agent_stops_after_max_steps(services, make_tool_call):
    loop = [llm_response(tool_calls=[make_tool_call(f"c{i}", "list_ecg_recordings", "{}")]) for i in range(3)]
    services.agent.client = FakeLLM(loop)
    out = services.agent.chat("loop", max_steps=3)
    assert "maximum number of tool steps" in out["answer"]
    assert len(out["tool_calls"]) == 3


# ---------------------------------------------------------------- providers and fallback
def _settings(**kw):
    from app.config import Settings

    base = dict(llm_provider="openai", openai_api_key="sk-o", gemini_api_key="g-key", llm_api_key=None,
                llm_base_url=None, llm_model=None, llm_fallback_provider=None)
    return Settings(_env_file=None, **{**base, **kw})


def test_provider_endpoints_resolve_keys_urls_and_default_models():
    s = _settings(llm_fallback_provider="gemini")
    main, fb = s.llm_endpoint(), s.llm_fallback_endpoint()
    assert (main.provider, main.api_key, main.model) == ("openai", "sk-o", "gpt-4o-mini")
    assert main.base_url == "https://api.openai.com/v1"
    assert (fb.provider, fb.api_key, fb.model) == ("gemini", "g-key", "gemini-2.5-flash")
    assert "generativelanguage.googleapis.com" in fb.base_url

    assert _settings(llm_model="gpt-4.1-mini").llm_endpoint().model == "gpt-4.1-mini"
    # LLM_MODEL belongs to the main provider; it must not leak into the fallback
    assert _settings(llm_model="gpt-4.1-mini", llm_fallback_provider="gemini").llm_fallback_endpoint().model == "gemini-2.5-flash"
    custom = _settings(llm_provider="custom", llm_api_key="k", llm_base_url="http://localhost:11434/v1", llm_model="llama3.1")
    assert (custom.llm_endpoint().base_url, custom.llm_endpoint().model) == ("http://localhost:11434/v1", "llama3.1")
    assert _settings(llm_fallback_provider="none").llm_fallback_endpoint() is None


def test_missing_provider_key_gives_clear_error():
    import pytest

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        _settings(llm_provider="gemini", gemini_api_key=None).llm_endpoint()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _settings(llm_provider="anthropic").llm_endpoint()


def _failing_llm():
    import httpx
    import openai

    def fail(**_):
        raise openai.AuthenticationError("bad key", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)

    fake = FakeLLM([])
    fake.chat.completions.create = fail
    return fake


def test_fallback_provider_answers_when_main_fails(services, make_tool_call):
    rec = services.registry.add([0.0] * 3600, 360)
    backup = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": rec.id}))]),
        llm_response(content="From the fallback."),
    ])
    services.agent.client = _failing_llm()
    services.agent.fallback = (backup, "gemini-2.5-flash", "gemini")

    out = services.agent.chat("analyze", ecg_id=rec.id)

    assert out["answer"] == "From the fallback."
    assert out["llm"] == {"provider": "gemini", "model": "gemini-2.5-flash", "fallback_used": True}
    assert out["tool_calls"][0]["tool"] == "analyze_ecg"
    # once switched, the rest of the conversation stays on the fallback (2 calls, both to the backup)
    assert len(backup.requests) == 2 and all(r["model"] == "gemini-2.5-flash" for r in backup.requests)


def test_error_propagates_when_fallback_also_fails(services):
    import openai
    import pytest

    services.agent.client = _failing_llm()
    services.agent.fallback = (_failing_llm(), "gemini-2.5-flash", "gemini")
    with pytest.raises(openai.AuthenticationError):
        services.agent.chat("hi")


def test_temperature_is_sent_only_when_configured(services):
    services.agent.client = fake = FakeLLM([llm_response(content="a"), llm_response(content="b")])
    services.agent.temperature = None
    services.agent.chat("x")
    assert "temperature" not in fake.requests[0]
    services.agent.temperature = 0.0
    services.agent.chat("y")
    assert fake.requests[1]["temperature"] == 0.0


def test_trace_records_each_llm_decision_and_tool_execution(services, make_tool_call):
    rec = services.registry.add([0.0] * 3600, 360)
    services.agent.client = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": rec.id})),
                                 make_tool_call("c2", "analyze_ecg", json.dumps({"ecg_id": "nope"}))]),
        llm_response(content="done"),
    ])
    steps = services.agent.chat("x", ecg_id=rec.id)["steps"]
    assert [s["type"] for s in steps] == ["llm", "tool", "tool", "llm"]
    assert steps[0]["decision"] == "call_tools" and steps[0]["tool_calls"] == ["analyze_ecg", "analyze_ecg"]
    assert steps[1]["ok"] and steps[1]["error"] is None and steps[1]["duration_ms"] >= 0
    assert not steps[2]["ok"] and "Unknown ecg_id" in steps[2]["error"]
    assert steps[3]["decision"] == "answer"


def test_llm_gets_validation_errors_for_bad_arguments(services):
    assert "Invalid arguments" in services.agent.run_tool("analyze_ecg", {})["error"]
    assert "ecg_id must be string" in services.agent.run_tool("analyze_ecg", {"ecg_id": 5})["error"]
    assert "unknown: foo" in services.agent.run_tool("list_ecg_recordings", {"foo": 1})["error"]
    assert "top_k must be integer" in services.agent.run_tool("search_knowledge_base", {"query": "q", "top_k": True})["error"]
