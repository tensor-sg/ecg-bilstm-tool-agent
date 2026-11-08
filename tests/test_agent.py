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
    assert out == {"answer": "Hello", "tool_calls": [], "sources": []}


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
    services.agent.classifier = None
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
