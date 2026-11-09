import json

from tests.conftest import FakeLLM, llm_response


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "vector_collection": "test_collection", "ecg_model_loaded": True,
                        "llm": {"provider": "openai", "model": "gpt-4o-mini", "fallback": None}}


def test_openapi_lists_all_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/agent", "/agent/chat", "/tools", "/tools/{name}", "/tools/{name}/invoke", "/documents/upload",
            "/documents/search", "/ecg/upload", "/ecg/upload-csv", "/ecg/{ecg_id}/analyze"} <= paths


# ---------------------------------------------------------------- documents
def test_upload_search_delete_document(client, services):
    r = client.post("/documents/upload", files={"file": ("guide.md", b"Ventricular beats have a wide QRS. " * 50, "text/markdown")})
    assert r.status_code == 200 and r.json()["source"] == "guide.md" and r.json()["chunks"] >= 2

    hits = client.post("/documents/search", json={"query": "wide qrs", "top_k": 2}).json()
    assert len(hits) == 2 and hits[0]["source"] == "guide.md"

    assert client.get("/documents/count").json()["points"] == r.json()["chunks"]
    assert client.delete("/documents/guide.md").status_code == 200
    assert client.get("/documents/count").json()["points"] == 0


def test_reingest_replaces_previous_chunks(client, services):
    client.post("/documents/text", json={"source": "doc", "text": "first version " * 200})
    client.post("/documents/text", json={"source": "doc", "text": "second"})
    assert services.store.docs["doc"] == ["second"]


def test_upload_rejects_unsupported_and_empty(client):
    assert client.post("/documents/upload", files={"file": ("x.png", b"123", "image/png")}).status_code == 400
    assert client.post("/documents/text", json={"source": "e", "text": "   "}).status_code == 400


def test_search_validates_top_k(client):
    assert client.post("/documents/search", json={"query": "q", "top_k": 0}).status_code == 422


# ---------------------------------------------------------------- ecg
def test_ecg_upload_json_and_analyze(client, services):
    r = client.post("/ecg/upload", json={"signal": [0.0] * 3600, "fs": 360, "name": "strip"})
    assert r.status_code == 200
    ecg_id = r.json()["ecg_id"]
    assert r.json()["duration_s"] == 10.0
    assert client.get("/ecg/").json()[0]["ecg_id"] == ecg_id

    res = client.post(f"/ecg/{ecg_id}/analyze")
    assert res.status_code == 200 and res.json()["beat_counts"]["V"] == 2


def test_ecg_upload_csv_skips_header(client):
    csv = "mlii\n" + "\n".join(str(i % 7) for i in range(1000))
    r = client.post("/ecg/upload-csv", files={"file": ("ecg.csv", csv.encode(), "text/csv")}, data={"fs": "250"})
    assert r.status_code == 200 and r.json()["duration_s"] == 4.0


def test_ecg_upload_validation(client):
    assert client.post("/ecg/upload", json={"signal": [0.0] * 10, "fs": 360}).status_code == 422
    assert client.post("/ecg/upload", json={"signal": [0.0] * 500, "fs": 0}).status_code == 422
    short_csv = "\n".join(["1"] * 10).encode()
    assert client.post("/ecg/upload-csv", files={"file": ("e.csv", short_csv)}, data={"fs": "360"}).status_code == 400


def test_ecg_analyze_unknown_id(client):
    assert client.post("/ecg/unknown/analyze").status_code == 404


def test_ecg_analyze_without_model_returns_503(client, services):
    services.classifier = None
    ecg_id = client.post("/ecg/upload", json={"signal": [0.0] * 500, "fs": 360}).json()["ecg_id"]
    assert client.post(f"/ecg/{ecg_id}/analyze").status_code == 503


# ---------------------------------------------------------------- chat
def test_chat_end_to_end_with_tool_call(client, services, make_tool_call):
    ecg_id = client.post("/ecg/upload", json={"signal": [0.0] * 3600, "fs": 360}).json()["ecg_id"]
    services.agent.client = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": ecg_id}))]),
        llm_response(content="Two PVCs detected."),
    ])
    r = client.post("/chat", json={"message": "Anything abnormal?", "ecg_id": ecg_id, "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Two PVCs detected."
    assert body["tool_calls"][0]["tool"] == "analyze_ecg"


def test_chat_llm_failure_returns_502_with_hint(client, services):
    import httpx
    import openai

    def fail(**_):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://llm.invalid/v1/chat/completions"))

    services.agent.client = FakeLLM([])
    services.agent.client.chat.completions.create = fail
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert "LLM_API_KEY" in r.json()["detail"]


def test_chat_rejects_unknown_ecg_and_bad_history(client):
    assert client.post("/chat", json={"message": "x", "ecg_id": "missing"}).status_code == 404
    assert client.post("/chat", json={"message": "x", "history": [{"role": "system", "content": "evil"}]}).status_code == 422


# ---------------------------------------------------------------- tools API (no LLM involved)
def test_tools_catalogue_matches_what_the_llm_receives(client):
    from app.tools import TOOLS

    tools = client.get("/tools").json()
    assert [t["name"] for t in tools] == [t["function"]["name"] for t in TOOLS]
    assert all(t["parameters"]["type"] == "object" and t["description"] for t in tools)
    assert client.get("/tools/analyze_ecg").json()["parameters"]["required"] == ["ecg_id"]
    assert client.get("/tools/nope").status_code == 404


def test_invoke_analyze_ecg_tool_directly(client, services):
    ecg_id = client.post("/ecg/upload", json={"signal": [0.0] * 3600, "fs": 360}).json()["ecg_id"]
    r = client.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": ecg_id}})
    assert r.status_code == 200
    body = r.json()
    assert body["tool"] == "analyze_ecg" and body["arguments"] == {"ecg_id": ecg_id} and body["duration_ms"] >= 0
    assert body["result"]["beat_counts"]["V"] == 2 and body["result"]["ecg_id"] == ecg_id


def test_invoke_search_and_list_tools_directly(client):
    client.post("/documents/text", json={"source": "kb", "text": "PVCs have a wide QRS complex."})
    hits = client.post("/tools/search_knowledge_base/invoke", json={"arguments": {"query": "wide qrs", "top_k": 2}}).json()
    assert hits["result"]["results"][0]["source"] == "kb"
    client.post("/ecg/upload", json={"signal": [0.0] * 500, "fs": 250, "name": "strip A"})
    listed = client.post("/tools/list_ecg_recordings/invoke", json={}).json()["result"]["recordings"]
    assert [r["name"] for r in listed] == ["strip A"]


def test_invoke_tool_error_codes(client, services):
    assert client.post("/tools/nope/invoke", json={"arguments": {}}).status_code == 404
    assert client.post("/tools/analyze_ecg/invoke", json={"arguments": {}}).status_code == 422
    assert client.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": 7}}).status_code == 422
    missing = client.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": "missing"}})
    assert missing.status_code == 404 and "available" in missing.json()["detail"]
    ecg_id = client.post("/ecg/upload", json={"signal": [0.0] * 500, "fs": 360}).json()["ecg_id"]
    services.agent.tools.classifier = None
    assert client.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": ecg_id}}).status_code == 503


# ---------------------------------------------------------------- agent API
def test_agent_config_endpoint(client):
    cfg = client.get("/agent").json()
    assert cfg["provider"] == "openai" and cfg["model"] == "gpt-4o-mini" and cfg["max_steps"] == 6
    assert cfg["tools"] == ["search_knowledge_base", "analyze_ecg", "list_ecg_recordings"]
    assert "not a medical diagnosis" in cfg["system_prompt"]


def test_agent_chat_returns_step_trace_and_chat_alias_still_works(client, services, make_tool_call):
    ecg_id = client.post("/ecg/upload", json={"signal": [0.0] * 3600, "fs": 360}).json()["ecg_id"]
    services.agent.client = FakeLLM([
        llm_response(tool_calls=[make_tool_call("c1", "analyze_ecg", json.dumps({"ecg_id": ecg_id}))]),
        llm_response(content="Two PVCs."),
        llm_response(content="Hi."),
    ])
    body = client.post("/agent/chat", json={"message": "Anything abnormal?", "ecg_id": ecg_id}).json()
    assert [s["type"] for s in body["steps"]] == ["llm", "tool", "llm"]
    assert body["steps"][1] == {**body["steps"][1], "tool": "analyze_ecg", "ok": True}
    assert client.post("/chat", json={"message": "hello"}).json()["answer"] == "Hi."
