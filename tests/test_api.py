import json

from tests.conftest import FakeLLM, llm_response


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "vector_collection": "test_collection", "ecg_model_loaded": True}


def test_openapi_lists_all_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/chat", "/documents/upload", "/documents/search", "/ecg/upload", "/ecg/upload-csv", "/ecg/{ecg_id}/analyze"} <= paths


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
