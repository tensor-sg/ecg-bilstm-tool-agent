# API reference

Base URL: `http://localhost:8000` (start with `uvicorn app.main:app`). Interactive docs: `/docs` (Swagger UI) and
`/redoc`. All examples below are real responses from the running service (record 233 of MIT-BIH, a held-out
patient; long lists shortened with `"..."`).

The service is split into four parts:

| Part | Prefix | Purpose | Needs an LLM? |
|---|---|---|---|
| Agent API | `/agent` | OpenAI agent that decides when to call the tools and answers in natural language | yes |
| Tools API | `/tools` | the agent's tools as plain REST calls (LSTM, knowledge-base search, recordings list) | no |
| ECG | `/ecg` | upload recordings, run the LSTM directly | no |
| Documents | `/documents` | build and search the RAG knowledge base (Chroma) | no |

There is no authentication; run the service on a trusted network only.

---

## 1. How the agent works

```
POST /agent/chat {message, ecg_id?, history?}
   │
   ▼
Agent ── system prompt + history + message + tool schemas ──► OpenAI gpt-4o-mini
   ▲                                                              │
   │◄──────────── tool calls, e.g. analyze_ecg(ecg_id) ───────────┘
   │
   ├─► ToolBox validates arguments against the JSON schema and runs the tool
   │      analyze_ecg ─────────► Conv-BiLSTM on the stored recording
   │      search_knowledge_base ► Chroma similarity search
   │      list_ecg_recordings ──► in-memory registry
   │
   └─► tool results (JSON) sent back to the LLM ... repeated until it answers (max 6 LLM calls)
```

- **Main provider:** OpenAI `gpt-4o-mini` (`LLM_PROVIDER=openai`, `OPENAI_API_KEY`).
- **Fallback:** if a call to the main provider raises an API error (invalid key, quota, outage), the agent repeats
  it on Gemini `gemini-2.5-flash` (`LLM_FALLBACK_PROVIDER=gemini`, `GEMINI_API_KEY`) and stays on Gemini for the rest
  of that request. The response says which model answered.
- **Tool errors** (unknown `ecg_id`, invalid arguments, model not loaded) are returned to the LLM as
  `{"error": ...}` so it can recover; the same errors are HTTP status codes in the Tools API.
- **System prompt:** call `analyze_ecg` instead of guessing, cite knowledge-base sources, mention model confidence
  and limitations, ignore results flagged by `signal_quality`, state that the output is not a diagnosis
  (`GET /agent` returns it in full).

**Data sent to the LLM provider:** the question, the conversation history and the tool results (heart rate, beat
counts, abnormal beat times, retrieved document chunks). The raw ECG signal is not sent.

---

## 2. Agent API

### `GET /agent`
Configuration in use.
```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "fallback": "gemini:gemini-2.5-flash",
  "temperature": 0.2,
  "max_steps": 6,
  "tools": ["search_knowledge_base", "analyze_ecg", "list_ecg_recordings"],
  "system_prompt": "You are a cardiology research assistant.\nYou have tools to (1) search a knowledg..."
}
```

### `POST /agent/chat`
| Field | Type | Required | Notes |
|---|---|---|---|
| `message` | string | yes | the question |
| `ecg_id` | string | no | attaches an uploaded recording; 404 if unknown |
| `history` | list of `{role, content}` | no | earlier turns; `role` must be `user` or `assistant` (422 otherwise) |

Request:
```json
{"message": "Is anything abnormal in this ECG? Answer in three sentences.", "ecg_id": "de9cf8e3add8"}
```
Response:
```json
{
  "answer": "The ECG recording shows a heart rate of approximately 101.8 beats per minute, which is on the higher side. There are 36 normal beats and 13 ventricular ectopic beats, indicating some abnormal activity in the heart. The model's confidence in identifying these ventricular ectopic beats is high, with values ranging from 85.8% to 98.9%. \n\nPlease note that these results are not a medical diagnosis.",
  "tool_calls": [{"tool": "analyze_ecg", "arguments": {"ecg_id": "de9cf8e3add8"}}],
  "sources": [],
  "llm": {"provider": "openai", "model": "gpt-4o-mini", "fallback_used": false},
  "steps": [
    {"type": "llm", "provider": "openai", "model": "gpt-4o-mini", "duration_ms": 2622.4,
     "tokens": {"prompt": 363, "completion": 25}, "decision": "call_tools", "tool_calls": ["analyze_ecg"]},
    {"type": "tool", "tool": "analyze_ecg", "arguments": {"ecg_id": "de9cf8e3add8"},
     "duration_ms": 104.9, "ok": true, "error": null},
    {"type": "llm", "provider": "openai", "model": "gpt-4o-mini", "duration_ms": 2568.7,
     "tokens": {"prompt": 1013, "completion": 88}, "decision": "answer", "tool_calls": []}
  ]
}
```
| Field | Meaning |
|---|---|
| `tool_calls` | every tool the LLM called, with its arguments |
| `sources` | knowledge-base chunks returned by `search_knowledge_base` (`source`, `chunk_index`, `score`) |
| `llm` | provider and model that produced the answer; `fallback_used` is true if Gemini took over |
| `steps` | the full loop in order: each LLM call (decision, duration, tokens) and each tool run (duration, success, error) |

Errors: `404` unknown `ecg_id`, `422` invalid body, `502` both LLM providers failed (message names the settings to check).

`POST /chat` is an alias of `/agent/chat` kept for existing clients.

---

## 3. Tools API

The same tools, schemas and code the agent uses, callable without an LLM.

### `GET /tools`
```json
[
  {"name": "search_knowledge_base", "description": "Semantic search over the uploaded document knowledge base.",
   "parameters": {"type": "object", "properties": {
       "query": {"type": "string", "description": "Natural-language search query."},
       "top_k": {"type": "integer", "description": "Number of chunks to return (1-10).", "default": 5}},
     "required": ["query"]}},
  {"name": "analyze_ecg", "description": "Run the trained LSTM classifier on an uploaded single-lead ECG recording. Returns heart rate, RR variability, per-class beat counts (AAMI N/S/V) and abnormal beat times.",
   "parameters": {"type": "object", "properties": {
       "ecg_id": {"type": "string", "description": "Id returned by POST /ecg/upload."}},
     "required": ["ecg_id"]}},
  {"name": "list_ecg_recordings", "description": "List the ECG recordings currently available for analysis.",
   "parameters": {"type": "object", "properties": {}}}
]
```

### `GET /tools/{name}`
One entry of the list above; `404` for an unknown tool.

### `POST /tools/{name}/invoke`
Body: `{"arguments": {...}}` (may be omitted for tools without arguments).

```json
POST /tools/analyze_ecg/invoke
{"arguments": {"ecg_id": "de9cf8e3add8"}}

{
  "tool": "analyze_ecg",
  "arguments": {"ecg_id": "de9cf8e3add8"},
  "result": {"ecg_id": "de9cf8e3add8", "n_beats_classified": 49, "heart_rate_bpm": 101.8,
             "beat_counts": {"N": 36, "S": 0, "V": 13}, "...": "same fields as /ecg/{id}/analyze"},
  "duration_ms": 116.3
}
```
```json
POST /tools/search_knowledge_base/invoke
{"arguments": {"query": "compensatory pause", "top_k": 1}}

{
  "tool": "search_knowledge_base",
  "arguments": {"query": "compensatory pause", "top_k": 1},
  "result": {"results": [{"score": 0.2528, "source": "ecg_reference.md", "chunk_index": 2,
                          "text": "followed by a compensatory pause.\n- **F (fusion beats):** ..."}]},
  "duration_ms": 301.7
}
```

| Status | When | Example `detail` |
|---|---|---|
| 404 | unknown tool | `"Unknown tool 'nope'"` |
| 422 | argument missing, unknown or of the wrong type | `"Invalid arguments for analyze_ecg: ecg_id must be string"` |
| 404 | unknown `ecg_id` | `{"message": "Unknown ecg_id 'missing'.", "available": [{"ecg_id": "de9cf8e3add8", "name": "mitdb_233", ...}]}` |
| 503 | ECG model not loaded | `"ECG model is not loaded. Train it with notebooks/02_lstm_training.ipynb."` |

---

## 4. ECG

### `POST /ecg/upload`
| Field | Type | Required | Notes |
|---|---|---|---|
| `signal` | list of numbers | yes | single lead, at least 256 samples (any unit; the model is gain-invariant) |
| `fs` | number > 0 | yes | sampling rate in Hz; resampled to 360 Hz internally |
| `name` | string | no | label shown in listings |
| `r_peaks` | list of ints | no | R-peak sample indices; detected automatically if omitted |

```json
{"ecg_id": "de9cf8e3add8", "duration_s": 30.0}
```

### `POST /ecg/upload-csv`
Multipart form: `file` (CSV, one value per row; header and malformed rows are skipped), `fs` (Hz),
optional `column` (0-based). `400` if fewer than 256 numeric samples.

### `GET /ecg/`
```json
[{"ecg_id": "de9cf8e3add8", "name": "mitdb_233", "fs": 360.0, "duration_s": 30.0, "created_at": "2026-11-09T05:01:04.891403+00:00"}]
```

### `POST /ecg/{ecg_id}/analyze`
Runs the LSTM pipeline: filter, detect beats, classify, summarise.
```json
{
  "duration_s": 30.0,
  "n_beats_classified": 49,
  "heart_rate_bpm": 101.8,
  "rr_sdnn_ms": 120.6,
  "beat_counts": {"N": 36, "S": 0, "V": 13},
  "beat_percentages": {"N": 73.47, "S": 0.0, "V": 26.53},
  "class_descriptions": {"N": "Normal / bundle branch block / escape beat",
                         "S": "Supraventricular ectopic beat", "V": "Ventricular ectopic beat"},
  "abnormal_beats": [{"beat_index": 2, "time_s": 1.383, "label": "V", "confidence": 0.904},
                     {"beat_index": 5, "time_s": 3.081, "label": "V", "confidence": 0.984}, "..."],
  "abnormal_beats_truncated": false,
  "signal_quality": {"ok": true, "low_quality_fraction": 0.0, "median_kurtosis": 5.12, "warning": null},
  "disclaimer": "Automated research model output; not a medical diagnosis."
}
```
| Field | Meaning |
|---|---|
| `abnormal_beats` | up to 50 S/V beats with time and softmax confidence (`abnormal_beats_truncated` if more) |
| `signal_quality` | kurtosis-based check per 10 s window; `ok: false` with a `warning` when the input looks like noise or artefact |

Errors: `404` unknown id, `503` model not loaded.

---

## 5. Documents (knowledge base)

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/documents/upload` | multipart `file` (.pdf, .txt, .md) | `{"source": "ecg_reference.md", "chunks": 5}` |
| POST | `/documents/text` | `{"source": "...", "text": "..."}` | same as upload |
| POST | `/documents/search` | `{"query": "...", "top_k": 1-20, "source": optional}` | list of hits (below) |
| DELETE | `/documents/{source}` | | `{"deleted": "ecg_reference.md"}` |
| GET | `/documents/count` | | `{"collection": "ecg_knowledge", "points": 5}` |

```json
POST /documents/search {"query": "What is a PVC?", "top_k": 2}
[
  {"score": 0.3138, "source": "ecg_reference.md", "chunk_index": 4,
   "text": "## Clinical meaning of ectopic beats\n\nOccasional PVCs and atrial premature beats are commo..."},
  {"score": 0.267, "source": "ecg_reference.md", "chunk_index": 1,
   "text": "- **N (normal and bundle branch block beats):** normal sinus beats, left bundle branch blo..."}
]
```
Chunks are 800 characters with 120 overlap, embedded locally with all-MiniLM-L6-v2 and stored in Chroma under cosine
distance (`score = 1 − distance`). Uploading a document with an existing name replaces its chunks. Errors: `400`
unsupported file type or no extractable text.

---

## 6. Health

### `GET /health`
```json
{"status": "ok", "vector_collection": "ecg_knowledge", "ecg_model_loaded": true,
 "llm": {"provider": "openai", "model": "gpt-4o-mini", "fallback": "gemini:gemini-2.5-flash"}}
```

---

## 7. Python client example

```python
import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=120) as api:
    ecg_id = api.post("/ecg/upload", json={"signal": samples, "fs": 250, "name": "patient strip"}).json()["ecg_id"]

    # tool only (no LLM): the LSTM result
    counts = api.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": ecg_id}}).json()["result"]["beat_counts"]

    # agent: OpenAI decides to call the LSTM and explains the result
    reply = api.post("/agent/chat", json={"message": "Is this ECG abnormal?", "ecg_id": ecg_id}).json()
    print(reply["answer"])
    for step in reply["steps"]:
        print(step["type"], step.get("tool") or step.get("decision"), step["duration_ms"], "ms")
```

`scripts/demo_agent.py` runs a complete session against a running server and prints the trace.
