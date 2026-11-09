# ECG RAG + LSTM Agent

**Author:** Priyatosh Satpati (Reg. No. 25113170096) · priyatoshsatpatiofficial@gmail.com
M.Sc. Data Science, 4th semester project, SASTRA Deemed to be University · Guide: Mr. G. Swaminathan

A FastAPI service that combines:

- **RAG over a persistent local Chroma vector store**: upload PDFs, text or markdown, then run semantic search
- An **LSTM ECG beat classifier** trained on the MIT-BIH Arrhythmia Database
- An **LLM agent** whose agentic core is **OpenAI `gpt-4o-mini`**: it decides which tools to call and writes the answer.
  Our LSTM and the knowledge-base search are its tools. If OpenAI fails (bad key, quota, outage), the agent switches
  to **Gemini `gemini-2.5-flash`** automatically.

```
                 ┌───────────────────────────── FastAPI ─────────────────────────────┐
 user ─► /agent/chat ─► Agent (app/agent.py)  ◄── decides / answers ──►  OpenAI gpt-4o-mini
                 │         │ tool calls                                  (fallback: Gemini)
                 │         ▼                                                         │
 any client ─► /tools/{name}/invoke ─► ToolBox (app/tools.py)                         │
                 │         ├─ analyze_ecg ─────────────► Conv-BiLSTM (models/ecg_lstm.keras)
                 │         ├─ search_knowledge_base ───► Chroma (chroma_db/, on disk)  │
                 │         └─ list_ecg_recordings ─────► ECG registry                  │
                 │  /ecg/*  (upload, analyze)   /documents/*  (ingest, search)   /health
                 └───────────────────────────────────────────────────────────────────┘
```

The agent and the tools are **separate APIs**. `/tools` exposes exactly the functions and JSON schemas the LLM is
offered, so any client (another agent, a script, a different LLM) can call the LSTM the same way. `/agent/chat`
runs the full loop and returns a **step trace**: every LLM decision and tool execution, with timings and token counts.

### Request flow of `/agent/chat`

1. The client uploads an ECG (`/ecg/upload`) and gets an `ecg_id`, then asks a question with that id.
2. The agent sends the system prompt, history, question and tool schemas to OpenAI.
3. OpenAI answers with tool calls, e.g. `analyze_ecg(ecg_id)` or `search_knowledge_base(query)`.
4. The agent runs them through the ToolBox (arguments validated against the JSON schema) and sends the results back.
5. Steps 3–4 repeat (at most 6 LLM calls) until OpenAI writes the final answer from the tool results.
6. The response has the answer, the tool calls, the cited knowledge-base chunks, the model used and the step trace.

## Results (22 held-out patients, MIT-BIH DS2)

| | Accuracy | Macro-F1 | N Se / PPV | S Se / PPV | V Se / PPV |
|---|---:|---:|---:|---:|---:|
| Always "normal" baseline | 0.897 | 0.315 | 1.00 / 0.90 | 0 / – | 0 / – |
| RR-only logistic regression | 0.872 | 0.530 | 0.91 / 0.99 | 0.07 / 0.05 | 0.85 / 0.45 |
| **Conv-BiLSTM** | **0.945** | **0.650** | **0.99 / 0.96** | 0.10 / 0.36 | **0.80 / 0.85** |

End to end through the API's own beat detector: detection sensitivity 0.979, PPV 0.998, accuracy 0.942.
The model is reliable for normal and ventricular beats but **misses most supraventricular beats**.
Details, confidence intervals and limitations: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and [reports/evaluation.md](reports/evaluation.md).
API reference with real request and response examples: [docs/API.md](docs/API.md). Changes: [CHANGELOG.md](CHANGELOG.md).
Speed, calibration (ECE 0.013) and error analysis: [reports/performance.md](reports/performance.md). Test results: [reports/test_report.md](reports/test_report.md).

## Academic documents

| File | What it is |
|---|---|
| [docs/academic/25113170096.pdf](docs/academic/25113170096.pdf) (+ `.docx`) | Project synopsis in the SASTRA template (≤ 300 words) |
| [docs/academic/Project_Report_25113170096.pdf](docs/academic/Project_Report_25113170096.pdf) (+ `.docx`) | Full project report: literature review, methodology, system design and ERD, testing, results, references |
| [docs/figures/](docs/figures/) | Figures and diagrams (architecture, model, ERD, request flow, results), made by `scripts/make_figures.py` |

> **Not a medical device.** The model is a research tool. Read [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for what it can and cannot do.

## Project layout

```
app/
  main.py            FastAPI app and lifespan wiring
  config.py          settings loaded from .env
  agent.py           agentic core: LLM loop, provider fallback, step trace
  tools.py           tool layer: JSON schemas, argument validation, handlers, ECG registry
  vector_store.py    persistent Chroma collection: upsert, search, delete
  embeddings.py      local ONNX MiniLM (default) or OpenAI-compatible embeddings
  ingest.py          text extraction (pdf/txt/md) and chunking
  schemas.py         request / response models
  routers/           agent, tools, ecg, documents endpoints
  ecg/
    preprocessing.py filtering, R-peak detection, beat segmentation, RR features (shared by notebooks and API)
    mitbih.py        MIT-BIH download, loading, patient splits
    classifier.py    inference wrapper used by the API and the LLM tool
    evaluation.py    AAMI metrics, patient-level bootstrap CIs, end-to-end evaluation
notebooks/
  01_ecg_analysis.ipynb       exploratory ECG analysis
  02_lstm_training.ipynb      train, validate and test the LSTM, save to models/
  03_lstm_as_llm_tool.ipynb   the LSTM as an OpenAI tool, with and without RAG, the Agent and the REST APIs
scripts/
  evaluate_model.py           held-out accuracy evaluation -> reports/evaluation.md
  benchmark_model.py          speed, memory, calibration, error analysis -> reports/performance.md
  test_report.py              runs the whole suite -> reports/test_report.md
  make_figures.py             report figures and diagrams -> docs/figures/
  demo_agent.py               live session against a running server: tools API + agent trace
reports/                      committed evaluation, performance and test results
tests/                        pytest suite (unit, API, integration, model, performance, live LLM)
docs/
  API.md                      API reference: agent flow, every endpoint, real examples, error codes
  MODEL_CARD.md               model details, validation protocol, results, limitations
  ecg_reference.md            seed document for the knowledge base
  academic/                   synopsis and project report (.docx and .pdf) and their generator
  figures/                    figures and diagrams used in the report
```

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then set OPENAI_API_KEY (and GEMINI_API_KEY for the fallback)
```

| Variable | Default | Notes |
|---|---|---|
| `CHROMA_PATH` | `chroma_db` | Folder for the persistent vector store (gitignored) |
| `CHROMA_COLLECTION` | `ecg_knowledge` | |
| `LLM_PROVIDER`, `LLM_MODEL` | `openai`, `gpt-4o-mini` | Agentic core. `openai` uses `OPENAI_API_KEY`; `gemini` uses `GEMINI_API_KEY`; `custom` uses `LLM_API_KEY` + `LLM_BASE_URL` (Ollama, vLLM, LM Studio) |
| `LLM_FALLBACK_PROVIDER`, `LLM_FALLBACK_MODEL` | none | e.g. `gemini`, `gemini-2.5-flash`: used when the main provider fails |
| `LLM_TEMPERATURE` | provider default | e.g. `0.2` |
| `OPENAI_API_KEY`, `GEMINI_API_KEY` | | provider keys (never committed; `.env` is gitignored) |
| `EMBEDDING_PROVIDER` | `local` | `local` runs all-MiniLM-L6-v2 on CPU with no key (downloaded once, ~80 MB). `openai` uses `EMBEDDING_*` settings |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | 800 / 120 / 5 | RAG chunking and retrieval |
| `ECG_MODEL_PATH`, `ECG_META_PATH` | `models/ecg_lstm.*` | Written by notebook 02 |

A Chroma collection remembers which embedding model built it. If you change `EMBEDDING_PROVIDER` or
`EMBEDDING_MODEL`, use a new `CHROMA_COLLECTION` or delete `chroma_db/`. The app refuses to mix embeddings.

## Train the model

Run the notebooks in order from the `notebooks/` folder (`jupyter lab`):

1. `01_ecg_analysis.ipynb` downloads MIT-BIH (~100 MB) into `data/mitdb/` and explores it.
2. `02_lstm_training.ipynb` trains the Conv1D + BiLSTM classifier and writes `models/ecg_lstm.keras`,
   `models/ecg_lstm_meta.json` and `models/training_log.csv`. It takes about 10–15 minutes on a 2-core laptop CPU.
3. `03_lstm_as_llm_tool.ipynb` wires the model to OpenAI as a tool, adds Chroma RAG, then uses the production
   Agent and the Tools and Agent APIs (needs `OPENAI_API_KEY`; section 4 needs the server running).

### Validation protocol

All splits are **by patient**, so no person's heartbeats appear in more than one set:

| Set | MIT-BIH records | Used for |
|---|---|---|
| Train | DS1 without the validation patients (17) | fitting weights |
| Validation | 106, 118, 124, 201, 205 | early stopping and checkpoint selection on macro-F1 |
| Test | DS2 (22 records, de Chazal et al. 2004) | final evaluation only |

Random beat-level splits are common in tutorials but let the model see the test patients during training,
which inflates scores. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the full methodology.

## Evaluate

```bash
python scripts/evaluate_model.py           # DS2 test patients -> reports/evaluation.md + .json
```

The report covers:
- per-class sensitivity, PPV, specificity and F1, with **patient-level bootstrap 95% CIs**
- **end-to-end** results through the API's own R-peak detector, not only the annotated peaks
- what the model predicts for beat types it was not trained on (fusion, paced)
- per-record breakdown

## Test

```bash
pytest                      # 108 tests (~2.5 min); model tests are skipped if models/ or data/ are missing
pytest -m "not model"       # fast unit/API tests only (~10 s, no trained model or data needed)
pytest -m model             # trained model on held-out patients: accuracy, performance, integration
pytest -m integration       # real app: lifespan, trained model, Chroma on disk, local embeddings
RUN_LLM_TESTS=1 pytest -m llm  # 11 live tests with the real OpenAI and Gemini APIs (~1 min, uses API credit)
RUN_LLM_TESTS=1 python scripts/test_report.py   # all 119 tests -> reports/test_report.md
```

| File | Covers |
|---|---|
| `test_preprocessing.py` | resampling, band-pass filter attenuation, R-peak detection accuracy on synthetic ECG (60–120 bpm, 250 Hz input), heart-rate-independent RR features, segmentation edge cases |
| `test_ingest.py` | txt/md/pdf extraction, chunk size and overlap |
| `test_vector_store.py` | real Chroma on a temp dir: ranking, persistence across restarts, upsert de-duplication, source filter, delete, embedding-model mismatch guard |
| `test_agent.py` | agent loop with a scripted LLM: tool dispatch, results fed back, sources, step trace, argument validation, unknown tools, missing model, step limit; provider settings, fallback to Gemini, error when both fail, temperature |
| `test_api.py` | every endpoint: tools catalogue and direct invocation (404/422/503), agent config and chat trace, `/chat` alias, ECG and document endpoints, validation errors, 502 on LLM failure |
| `test_model.py` | trained model on DS2 patients: metadata and patient-leakage checks, determinism, API path matches training path, invariance to gain/offset, robustness to 250 Hz input and 25 dB noise, signal quality on real records, accuracy regression floors, beats the baselines, end-to-end detection |
| `test_model_performance.py` | model size, batched throughput, 30-min record in < 30 s, batch-size invariance, no memory leak, calibration (ECE, Brier), ROC AUC per class, selective accuracy, false V alarms, V specificity, S detection outside record 232, per-patient V sensitivity, bootstrap CI and paired bootstrap vs the RR baseline |
| `test_api_integration.py` | the real app with its lifespan: model loaded, ingest and semantic search with real embeddings, persistence across restart, 250 Hz CSV of a held-out patient vs cardiologist labels, supplied R-peaks, response contract, latency, noise flagged by signal quality, agent tool calls on real services |
| `test_llm_live.py` | **real OpenAI agent**: calls the LSTM and quotes its numbers, flags noise, lists recordings, multi-turn follow-up, RAG answers cite the knowledge base, both tools in one question, direct tool call matches the agent's, full step trace with tokens; **Gemini** as main agent and as automatic fallback when the OpenAI key is rejected |

The suite was mutation-checked: deliberately injected bugs (swapped RR features, a wrong similarity score,
a dropped tool result, RR features ignored or zeroed, the signal-quality check disabled) each make tests fail.

## Run the API

```bash
uvicorn app.main:app --reload
# interactive docs at http://localhost:8000/docs
```

The API starts without a trained model; the ECG endpoints and the `analyze_ecg` tool return an error until
`models/ecg_lstm.keras` exists.

### Live demo

```bash
uvicorn app.main:app &
python scripts/demo_agent.py          # uploads a held-out patient's ECG, calls the tool directly, then chats
```

Example trace printed by the demo (record 233, real OpenAI call):

```
direct tool call (no LLM): {'N': 36, 'S': 0, 'V': 13} in 1239 ms
=== Is anything abnormal in this ECG? Give the numbers.
  [LLM     2027 ms] openai:gpt-4o-mini decides to call analyze_ecg  (361+24 tokens)
  [TOOL     104 ms] analyze_ecg(ecg_id='4e566362da1b')
  [LLM     7207 ms] openai:gpt-4o-mini decides to answer  (1009+531 tokens)
  ... Normal beats (N): 36 (73.47%) ... Ventricular ectopic beats (V): 13 (26.53%) ...
```

### Examples

```bash
# ingest a document into the knowledge base
curl -F "file=@docs/ecg_reference.md" localhost:8000/documents/upload

# semantic search
curl -X POST localhost:8000/documents/search -H 'Content-Type: application/json' \
     -d '{"query": "what is a PVC?", "top_k": 3}'

# upload an ECG (single lead, one value per row) and analyse it
curl -F "file=@my_ecg.csv" -F "fs=250" localhost:8000/ecg/upload-csv     # -> {"ecg_id": "..."}
curl -X POST localhost:8000/ecg/<ecg_id>/analyze

# call the LSTM tool directly (no LLM), exactly as the agent would
curl localhost:8000/tools
curl -X POST localhost:8000/tools/analyze_ecg/invoke -H 'Content-Type: application/json' \
     -d '{"arguments": {"ecg_id": "<ecg_id>"}}'

# ask the agent; OpenAI decides when to call the LSTM and the knowledge base
curl localhost:8000/agent                      # provider, model, fallback, tools, system prompt
curl -X POST localhost:8000/agent/chat -H 'Content-Type: application/json' \
     -d '{"message": "Is there anything abnormal in this ECG?", "ecg_id": "<ecg_id>"}'
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, vector collection, whether the model is loaded, LLM in use |
| GET | `/agent` | agent configuration: provider, model, fallback, temperature, tools, system prompt |
| POST | `/agent/chat` | run the agent (`message`, optional `ecg_id` and `history`): answer, tool calls, sources, `llm`, `steps` trace |
| GET | `/tools` | tool catalogue with JSON schemas (what the LLM is offered) |
| GET | `/tools/{name}` | one tool's schema |
| POST | `/tools/{name}/invoke` | run a tool directly with `{"arguments": {...}}` (404 unknown tool or ecg_id, 422 invalid arguments, 503 model not loaded) |
| POST | `/documents/upload` | upload pdf / txt / md; chunk, embed and store in Chroma |
| POST | `/documents/text` | ingest raw text under a source name |
| POST | `/documents/search` | semantic search (optional `source` filter) |
| DELETE | `/documents/{source}` | remove all chunks of a source |
| GET | `/documents/count` | number of chunks in the collection |
| POST | `/ecg/upload` | upload ECG as JSON `{signal, fs, r_peaks?}` |
| POST | `/ecg/upload-csv` | upload ECG as CSV (`fs` form field, optional `column`) |
| GET | `/ecg/` | list uploaded recordings |
| POST | `/ecg/{ecg_id}/analyze` | run the LSTM directly |
| POST | `/chat` | alias of `/agent/chat`, kept for existing clients |

The `analyze` response contains heart rate, RR variability (SDNN), per-class beat counts and percentages,
up to 50 abnormal beats with timestamps and confidence, and a `signal_quality` block. A kurtosis-based index flags
input that does not look like an ECG (noise, artefact, no QRS complexes); the agent is told not to interpret such results.

## Notes

- **Data sent to the LLM provider:** questions, conversation history and tool results (heart rate, beat counts,
  abnormal beat times, retrieved document chunks) go to OpenAI, or to Google when the fallback is used. The raw ECG
  signal is not sent. Use `LLM_PROVIDER=custom` with a local server (e.g. Ollama) to keep everything on-premises.
- **Cost:** a typical ECG question uses two LLM calls of about 1,500 tokens in total with `gpt-4o-mini`.
- The API has no authentication; run it on a trusted network.
- Uploaded ECGs are kept in memory and are lost on restart. Use Redis or a database for persistence.
- Re-uploading a document with the same name replaces its chunks.

## Citation

If you use this project, please cite it as described in [CITATION.cff](CITATION.cff), and cite the MIT-BIH
Arrhythmia Database (Moody & Mark, 2001) and PhysioNet (Goldberger et al., 2000).
