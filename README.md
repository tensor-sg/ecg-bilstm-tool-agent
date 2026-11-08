# ECG RAG + LSTM Agent

A FastAPI service that combines:

- **RAG over a persistent local Chroma vector store**: upload PDFs, text or markdown, then run semantic search
- An **LSTM ECG beat classifier** trained on the MIT-BIH Arrhythmia Database
- An **LLM agent** that calls both as tools: it can search the knowledge base and run the LSTM on uploaded ECG recordings

```
            ┌──────────────────── FastAPI ────────────────────┐
 user ──►   │ /chat ─► LLM (any OpenAI-compatible endpoint)   │
            │            │ tool calls                         │
            │            ├─► search_knowledge_base ───────────┼──► Chroma (chroma_db/, on disk)
            │            ├─► analyze_ecg ─────────────────────┼──► Keras Conv-BiLSTM model
            │            └─► list_ecg_recordings              │
            │ /documents/*   /ecg/*   /health                 │
            └─────────────────────────────────────────────────┘
```

## Results (22 held-out patients, MIT-BIH DS2)

| | Accuracy | Macro-F1 | N Se / PPV | S Se / PPV | V Se / PPV |
|---|---:|---:|---:|---:|---:|
| Always "normal" baseline | 0.897 | 0.315 | 1.00 / 0.90 | 0 / – | 0 / – |
| RR-only logistic regression | 0.872 | 0.530 | 0.91 / 0.99 | 0.07 / 0.05 | 0.85 / 0.45 |
| **Conv-BiLSTM** | **0.945** | **0.650** | **0.99 / 0.96** | 0.10 / 0.36 | **0.80 / 0.85** |

End to end through the API's own beat detector: detection sensitivity 0.979, PPV 0.998, accuracy 0.942.
The model is reliable for normal and ventricular beats but **misses most supraventricular beats**.
Details, confidence intervals and limitations: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and [reports/evaluation.md](reports/evaluation.md).

> **Not a medical device.** The model is a research tool. Read [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for what it can and cannot do.

## Project layout

```
app/
  main.py            FastAPI app and lifespan wiring
  config.py          settings loaded from .env
  agent.py           tool-calling loop, tool schemas, in-memory ECG registry
  vector_store.py    persistent Chroma collection: upsert, search, delete
  embeddings.py      local ONNX MiniLM (default) or OpenAI-compatible embeddings
  ingest.py          text extraction (pdf/txt/md) and chunking
  schemas.py         request / response models
  routers/           documents, ecg, chat endpoints
  ecg/
    preprocessing.py filtering, R-peak detection, beat segmentation, RR features (shared by notebooks and API)
    mitbih.py        MIT-BIH download, loading, patient splits
    classifier.py    inference wrapper used by the API and the LLM tool
    evaluation.py    AAMI metrics, patient-level bootstrap CIs, end-to-end evaluation
notebooks/
  01_ecg_analysis.ipynb       exploratory ECG analysis
  02_lstm_training.ipynb      train, validate and test the LSTM, save to models/
  03_lstm_as_llm_tool.ipynb   use the LSTM as an LLM tool, with and without RAG
scripts/evaluate_model.py     full held-out evaluation, writes reports/evaluation.md
reports/                      committed evaluation results
tests/                        pytest suite (unit, API, vector store, model)
docs/
  MODEL_CARD.md               model details, validation protocol, results, limitations
  ecg_reference.md            seed document for the knowledge base
```

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then set LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL if not OpenAI)
```

| Variable | Default | Notes |
|---|---|---|
| `CHROMA_PATH` | `chroma_db` | Folder for the persistent vector store (gitignored) |
| `CHROMA_COLLECTION` | `ecg_knowledge` | |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | OpenAI, `gpt-4o-mini` | Any OpenAI-compatible endpoint with tool calling (OpenAI, Ollama `http://localhost:11434/v1`, vLLM, LM Studio) |
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
3. `03_lstm_as_llm_tool.ipynb` wires the model to an LLM as a tool, adds Chroma RAG, and calls the API.

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
pytest                      # everything; model tests are skipped if models/ or data/ are missing
pytest -m "not model"       # fast unit/API tests only (~10 s, no trained model or data needed)
pytest -m model             # tests against the trained model on held-out patients (~1.5 min)
```

| File | Covers |
|---|---|
| `test_preprocessing.py` | resampling, band-pass filter attenuation, R-peak detection accuracy on synthetic ECG (60–120 bpm, 250 Hz input), heart-rate-independent RR features, segmentation edge cases |
| `test_ingest.py` | txt/md/pdf extraction, chunk size and overlap |
| `test_vector_store.py` | real Chroma on a temp dir: ranking, persistence across restarts, upsert de-duplication, source filter, delete, embedding-model mismatch guard |
| `test_agent.py` | tool-calling loop with a scripted LLM: tool dispatch, tool results fed back, sources, bad arguments, unknown tools, missing model, step limit |
| `test_api.py` | every endpoint, validation errors (422/400/404/503), chat with tool calls |
| `test_model.py` | trained model on DS2 patients: metadata and patient-leakage checks, determinism, API path matches training path, invariance to gain/offset, robustness to 250 Hz input and 25 dB noise, accuracy regression floors, beats the baselines, end-to-end detection |

The suite was mutation-checked: deliberately injected bugs (swapped RR features, a wrong similarity score,
a dropped tool result) each make tests fail.

## Run the API

```bash
uvicorn app.main:app --reload
# interactive docs at http://localhost:8000/docs
```

The API starts without a trained model; the ECG endpoints and the `analyze_ecg` tool return an error until
`models/ecg_lstm.keras` exists.

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

# ask the agent; it decides when to call the LSTM and the knowledge base
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"message": "Is there anything abnormal in this ECG?", "ecg_id": "<ecg_id>"}'
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, vector collection, whether the model is loaded |
| POST | `/documents/upload` | upload pdf / txt / md; chunk, embed and store in Chroma |
| POST | `/documents/text` | ingest raw text under a source name |
| POST | `/documents/search` | semantic search (optional `source` filter) |
| DELETE | `/documents/{source}` | remove all chunks of a source |
| GET | `/documents/count` | number of chunks in the collection |
| POST | `/ecg/upload` | upload ECG as JSON `{signal, fs, r_peaks?}` |
| POST | `/ecg/upload-csv` | upload ECG as CSV (`fs` form field, optional `column`) |
| GET | `/ecg/` | list uploaded recordings |
| POST | `/ecg/{ecg_id}/analyze` | run the LSTM directly |
| POST | `/chat` | agent with RAG and ECG tools (`message`, optional `ecg_id` and `history`) |

The `analyze` response contains heart rate, RR variability (SDNN), per-class beat counts and percentages,
and up to 50 abnormal beats with timestamps and confidence.

## Notes

- Uploaded ECGs are kept in memory and are lost on restart. Use Redis or a database for persistence.
- Re-uploading a document with the same name replaces its chunks.
