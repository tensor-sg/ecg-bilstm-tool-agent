# Testing

How the system is tested, what each suite covers, how thresholds were chosen, and what testing found.
Latest results: [reports/test_report.md](../reports/test_report.md), with every test case listed:
**119 passed, 0 failed, 0 skipped**.

## 1. Running the tests

```bash
pytest -m "not model"                    # 70 fast tests, ~10 s: no trained model, data, network or API key needed
pytest                                   # 108 tests, ~2.5 min: adds model, performance and integration tests
RUN_LLM_TESTS=1 pytest -m llm            # 11 live tests with the real OpenAI and Gemini APIs, ~1 min, uses API credit
RUN_LLM_TESTS=1 python scripts/test_report.py   # all 119 -> reports/test_report.md and reports/junit.xml
```

Markers (`pytest.ini`): `model` needs `models/` and `data/mitdb/` (run notebooks 01 and 02 first; skipped
otherwise), `integration` runs the real app, `llm` calls paid APIs and is skipped unless `RUN_LLM_TESTS=1`.

## 2. Strategy

| Level | What is real | What is replaced | Why |
|---|---|---|---|
| Unit and API | code under test, FastAPI routing and validation | LLM, vector store and classifier by fakes | fast, deterministic, runs anywhere |
| Component | Chroma on a temporary directory | embeddings by a hash embedder | tests persistence and ranking offline |
| Model and performance | trained LSTM, MIT-BIH data | nothing | accuracy, robustness, speed and calibration on real patients |
| Integration | whole app with its lifespan: model, Chroma, local embeddings | LLM by a scripted fake | tests the deployed wiring without API cost |
| Live | everything, including OpenAI and Gemini | nothing | proves the agent really calls the tools and uses their output |

Model, integration and live tests use **only the 22 DS2 test patients**, never the training or validation
patients. The scripted LLM in offline tests issues real tool calls that run against real code.

## 3. Suites

| File | Tests | Covers |
|---|---:|---|
| `test_preprocessing.py` | 25 | resampling, filter attenuation, R-peak detection on synthetic ECG (60–120 bpm, 250 Hz), heart-rate-independent RR features, segmentation edges, signal quality (accepts ECG, rejects noise/sine/flat, scale-invariant) |
| `test_ingest.py` | 5 | text extraction (txt, md, pdf), chunk size and overlap |
| `test_vector_store.py` | 5 | real Chroma: ranking, persistence across restart, upsert without duplicates, source filter, delete, embedding-model mismatch guard |
| `test_agent.py` | 15 | tool loop, results fed back, sources, step trace, argument validation, unknown tools, missing model, step limit, provider resolution, missing keys, Gemini fallback, both providers failing, temperature |
| `test_api.py` | 20 | every endpoint with fakes: tools catalogue and invocation (404/422/503), agent config and trace, `/chat` alias, ECG and document endpoints, validation errors, 502 on LLM failure |
| `test_model.py` | 13 | no patient leakage, determinism, API path equals training path, gain/offset invariance, 250 Hz input, 25 dB noise, signal quality on all DS2 records, accuracy floors, beats baselines, end-to-end detection |
| `test_model_performance.py` | 14 | size, throughput, 30-min record < 30 s, batch-size invariance, memory leak, calibration (ECE, Brier), ROC AUC, selective accuracy, false V alarms, V specificity, S detection without record 232, per-patient V sensitivity, bootstrap CI, paired bootstrap vs RR baseline |
| `test_api_integration.py` | 11 | real lifespan; ingest and semantic search with real embeddings; persistence across restart; 250 Hz CSV vs cardiologist labels; supplied R-peaks; response contract; latency; noise flagged; agent tool calls on real services |
| `test_llm_live.py` | 11 | real OpenAI agent: quotes the LSTM's numbers, states it is not a diagnosis, flags noise, lists recordings, multi-turn follow-up, RAG cites sources, both tools in one question, direct tool call equals agent's, full trace with tokens; Gemini as main agent and as fallback |

## 4. Thresholds

Accuracy and performance tests compare against fixed floors set **a few points below the measured values**, so they
fail on real regressions (a preprocessing bug, a worse retrain, a slower pipeline) but not on normal variation
between machines or runs.

| Measure (DS2) | Measured | Floor in tests |
|---|---:|---:|
| Accuracy | 0.945 | 0.92 |
| Macro-F1 | 0.650 | 0.60 |
| N sensitivity | 0.991 | 0.97 |
| V sensitivity / PPV | 0.802 / 0.847 | 0.75 / 0.78 |
| Beat detection sensitivity / PPV | 0.979 / 0.998 | 0.97 / 0.99 |
| ECE / Brier | 0.013 / 0.092 | < 0.05 / < 0.15 |
| ROC AUC V / N / S | 0.984 / 0.926 / 0.764 | 0.95 / 0.88 / 0.70 |
| Normal beats called V | 0.33% | < 1% |
| S sensitivity without record 232 | 0.382 | 0.30 |
| Batched inference | ≈ 0.5 ms/beat | < 5 ms/beat |
| 30-minute record, full pipeline | ≈ 1.3 s | < 30 s |
| 5-minute strip over HTTP | ≈ 0.45 s | < 5 s |

Live LLM tests never compare wording. They check which tools were called with which arguments, that cited sources
exist, and that numbers in the answer match the classifier's output. They run at temperature 0.

## 5. Mutation testing

Bugs were injected deliberately to check that the tests catch them; every one was caught.

| Injected bug | Tests that failed |
|---|---|
| RR features passed in the wrong order | RR feature unit tests |
| Similarity score = distance instead of 1 − distance | vector store ranking |
| Tool result not returned to the LLM | agent loop |
| API path ignores RR features | 2 model tests |
| Signal-quality check disabled | 3 unit + 1 integration |
| RR features replaced by zeros at inference | 7 of 14 performance tests |

## 6. Defects found by testing

| Found by | Defect | Fix |
|---|---|---|
| validation design review | model v1 validation set contained training patients | patient-wise validation split |
| held-out evaluation | model v2 labelled 11% of normal beats as fusion | fusion removed from trained classes (v3) |
| API smoke test | invalid LLM key gave HTTP 500 | 502 with a configuration hint |
| integration test | 20 s of pure noise produced 39 "beats" | kurtosis signal-quality check; agent told not to interpret flagged results |
| consistency test | training loader dropped edge beats without recording which | R-peak of each kept beat recorded |
