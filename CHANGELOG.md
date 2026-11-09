# Changelog

## 0.2.0 — 2026-11-09

### Agent and APIs
- OpenAI `gpt-4o-mini` is the agentic core (`LLM_PROVIDER=openai`, `OPENAI_API_KEY`); Gemini `gemini-2.5-flash`
  takes over automatically when the OpenAI call fails (`LLM_FALLBACK_PROVIDER=gemini`, `GEMINI_API_KEY`).
  `custom` provider for any OpenAI-compatible server (Ollama, vLLM, LM Studio).
- Agent and tools are separate: `app/tools.py` (schemas, argument validation, handlers) has no LLM code.
- New **Tools API**: `GET /tools`, `GET /tools/{name}`, `POST /tools/{name}/invoke`.
- New **Agent API**: `GET /agent`, `POST /agent/chat` with a step trace (LLM decisions, tool runs, timings, tokens)
  and the model that answered. `/chat` remains as an alias.
- `/health` reports the LLM in use.

### Model and analysis
- `signal_quality` block in every analysis (kurtosis-based); integration tests showed the beat detector reports
  "beats" in pure noise. The agent is told not to interpret flagged recordings.
- Probability metrics (ROC AUC, Brier, ECE, selective accuracy) and `scripts/benchmark_model.py`.

### Tests and documentation
- 119 tests: model performance (14), API integration with the real app (11) and live tests with the real OpenAI and
  Gemini APIs (11, opt-in with `RUN_LLM_TESTS=1`).
- `docs/API.md` API reference with real responses; `scripts/demo_agent.py`; `scripts/test_report.py`.
- Academic synopsis and project report (`docs/academic/`), figures and ERD (`docs/figures/`).

## 0.1.0 — 2026-11-09

- FastAPI service with persistent Chroma RAG (local MiniLM embeddings) and a tool-calling agent.
- Conv-BiLSTM beat classifier (N/S/V) trained on MIT-BIH with an inter-patient split; evaluated on 22 held-out
  patients with bootstrap confidence intervals, baselines and end-to-end detection.
- Notebooks for analysis, training and LLM tool use; evaluation report; model card; 64 tests.
