# Documentation

| Document | For | Contents |
|---|---|---|
| [../README.md](../README.md) | everyone | what the project does, results, setup, running, testing, endpoints |
| [ARCHITECTURE.md](ARCHITECTURE.md) | developers, reviewers | components, agent/tool split, providers, request flows, ERD and data dictionary, design decisions, deployment |
| [API.md](API.md) | API users | every endpoint with real request/response examples, error codes, agent flow, Python client |
| [MODEL_CARD.md](MODEL_CARD.md) | anyone using the model's output | architecture, data and validation protocol, results with confidence intervals, limitations, history |
| [TESTING.md](TESTING.md) | developers, examiners | test strategy, suites, thresholds and how they were set, mutation testing, defects found |
| [ecg_reference.md](ecg_reference.md) | the agent | seed document for the RAG knowledge base: AAMI classes, normal intervals, ectopy, this model's limitations |
| [academic/](academic/) | SASTRA submission | synopsis and project report (.docx, .pdf), generator, submission checklist |
| [figures/](figures/) | report, docs | architecture, model, ERD, request flow, training curves, confusion matrix, calibration, baselines, ECG example (made by `scripts/make_figures.py`) |

Generated results live in [../reports/](../reports/): `evaluation.md` (accuracy), `performance.md` (speed,
calibration, error analysis) and `test_report.md` (all test cases). Changes are listed in [../CHANGELOG.md](../CHANGELOG.md).
