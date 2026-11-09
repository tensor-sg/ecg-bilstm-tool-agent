"""Performance benchmark of the trained ECG LSTM: speed, memory, probability quality and error analysis.

Usage:
    python scripts/benchmark_model.py

Writes reports/performance.json and reports/performance.md. Complements scripts/evaluate_model.py,
which reports classification accuracy.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_API_KEY", "unused")

import psutil  # noqa: E402

from app.ecg import mitbih  # noqa: E402
from app.ecg.classifier import ECGClassifier  # noqa: E402
from app.ecg.evaluation import collect_predictions, metrics_from_cm, probability_metrics  # noqa: E402

DATA_DIR = ROOT / "data" / "mitdb"
MODEL_PATH = ROOT / "models" / "ecg_lstm.keras"
META_PATH = ROOT / "models" / "ecg_lstm_meta.json"
OUT = ROOT / "reports"


def hardware() -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")),
               platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    import tensorflow as tf

    return {"cpu": cpu, "logical_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "gpu": bool(tf.config.list_physical_devices("GPU")), "python": platform.python_version(), "tensorflow": tf.__version__}


def timed(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def speed(clf: ECGClassifier) -> dict:
    d = mitbih.record_beats(DATA_DIR, "100")
    clf.predict_beats(d["beats"][:8], d["rr"][:8])  # warm-up (graph tracing)
    batches = {}
    for n in (1, 64, 512, 2048):
        t = np.median(timed(lambda: clf.predict_beats(d["beats"][:n], d["rr"][:n]), 5 if n > 1 else 30))
        batches[str(n)] = {"latency_ms": round(t * 1000, 2), "ms_per_beat": round(t * 1000 / n, 3), "beats_per_s": round(n / t)}

    sig, _, _, fs = mitbih.load_record(DATA_DIR, "100")
    t = np.median(timed(lambda: clf.analyze(sig, fs), 3))
    rec_s = len(sig) / fs
    return {
        "predict_beats": batches,
        "analyze_full_record": {"record": "100", "signal_s": round(rec_s, 1), "wall_s": round(float(t), 2),
                                "faster_than_real_time": round(rec_s / t), "process_rss_mb": round(psutil.Process().memory_info().rss / 1e6)},
    }


def api_latency(clf: ECGClassifier) -> dict:
    """POST /ecg/upload + /ecg/{id}/analyze with the real model, through FastAPI's ASGI stack."""
    from fastapi.testclient import TestClient

    from app.agent import Agent, ECGRegistry
    from app.config import get_settings
    from app.deps import AppState
    from app.main import app

    settings = get_settings()
    registry = ECGRegistry()
    app.state.services = AppState(settings=settings, store=None, registry=registry, classifier=clf,
                                  agent=Agent(settings, None, registry, clf))
    client = TestClient(app)
    sig, _, _, fs = mitbih.load_record(DATA_DIR, "233")
    out = {}
    for minutes in (1, 5):
        payload = {"signal": sig[: int(minutes * 60 * fs)].tolist(), "fs": fs}

        def call():
            ecg_id = client.post("/ecg/upload", json=payload).json()["ecg_id"]
            assert client.post(f"/ecg/{ecg_id}/analyze").status_code == 200

        call()
        ts = timed(call, 5)
        out[f"{minutes}_min_strip"] = {"median_ms": round(float(np.median(ts)) * 1000), "max_ms": round(max(ts) * 1000)}
    return out


def error_analysis(pred: dict, classes: list[str]) -> dict:
    y, p, rec = pred["y_true"], pred["proba"].argmax(1), pred["record"]
    n_cls = len(classes)

    def cm(mask):
        return np.bincount(y[mask] * n_cls + p[mask], minlength=n_cls * n_cls).reshape(n_cls, n_cls)

    s, v = classes.index("S"), classes.index("V")
    without_232 = metrics_from_cm(cm(rec != "232"), classes)
    per_rec_v = {r: round(float((p[(rec == r) & (y == v)] == v).mean()), 3)
                 for r in np.unique(rec) if ((rec == r) & (y == v)).sum() >= 50}
    return {
        "S_beats_in_232": int(((rec == "232") & (y == s)).sum()),
        "S_beats_total": int((y == s).sum()),
        "without_record_232": {"accuracy": without_232["accuracy"], "macro_f1": without_232["macro_f1"],
                               "S": without_232["per_class"]["S"], "V": without_232["per_class"]["V"]},
        "normal_called_V_rate": round(float(((y == 0) & (p == v)).sum() / (y == 0).sum()), 4),
        "V_sensitivity_per_record": per_rec_v,
    }


def to_markdown(r: dict) -> str:
    hw, sp, api, pq, ea = r["hardware"], r["speed"], r["api"], r["probability"], r["errors"]
    batch_rows = "\n".join(f"| {n} | {v['latency_ms']} | {v['ms_per_beat']} | {v['beats_per_s']:,} |" for n, v in sp["predict_beats"].items())
    full = sp["analyze_full_record"]
    rel = "\n".join(f"| {b['lo']:.1f}–{b['hi']:.1f} | {b['n']:,} | {b['confidence']:.3f} | {b['accuracy']:.3f} |" for b in pq["reliability"])
    sel = "\n".join(f"| ≥ {t} | {v['coverage']:.3f} | {v['accuracy']:.3f} |" for t, v in pq["selective"].items())
    vrec = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(ea["V_sensitivity_per_record"].items(), key=lambda kv: kv[1]))
    w = ea["without_record_232"]
    return f"""# ECG LSTM performance benchmark

Generated by `scripts/benchmark_model.py` on {r['generated_at']}.
Hardware: {hw['cpu']}, {hw['logical_cores']} logical cores, {hw['ram_gb']} GB RAM, GPU: {'yes' if hw['gpu'] else 'no'} ·
Python {hw['python']} · TensorFlow {hw['tensorflow']}.
Model: {r['model']['params']:,} parameters, {r['model']['file_mb']} MB on disk.

## 1. Inference speed

`ECGClassifier.predict_beats` (median of repeated runs, after warm-up):

| Batch size | Latency (ms) | ms per beat | Beats per second |
|---:|---:|---:|---:|
{batch_rows}

Full pipeline `ECGClassifier.analyze` on record {full['record']} ({full['signal_s'] / 60:.0f} min, filtering, R-peak detection,
segmentation and classification): **{full['wall_s']} s**, about **{full['faster_than_real_time']:,}× faster than real time**,
whole process (Python, TensorFlow and model) using {full['process_rss_mb']} MB of RAM.

## 2. API latency

`POST /ecg/upload` followed by `POST /ecg/{{id}}/analyze` through the FastAPI ASGI stack with the real model (5 runs):

| Strip | Median (ms) | Max (ms) |
|---|---:|---:|
| 1 minute | {api['1_min_strip']['median_ms']} | {api['1_min_strip']['max_ms']} |
| 5 minutes | {api['5_min_strip']['median_ms']} | {api['5_min_strip']['max_ms']} |

## 3. Quality of the predicted probabilities (DS2, {r['n_beats']:,} beats)

ROC AUC (one-vs-rest): {', '.join(f'{c} **{v:.3f}**' for c, v in pq['roc_auc'].items())} ·
Brier score **{pq['brier']:.4f}** · expected calibration error (ECE, 10 bins) **{pq['ece']:.4f}** ·
mean confidence {pq['mean_confidence']:.3f}.

Reliability (confidence of the predicted class vs observed accuracy):

| Confidence bin | Beats | Mean confidence | Accuracy |
|---|---:|---:|---:|
{rel}

Selective prediction (only beats whose confidence clears a threshold):

| Threshold | Coverage | Accuracy |
|---|---:|---:|
{sel}

## 4. Error analysis

- {ea['S_beats_in_232']:,} of the {ea['S_beats_total']:,} S beats in DS2 come from record 232.
  Without that record: accuracy {w['accuracy']:.3f}, macro-F1 {w['macro_f1']:.3f},
  S sensitivity {w['S']['sensitivity']:.3f} (PPV {w['S']['ppv']:.3f}), V sensitivity {w['V']['sensitivity']:.3f}.
- Normal beats wrongly called V: {ea['normal_called_V_rate'] * 100:.2f}%.
- V sensitivity per record (records with ≥ 50 V beats): {vrec}.
"""


def main() -> None:
    clf = ECGClassifier(MODEL_PATH, META_PATH)
    pred = collect_predictions(clf, DATA_DIR, mitbih.DS2)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "hardware": hardware(),
        "model": {"params": int(clf.model.count_params()), "file_mb": round(MODEL_PATH.stat().st_size / 1e6, 2)},
        "speed": speed(clf),
        "api": api_latency(clf),
        "n_beats": int(len(pred["y_true"])),
        "probability": probability_metrics(pred["y_true"], pred["proba"], clf.classes),
        "errors": error_analysis(pred, clf.classes),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "performance.json").write_text(json.dumps(report, indent=2))
    (OUT / "performance.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
