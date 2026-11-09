"""Performance tests of the trained model: speed, resources, probability quality, statistical
significance and per-class error rates. DS2 test patients only.

Thresholds sit below the values measured in reports/performance.md (Intel i5-7200U, CPU only),
so they fail on real regressions, not on normal machine-to-machine variation.
"""

import json
import time

import numpy as np
import pytest

from app.ecg import mitbih
from app.ecg.evaluation import collect_predictions, metrics_from_cm, probability_metrics
from tests.conftest import DATA_DIR, META_PATH, MODEL_PATH, requires_model

pytestmark = [pytest.mark.model, requires_model]


@pytest.fixture(scope="module")
def ds2(trained_classifier):
    return collect_predictions(trained_classifier, DATA_DIR, mitbih.DS2)


@pytest.fixture(scope="module")
def ds2_proba_metrics(ds2, trained_classifier):
    return probability_metrics(ds2["y_true"], ds2["proba"], trained_classifier.classes)


def _cm(y, p, n):
    return np.bincount(y * n + p, minlength=n * n).reshape(n, n)


# ---------------------------------------------------------------- size and speed
def test_model_is_small_enough_for_cpu_deployment(trained_classifier):
    assert trained_classifier.model.count_params() < 100_000          # measured 49,571
    assert MODEL_PATH.stat().st_size < 2_000_000                      # measured 0.71 MB


def test_batched_inference_throughput(trained_classifier):
    d = mitbih.record_beats(DATA_DIR, "100")
    beats, rr = d["beats"][:2048], d["rr"][:2048]
    trained_classifier.predict_beats(beats[:8], rr[:8])               # warm-up
    t = time.perf_counter()
    trained_classifier.predict_beats(beats, rr)
    per_beat_ms = (time.perf_counter() - t) * 1000 / len(beats)
    assert per_beat_ms < 5, f"{per_beat_ms:.2f} ms/beat (measured ~0.7)"


def test_full_record_analysed_much_faster_than_real_time(trained_classifier):
    """A 30-minute Holter record must be analysed in well under a minute (measured ~1.5 s)."""
    sig, _, _, fs = mitbih.load_record(DATA_DIR, "100")
    t = time.perf_counter()
    res = trained_classifier.analyze(sig, fs)
    elapsed = time.perf_counter() - t
    assert res["n_beats_classified"] > 2000
    assert elapsed < 30, f"{elapsed:.1f} s for {len(sig) / fs / 60:.0f} min of ECG"


def test_predictions_do_not_depend_on_batch_size(trained_classifier):
    d = mitbih.record_beats(DATA_DIR, "233")
    batched = trained_classifier.predict_beats(d["beats"][:40], d["rr"][:40])
    single = np.concatenate([trained_classifier.predict_beats(d["beats"][i:i + 1], d["rr"][i:i + 1]) for i in range(40)])
    np.testing.assert_allclose(batched, single, atol=1e-4)


def test_memory_stays_bounded_on_repeated_analysis(trained_classifier):
    """No leak: analysing the same record repeatedly must not keep growing the process."""
    psutil = pytest.importorskip("psutil")
    sig, _, _, fs = mitbih.load_record(DATA_DIR, "233")
    sig = sig[: int(300 * fs)]
    proc = psutil.Process()
    trained_classifier.analyze(sig, fs)
    before = proc.memory_info().rss
    for _ in range(10):
        trained_classifier.analyze(sig, fs)
    assert (proc.memory_info().rss - before) / 1e6 < 200


# ---------------------------------------------------------------- probability quality
def test_probabilities_are_well_calibrated(ds2_proba_metrics):
    """Confidence should mean what it says: ECE measured 0.013, Brier 0.092."""
    assert ds2_proba_metrics["ece"] < 0.05
    assert ds2_proba_metrics["brier"] < 0.15


def test_roc_auc_per_class(ds2_proba_metrics):
    auc = ds2_proba_metrics["roc_auc"]                                  # measured N 0.926, S 0.764, V 0.984
    assert auc["V"] >= 0.95
    assert auc["N"] >= 0.88
    assert auc["S"] >= 0.70


def test_higher_confidence_means_higher_accuracy(ds2_proba_metrics):
    sel = ds2_proba_metrics["selective"]
    accs = [sel[t]["accuracy"] for t in ("0.5", "0.8", "0.9", "0.95")]
    assert accs == sorted(accs), f"selective accuracy not monotonic: {accs}"
    assert sel["0.9"]["coverage"] >= 0.85 and sel["0.9"]["accuracy"] >= 0.95


# ---------------------------------------------------------------- error rates that matter clinically
def test_few_normal_beats_raised_as_ventricular(ds2, trained_classifier):
    """False V alarms on normal beats would flood a report; measured 0.33%."""
    v = trained_classifier.classes.index("V")
    y, p = ds2["y_true"], ds2["proba"].argmax(1)
    assert ((y == 0) & (p == v)).sum() / (y == 0).sum() < 0.01


def test_v_specificity(ds2, trained_classifier):
    m = metrics_from_cm(_cm(ds2["y_true"], ds2["proba"].argmax(1), 3), trained_classifier.classes)
    assert m["per_class"]["V"]["specificity"] >= 0.98                  # measured 0.990


def test_s_detection_outside_record_232(ds2, trained_classifier):
    """S sensitivity is low overall because of one patient; on the remaining 21 it must stay usable (measured 0.38)."""
    keep = ds2["record"] != "232"
    m = metrics_from_cm(_cm(ds2["y_true"][keep], ds2["proba"][keep].argmax(1), 3), trained_classifier.classes)
    assert m["per_class"]["S"]["sensitivity"] >= 0.30
    assert m["macro_f1"] >= 0.68                                       # measured 0.740


def test_v_detected_in_most_patients(ds2, trained_classifier):
    """V sensitivity per patient with ≥ 50 V beats: at least 6 of 8 above 0.7, none below 0.3."""
    v = trained_classifier.classes.index("V")
    y, p, rec = ds2["y_true"], ds2["proba"].argmax(1), ds2["record"]
    se = [float((p[(rec == r) & (y == v)] == v).mean()) for r in np.unique(rec) if ((rec == r) & (y == v)).sum() >= 50]
    assert len(se) >= 6
    assert sum(s >= 0.7 for s in se) >= 0.75 * len(se), sorted(se)
    assert min(se) >= 0.3, sorted(se)


# ---------------------------------------------------------------- statistical significance
def test_macro_f1_ci_lower_bound_beats_baselines():
    """The patient-bootstrap 95% CI must sit entirely above the majority baseline."""
    meta = json.loads(META_PATH.read_text())
    lo, _ = meta["test_ci95"]["macro_f1"]
    assert lo > meta["baselines_test"]["majority (always N)"]["macro_f1"]


def test_paired_bootstrap_vs_rr_baseline(ds2, trained_classifier):
    """Paired patient-level bootstrap: the LSTM's macro-F1 beats the RR-only logistic regression
    in at least 90% of resamples (the baseline is refit on the training patients)."""
    from sklearn.linear_model import LogisticRegression

    meta = json.loads(META_PATH.read_text())
    classes = trained_classifier.classes

    def rr_and_labels(records):
        xs, ys = [], []
        for r in records:
            d = mitbih.record_beats(DATA_DIR, r)
            keep = np.isin(d["labels"], classes)
            xs.append(d["rr"][keep])
            ys.append([classes.index(c) for c in d["labels"][keep]])
        return np.concatenate(xs), np.concatenate(ys)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(*rr_and_labels(meta["train_records"]))
    # collect_predictions walks mitbih.DS2 in order, so the beats line up with rr_and_labels(DS2).
    rr_te, y_te = rr_and_labels(mitbih.DS2)
    y, rec = ds2["y_true"], ds2["record"]
    assert (y_te == y).all()
    p_lstm, p_lr = ds2["proba"].argmax(1), lr.predict(rr_te)

    records = np.unique(rec)
    cms_lstm = np.stack([_cm(y[rec == r], p_lstm[rec == r], 3) for r in records])
    cms_lr = np.stack([_cm(y[rec == r], p_lr[rec == r], 3) for r in records])
    rng = np.random.default_rng(0)
    wins = 0
    for _ in range(500):
        idx = rng.integers(0, len(records), len(records))
        wins += metrics_from_cm(cms_lstm[idx].sum(0), classes)["macro_f1"] > metrics_from_cm(cms_lr[idx].sum(0), classes)["macro_f1"]
    assert wins / 500 >= 0.9, f"LSTM better in only {wins / 500:.0%} of resamples"
