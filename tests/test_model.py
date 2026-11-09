"""Tests against the trained model and real MIT-BIH data (skipped if either is missing).

All checks use DS2 test patients only, never the training or validation patients.
"""

import json

import numpy as np
import pytest
from scipy import signal as sps

from app.ecg import mitbih
from app.ecg.evaluation import evaluate_annotated, evaluate_end_to_end
from app.ecg.preprocessing import BEAT_LEN, RR_FEATURES, SYMBOL_TO_AAMI, TARGET_FS
from tests.conftest import DATA_DIR, META_PATH, requires_model

pytestmark = [pytest.mark.model, requires_model]

# Regression floors for the shipped model on DS2. They sit a few points below the measured values
# (reports/evaluation.md), so a broken preprocessing change or a worse retrain fails loudly.
# Measured (2026-11-09): accuracy 0.945, macro-F1 0.650, Se_N 0.991, Se_V 0.802, PPV_V 0.847; detection Se 0.979, PPV 0.998.
FLOORS_ANNOTATED = {"accuracy": 0.92, "macro_f1": 0.60, "Se_N": 0.97, "Se_V": 0.75, "PPV_V": 0.78}
FLOORS_DETECTION = {"sensitivity": 0.97, "ppv": 0.99}


def strip(record, seconds=None):
    sig, samples, symbols, fs = mitbih.load_record(DATA_DIR, record)
    if seconds:
        n = int(seconds * fs)
        keep = samples < n
        return sig[:n], samples[keep], symbols[keep], fs
    return sig, samples, symbols, fs


@pytest.fixture(scope="module")
def ds2_annotated(trained_classifier):
    return evaluate_annotated(trained_classifier, DATA_DIR, mitbih.DS2)


# ---------------------------------------------------------------- metadata / artefact
def test_meta_matches_preprocessing_and_has_no_patient_leakage():
    meta = json.loads(META_PATH.read_text())
    assert meta["beat_len"] == BEAT_LEN and meta["fs"] == TARGET_FS and meta["rr_features"] == RR_FEATURES
    if meta["split"] == "inter_patient":
        tr, va, te = map(set, (meta["train_records"], meta["val_records"], meta["test_records"]))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert te == set(mitbih.DS2)


def test_predictions_are_deterministic_probabilities(trained_classifier):
    d = mitbih.record_beats(DATA_DIR, "100")
    p1 = trained_classifier.predict_beats(d["beats"][:300], d["rr"][:300])
    p2 = trained_classifier.predict_beats(d["beats"][:300], d["rr"][:300])
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(p1.sum(1), 1.0, atol=1e-4)
    assert p1.shape == (300, len(trained_classifier.classes))


# ---------------------------------------------------------------- consistency of the API path
def test_api_pipeline_matches_training_pipeline(trained_classifier):
    """analyze() on a raw strip with annotated peaks must reproduce the per-beat predictions of the training pipeline."""
    sig, samples, symbols, fs = strip("233", seconds=300)
    peaks = samples[np.isin(symbols, list(SYMBOL_TO_AAMI))]
    api = trained_classifier.analyze(sig, fs, r_peaks=peaks.tolist())

    full = mitbih.record_beats(DATA_DIR, "233")
    in_strip = np.isin(full["peak"], peaks)
    ref = trained_classifier.predict_beats(full["beats"][in_strip], full["rr"][in_strip]).argmax(1)
    ref_counts = {c: int((ref == i).sum()) for i, c in enumerate(trained_classifier.classes)}

    total = sum(ref_counts.values())
    diff = sum(abs(api["beat_counts"][c] - ref_counts[c]) for c in ref_counts)
    assert abs(api["n_beats_classified"] - total) <= 2      # only edge beats may differ
    assert diff / total <= 0.02


def test_regular_rhythm_record(trained_classifier):
    """Record 100 (DS2): sinus rhythm at ~75 bpm, 2,239 N vs 34 abnormal beats."""
    sig, _, _, fs = strip("100", seconds=600)
    res = trained_classifier.analyze(sig, fs)
    assert 65 <= res["heart_rate_bpm"] <= 85
    assert res["beat_percentages"]["N"] >= 90


def test_detects_ventricular_ectopy(trained_classifier):
    """Record 233 (DS2) has ~25% PVCs; the model must report a substantial V burden."""
    sig, samples, symbols, fs = strip("233", seconds=300)
    true_v = sum(SYMBOL_TO_AAMI.get(s) == "V" for s in symbols)
    res = trained_classifier.analyze(sig, fs)
    assert res["beat_counts"]["V"] >= 0.7 * true_v
    assert res["abnormal_beats"], "abnormal beats should be listed with timestamps"


# ---------------------------------------------------------------- robustness
def _counts(res):
    return np.array(list(res["beat_counts"].values()), dtype=float)


def test_invariant_to_gain_and_dc_offset(trained_classifier):
    sig, _, _, fs = strip("233", seconds=120)
    a = trained_classifier.analyze(sig, fs)
    b = trained_classifier.analyze(sig * 3.0 + 0.5, fs)
    np.testing.assert_allclose(_counts(a), _counts(b), atol=1)


def test_robust_to_lower_sampling_rate(trained_classifier):
    """Devices commonly record at 250 Hz; resampled input should give nearly the same beat counts."""
    sig, _, _, fs = strip("233", seconds=300)
    sig250 = sps.resample(sig, int(len(sig) * 250 / fs))
    a, b = trained_classifier.analyze(sig, fs), trained_classifier.analyze(sig250, 250)
    assert abs(a["n_beats_classified"] - b["n_beats_classified"]) <= 0.02 * a["n_beats_classified"]
    assert abs(a["beat_counts"]["V"] - b["beat_counts"]["V"]) <= 0.1 * max(a["beat_counts"]["V"], 1)
    assert abs(a["heart_rate_bpm"] - b["heart_rate_bpm"]) <= 2


def test_robust_to_moderate_noise(trained_classifier):
    sig, _, _, fs = strip("233", seconds=300)
    noise = np.random.default_rng(0).standard_normal(len(sig)) * sig.std() / 10 ** (25 / 20)   # 25 dB SNR
    a, b = trained_classifier.analyze(sig, fs), trained_classifier.analyze(sig + noise, fs)
    assert abs(a["beat_counts"]["V"] - b["beat_counts"]["V"]) <= 0.1 * max(a["beat_counts"]["V"], 1)
    assert abs(a["heart_rate_bpm"] - b["heart_rate_bpm"]) <= 2


def test_all_test_patients_pass_signal_quality():
    """The quality warning must not fire on real (non-paced) recordings: all 22 DS2 records pass."""
    from app.ecg.preprocessing import prepare_signal, signal_quality

    flagged = {}
    for rec in mitbih.DS2:
        sig, _, _, fs = mitbih.load_record(DATA_DIR, rec)
        q = signal_quality(prepare_signal(sig, fs))
        if not q["ok"]:
            flagged[rec] = q
    assert not flagged, flagged


def test_short_and_flat_inputs_do_not_crash(trained_classifier):
    assert trained_classifier.analyze(np.zeros(3600, dtype=np.float32), 360)["n_beats_classified"] >= 0
    sig, _, _, fs = strip("100", seconds=3)
    assert trained_classifier.analyze(sig, fs)["duration_s"] == 3.0


# ---------------------------------------------------------------- accuracy on held-out patients
def test_ds2_accuracy_floors(ds2_annotated):
    m, pc = ds2_annotated, ds2_annotated["per_class"]
    observed = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "Se_N": pc["N"]["sensitivity"],
                "Se_V": pc["V"]["sensitivity"], "PPV_V": pc["V"]["ppv"]}
    failures = {k: (observed[k], floor) for k, floor in FLOORS_ANNOTATED.items() if observed[k] < floor}
    assert not failures, f"below regression floor (observed, floor): {failures}"


def test_beats_majority_and_rr_only_baselines(ds2_annotated):
    baselines = json.loads(META_PATH.read_text())["baselines_test"]
    assert ds2_annotated["macro_f1"] > baselines["majority (always N)"]["macro_f1"]
    assert ds2_annotated["macro_f1"] > baselines["RR-only logistic regression"]["macro_f1"]


def test_end_to_end_detection_and_classification(trained_classifier):
    m = evaluate_end_to_end(trained_classifier, DATA_DIR, mitbih.DS2)
    for k, floor in FLOORS_DETECTION.items():
        assert m["detection"][k] >= floor, f"detection {k}={m['detection'][k]} < {floor}"
    assert m["per_class"]["V"]["sensitivity"] >= FLOORS_ANNOTATED["Se_V"] - 0.1
