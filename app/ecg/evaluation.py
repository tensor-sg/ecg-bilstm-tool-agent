"""Model evaluation on MIT-BIH, shared by scripts/evaluate_model.py and the accuracy tests.

Two evaluation modes:
- annotated: beats cut at the cardiologist-annotated R-peaks (pure classifier performance).
- end_to_end: R-peaks found by our own detector, exactly as the API does on user ECGs;
  detected beats are matched to annotations within a tolerance window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from app.ecg import mitbih
from app.ecg.classifier import ECGClassifier
from app.ecg.preprocessing import (
    SYMBOL_TO_AAMI,
    TARGET_FS,
    bandpass,
    compute_rr_features,
    detect_r_peaks,
    segment_beats,
)

MATCH_TOLERANCE_S = 0.15  # AAMI EC57 beat-matching window


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """AAMI-style per-class sensitivity, positive predictivity, specificity, F1, plus overall scores."""
    return metrics_from_cm(confusion_matrix(y_true, y_pred, labels=range(len(classes))), classes)


def metrics_from_cm(cm: np.ndarray, classes: list[str]) -> dict:
    total = cm.sum()
    per_class = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        se = tp / (tp + fn) if tp + fn else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        per_class[c] = {
            "support": int(cm[i].sum()),
            "sensitivity": round(float(se), 4),
            "ppv": round(float(ppv), 4),
            "specificity": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
            "f1": round(float(2 * se * ppv / (se + ppv)) if se + ppv else 0.0, 4),
        }
    return {
        "n_beats": int(total),
        "accuracy": round(float(np.trace(cm) / total), 4) if total else 0.0,
        "macro_f1": round(float(np.mean([m["f1"] for m in per_class.values()])), 4),
        "balanced_accuracy": round(float(np.mean([m["sensitivity"] for m in per_class.values()])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(record_cms: list[np.ndarray], classes: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CIs by resampling *patients* (records) with replacement.

    Beats within one patient are strongly correlated, so a beat-level bootstrap would give
    intervals that are far too narrow.
    """
    rng = np.random.default_rng(seed)
    stack = np.stack(record_cms)
    samples: dict[str, list[float]] = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    per_class: dict[str, dict[str, list[float]]] = {c: {"sensitivity": [], "ppv": [], "f1": []} for c in classes}
    for _ in range(n_boot):
        m = metrics_from_cm(stack[rng.integers(0, len(stack), len(stack))].sum(0), classes)
        for k in samples:
            samples[k].append(m[k])
        for c in classes:
            for k in per_class[c]:
                per_class[c][k].append(m["per_class"][c][k])

    def ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    return {
        "n_boot": n_boot,
        "unit": "record",
        **{k: ci(v) for k, v in samples.items()},
        "per_class": {c: {k: ci(v) for k, v in d.items()} for c, d in per_class.items()},
    }


def probability_metrics(y_true: np.ndarray, proba: np.ndarray, classes: list[str], n_bins: int = 10) -> dict:
    """Quality of the softmax outputs: one-vs-rest ROC AUC, Brier score, calibration (ECE) and
    selective accuracy (accuracy on beats whose confidence clears a threshold)."""
    from sklearn.metrics import roc_auc_score

    conf, pred = proba.max(1), proba.argmax(1)
    correct = pred == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(m.sum()),
                         "confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(correct[m].mean()), 4)})
    onehot = np.eye(len(classes))[y_true]
    return {
        "roc_auc": {c: round(float(roc_auc_score(onehot[:, i], proba[:, i])), 4) for i, c in enumerate(classes)},
        "brier": round(float(((proba - onehot) ** 2).sum(1).mean()), 4),
        "ece": round(float(ece), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "reliability": bins,
        "selective": {
            str(t): {"coverage": round(float((conf >= t).mean()), 4), "accuracy": round(float(correct[conf >= t].mean()), 4)}
            for t in (0.5, 0.8, 0.9, 0.95)
        },
    }


def collect_predictions(clf: ECGClassifier, data_dir: Path, records: list[str]) -> dict:
    """Softmax outputs for every in-class beat at annotated peaks, with true labels and record ids."""
    ys, ps, recs = [], [], []
    for rec in records:
        d = mitbih.record_beats(data_dir, rec)
        keep, y = _labels_to_idx(d["labels"], clf.classes)
        if keep.any():
            ys.append(y)
            ps.append(clf.predict_beats(d["beats"][keep], d["rr"][keep]))
            recs.append(np.full(len(y), rec))
    return {"y_true": np.concatenate(ys), "proba": np.concatenate(ps), "record": np.concatenate(recs)}


def _labels_to_idx(labels: np.ndarray, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(labels, classes)
    return keep, np.array([classes.index(c) for c in labels[keep]], dtype=int)


def evaluate_annotated(clf: ECGClassifier, data_dir: Path, records: list[str]) -> dict:
    """Beat classification at annotated R-peaks, overall and per record."""
    y_true_all, y_pred_all, per_record, record_cms = [], [], {}, []
    excluded: dict[str, dict[str, int]] = {}
    for rec in records:
        d = mitbih.record_beats(data_dir, rec)
        keep, y_true = _labels_to_idx(d["labels"], clf.classes)
        if (~keep).any():
            # Beats of classes the model was not trained on (e.g. F, Q): record what they are predicted as.
            ex_pred = clf.predict_beats(d["beats"][~keep], d["rr"][~keep]).argmax(1)
            for label, p in zip(d["labels"][~keep], ex_pred):
                bucket = excluded.setdefault(str(label), {c: 0 for c in clf.classes})
                bucket[clf.classes[p]] += 1
        if keep.sum() == 0:
            continue
        y_pred = clf.predict_beats(d["beats"][keep], d["rr"][keep]).argmax(1)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        record_cms.append(confusion_matrix(y_true, y_pred, labels=range(len(clf.classes))))
        per_record[rec] = {
            "n_beats": int(len(y_true)),
            "accuracy": round(float((y_true == y_pred).mean()), 4),
            "true_counts": {c: int((y_true == i).sum()) for i, c in enumerate(clf.classes)},
            "pred_counts": {c: int((y_pred == i).sum()) for i, c in enumerate(clf.classes)},
        }
    metrics = classification_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all), clf.classes)
    metrics["ci95"] = bootstrap_ci(record_cms, clf.classes)
    metrics["excluded_classes_predicted_as"] = excluded
    metrics["per_record"] = per_record
    return metrics


def match_peaks(reference: np.ndarray, detected: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching; returns (ref_idx, det_idx) of matched pairs."""
    ref_idx, det_idx = [], []
    used = np.zeros(len(detected), dtype=bool)
    for i, r in enumerate(reference):
        j = np.searchsorted(detected, r)
        best, best_d = -1, tol + 1
        for k in (j - 1, j):
            if 0 <= k < len(detected) and not used[k] and abs(int(detected[k]) - int(r)) < best_d:
                best, best_d = k, abs(int(detected[k]) - int(r))
        if best >= 0:
            used[best] = True
            ref_idx.append(i)
            det_idx.append(best)
    return np.asarray(ref_idx, dtype=int), np.asarray(det_idx, dtype=int)


def evaluate_end_to_end(clf: ECGClassifier, data_dir: Path, records: list[str]) -> dict:
    """Full API pipeline: own R-peak detection, segmentation, RR features from detected peaks, classification."""
    tol = int(MATCH_TOLERANCE_S * TARGET_FS)
    n_ref = n_det = n_matched = 0
    y_true_all, y_pred_all, record_cms = [], [], []
    for rec in records:
        sig, samples, symbols, fs = mitbih.load_record(data_dir, rec)
        sig = bandpass(sig, fs)
        is_beat = np.isin(symbols, list(SYMBOL_TO_AAMI))
        ref, ref_labels = samples[is_beat], np.array([SYMBOL_TO_AAMI[s] for s in symbols[is_beat]])

        det = detect_r_peaks(sig, fs)
        rr = compute_rr_features(det, fs)
        beats, kept = segment_beats(sig, det)
        pred = clf.predict_beats(beats, rr[kept]).argmax(1) if len(kept) else np.empty(0, dtype=int)
        pred_by_det = dict(zip(kept.tolist(), pred.tolist()))

        ri, di = match_peaks(ref, det, tol)
        n_ref += len(ref)
        n_det += len(det)
        n_matched += len(ri)
        rec_true, rec_pred = [], []
        for r_i, d_i in zip(ri, di):
            if ref_labels[r_i] in clf.classes and d_i in pred_by_det:
                rec_true.append(clf.classes.index(ref_labels[r_i]))
                rec_pred.append(pred_by_det[d_i])
        y_true_all += rec_true
        y_pred_all += rec_pred
        if rec_true:
            record_cms.append(confusion_matrix(rec_true, rec_pred, labels=range(len(clf.classes))))

    metrics = classification_metrics(np.asarray(y_true_all), np.asarray(y_pred_all), clf.classes)
    metrics["ci95"] = bootstrap_ci(record_cms, clf.classes)
    metrics["detection"] = {
        "reference_beats": n_ref,
        "detected_beats": n_det,
        "matched": n_matched,
        "sensitivity": round(n_matched / n_ref, 4) if n_ref else 0.0,
        "ppv": round(n_matched / n_det, 4) if n_det else 0.0,
        "tolerance_ms": int(MATCH_TOLERANCE_S * 1000),
    }
    return metrics
