"""Inference wrapper around the LSTM beat classifier trained in notebooks/02_lstm_training.ipynb."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.ecg.preprocessing import (
    AAMI_DESCRIPTIONS,
    TARGET_FS,
    compute_rr_features,
    detect_r_peaks,
    prepare_signal,
    segment_beats,
    signal_quality,
)

logger = logging.getLogger(__name__)


class ECGClassifier:
    def __init__(self, model_path: Path, meta_path: Path):
        import keras  # imported lazily so the API can start without TensorFlow loaded

        if not model_path.exists():
            raise FileNotFoundError(
                f"ECG model not found at {model_path}. Run notebooks/02_lstm_training.ipynb first."
            )
        self.model = keras.models.load_model(model_path)
        self.meta = json.loads(meta_path.read_text())
        self.classes: list[str] = self.meta["classes"]
        logger.info("Loaded ECG LSTM model from %s (classes=%s)", model_path, self.classes)

    def predict_beats(self, beats: np.ndarray, rr: np.ndarray) -> np.ndarray:
        """beats: [n, BEAT_LEN] normalised, rr: [n, 4] -> probabilities [n, n_classes]."""
        if len(beats) == 0:
            return np.empty((0, len(self.classes)), dtype=np.float32)
        return self.model.predict([beats[..., None], rr], verbose=0)

    def analyze(self, sig: list[float] | np.ndarray, fs: float, r_peaks: list[int] | None = None) -> dict:
        """Full pipeline for a single-lead ECG strip: filter, detect beats, classify, summarise."""
        sig = np.asarray(sig, dtype=np.float32)
        duration_s = len(sig) / fs
        clean = prepare_signal(sig, fs)

        if r_peaks is not None:
            peaks = np.round(np.asarray(r_peaks) * TARGET_FS / fs).astype(int)
        else:
            peaks = detect_r_peaks(clean)

        beats, kept = segment_beats(clean, peaks)
        rr_feats = compute_rr_features(peaks)[kept]
        probs = self.predict_beats(beats, rr_feats)
        preds = probs.argmax(axis=1) if len(probs) else np.empty(0, dtype=int)

        counts = {c: int((preds == i).sum()) for i, c in enumerate(self.classes)}
        n_beats = int(len(preds))

        rr = np.diff(peaks) / TARGET_FS if len(peaks) > 1 else np.empty(0)
        hr = float(60.0 / rr.mean()) if len(rr) else None
        rr_sdnn_ms = float(rr.std() * 1000) if len(rr) > 1 else None

        abnormal = [
            {
                "beat_index": int(k),
                "time_s": round(float(peaks[k]) / TARGET_FS, 3),
                "label": self.classes[p],
                "confidence": round(float(probs[j, p]), 3),
            }
            for j, (k, p) in enumerate(zip(kept, preds))
            if self.classes[p] != "N"
        ]

        return {
            "duration_s": round(duration_s, 2),
            "n_beats_classified": n_beats,
            "heart_rate_bpm": round(hr, 1) if hr else None,
            "rr_sdnn_ms": round(rr_sdnn_ms, 1) if rr_sdnn_ms else None,
            "beat_counts": counts,
            "beat_percentages": {c: round(100 * v / n_beats, 2) if n_beats else 0.0 for c, v in counts.items()},
            "class_descriptions": {c: AAMI_DESCRIPTIONS[c] for c in self.classes},
            "abnormal_beats": abnormal[:50],
            "abnormal_beats_truncated": len(abnormal) > 50,
            "signal_quality": signal_quality(clean),
            "disclaimer": "Automated research model output; not a medical diagnosis.",
        }
