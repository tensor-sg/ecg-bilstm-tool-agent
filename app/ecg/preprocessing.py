"""ECG preprocessing shared by the training notebooks and the API.

Keeping this in one module guarantees the beats the model sees at inference
time are built exactly like the beats it was trained on.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

# MIT-BIH Arrhythmia Database is sampled at 360 Hz.
TARGET_FS = 360
# Beat window around the R-peak (samples at TARGET_FS): 0.25 s before, ~0.46 s after.
WINDOW_BEFORE = 90
WINDOW_AFTER = 166
BEAT_LEN = WINDOW_BEFORE + WINDOW_AFTER  # 256

# AAMI EC57 grouping of MIT-BIH beat annotation symbols.
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_DESCRIPTIONS = {
    "N": "Normal / bundle branch block / escape beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion of ventricular and normal beat",
    "Q": "Paced / unclassifiable beat",
}
SYMBOL_TO_AAMI = {
    **{s: "N" for s in ["N", "L", "R", "e", "j"]},
    **{s: "S" for s in ["A", "a", "J", "S"]},
    **{s: "V" for s in ["V", "E"]},
    "F": "F",
    **{s: "Q" for s in ["/", "f", "Q"]},
}


def resample(sig: np.ndarray, fs: float, target_fs: int = TARGET_FS) -> np.ndarray:
    """Resample a 1-D signal to target_fs."""
    if int(round(fs)) == target_fs:
        return np.asarray(sig, dtype=np.float32)
    n_out = int(round(len(sig) * target_fs / fs))
    return sps.resample(sig, n_out).astype(np.float32)


def bandpass(sig: np.ndarray, fs: int = TARGET_FS, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Zero-phase Butterworth band-pass: removes baseline wander and high-frequency noise."""
    b, a = sps.butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, sig).astype(np.float32)


def detect_r_peaks(sig: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Simple Pan-Tompkins-style R-peak detector for signals without annotations."""
    diff = np.diff(sig, prepend=sig[0])
    energy = diff**2
    win = int(0.15 * fs)
    integrated = np.convolve(energy, np.ones(win) / win, mode="same")
    peaks, _ = sps.find_peaks(
        integrated,
        height=np.mean(integrated) + 0.5 * np.std(integrated),
        distance=int(0.25 * fs),  # refractory period, caps HR at 240 bpm
    )
    # Refine each candidate to the local maximum of |signal| within +/-50 ms.
    search = int(0.05 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - search), min(len(sig), p + search)
        refined.append(lo + int(np.argmax(np.abs(sig[lo:hi]))))
    return np.unique(np.asarray(refined, dtype=int))


QUALITY_KURTOSIS_MIN = 1.5
QUALITY_MAX_BAD_FRACTION = 0.5


def signal_quality(sig: np.ndarray, fs: int = TARGET_FS, window_s: float = 10.0) -> dict:
    """Kurtosis-based signal quality index on the filtered signal.

    QRS complexes make a clean ECG strongly peaked (excess kurtosis ~10 on MIT-BIH), while noise,
    flat lines and periodic artefacts sit near or below 0. Each 10 s window under the threshold
    counts as low quality; filter edge transients (0.5 s each side) are ignored. On MIT-BIH the
    threshold flags ~3% of 10 s windows (ventricular flutter, heavy noise) and 0% of Gaussian noise passes.
    """
    from scipy.stats import kurtosis

    sig = np.asarray(sig, dtype=np.float64)
    edge = int(0.5 * fs)
    if len(sig) > 4 * edge:
        sig = sig[edge:-edge]
    w = int(window_s * fs)
    windows = [sig[i:i + w] for i in range(0, max(len(sig) - w, 0) + 1, w)] or [sig]
    k = np.array([kurtosis(x) if x.std() > 0 else 0.0 for x in windows])
    bad = float((k < QUALITY_KURTOSIS_MIN).mean())
    ok = bad <= QUALITY_MAX_BAD_FRACTION
    return {
        "ok": ok,
        "low_quality_fraction": round(bad, 3),
        "median_kurtosis": round(float(np.median(k)), 2),
        "warning": None if ok else "Signal does not look like a clean ECG (noise, artefact or no QRS complexes); beat classifications are unreliable.",
    }


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """Per-beat z-score normalisation."""
    return ((beat - beat.mean()) / (beat.std() + 1e-8)).astype(np.float32)


RR_FEATURES = ["pre_rr_local", "post_rr_local", "pre_rr_global", "pre_post_ratio"]


def compute_rr_features(r_peaks: np.ndarray, fs: int = TARGET_FS, local_window: int = 10) -> np.ndarray:
    """Per-beat rhythm features [n_peaks, 4], all relative to the patient's own rhythm.

    - pre/post RR divided by the local mean RR (surrounding ~10 beats)
    - pre RR divided by the median RR of the whole recording
    - pre RR / post RR (prematurity followed by a compensatory pause)

    Morphology alone struggles with supraventricular beats; prematurity relative to the
    patient's own rhythm is the key cue. Absolute RR values (seconds) are deliberately left
    out: they identify patients and make the model overfit to the training subjects.
    """
    r_peaks = np.asarray(r_peaks)
    n = len(r_peaks)
    if n < 2:
        return np.ones((n, 4), dtype=np.float32)
    rr = np.diff(r_peaks) / fs
    pre = np.concatenate([[rr[0]], rr])
    post = np.concatenate([rr, [rr[-1]]])
    kernel = np.ones(local_window) / local_window
    local = np.convolve(np.pad(pre, (local_window // 2, local_window - local_window // 2 - 1), mode="edge"), kernel, mode="valid")
    return np.stack([pre / local, post / local, pre / np.median(rr), pre / post], axis=1).astype(np.float32)


def segment_beats(sig: np.ndarray, r_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed-length windows around R-peaks.

    Returns (beats [n, BEAT_LEN], kept_peak_indices).
    """
    beats, kept = [], []
    for i, r in enumerate(r_peaks):
        lo, hi = r - WINDOW_BEFORE, r + WINDOW_AFTER
        if lo < 0 or hi > len(sig):
            continue
        beats.append(normalize_beat(sig[lo:hi]))
        kept.append(i)
    if not beats:
        return np.empty((0, BEAT_LEN), dtype=np.float32), np.empty(0, dtype=int)
    return np.stack(beats), np.asarray(kept, dtype=int)


def prepare_signal(sig: np.ndarray, fs: float) -> np.ndarray:
    """Resample to TARGET_FS and band-pass filter."""
    return bandpass(resample(np.asarray(sig, dtype=np.float32), fs))
