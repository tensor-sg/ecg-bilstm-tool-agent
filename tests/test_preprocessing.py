import numpy as np
import pytest
from scipy import signal as sps

from app.ecg.preprocessing import (
    BEAT_LEN,
    SYMBOL_TO_AAMI,
    TARGET_FS,
    WINDOW_AFTER,
    WINDOW_BEFORE,
    bandpass,
    compute_rr_features,
    detect_r_peaks,
    normalize_beat,
    prepare_signal,
    resample,
    segment_beats,
)
from tests.conftest import synthetic_ecg


def test_beat_window_constants():
    assert BEAT_LEN == WINDOW_BEFORE + WINDOW_AFTER == 256


def test_aami_mapping_covers_all_classes():
    assert set(SYMBOL_TO_AAMI.values()) == {"N", "S", "V", "F", "Q"}
    assert SYMBOL_TO_AAMI["N"] == "N" and SYMBOL_TO_AAMI["V"] == "V" and SYMBOL_TO_AAMI["A"] == "S"


@pytest.mark.parametrize("fs", [128, 250, 500, 1000])
def test_resample_length(fs):
    out = resample(np.random.randn(fs * 10), fs)
    assert out.dtype == np.float32
    assert abs(len(out) - TARGET_FS * 10) <= 1


def test_resample_noop_at_target_fs():
    x = np.random.randn(1000).astype(np.float32)
    np.testing.assert_array_equal(resample(x, TARGET_FS), x)


def test_bandpass_removes_baseline_wander_keeps_qrs_band():
    t = np.arange(TARGET_FS * 20) / TARGET_FS
    wander = np.sin(2 * np.pi * 0.1 * t)
    qrs_band = np.sin(2 * np.pi * 10 * t)
    out = bandpass(wander + qrs_band)
    f, p = sps.welch(out, fs=TARGET_FS, nperseg=4096)
    p_in = sps.welch(wander + qrs_band, fs=TARGET_FS, nperseg=4096)[1]
    low = f < 0.2
    band = (f > 9) & (f < 11)
    assert p[low].sum() < 0.01 * p_in[low].sum()     # >20 dB attenuation of wander
    assert p[band].sum() > 0.8 * p_in[band].sum()    # 10 Hz passes almost untouched


def test_normalize_beat():
    b = normalize_beat(np.random.randn(BEAT_LEN) * 5 + 3)
    assert abs(b.mean()) < 1e-5 and abs(b.std() - 1) < 1e-3


def test_normalize_flat_beat_is_finite():
    assert np.isfinite(normalize_beat(np.ones(BEAT_LEN))).all()


@pytest.mark.parametrize("rr_s", [0.5, 0.8, 1.2])
def test_detect_r_peaks_on_synthetic_ecg(rr_s):
    sig, true_peaks = synthetic_ecg(rr_s=rr_s, seed=1)
    det = detect_r_peaks(bandpass(sig))
    tol = int(0.05 * TARGET_FS)
    matched = sum(np.min(np.abs(det - p)) <= tol for p in true_peaks)
    assert matched / len(true_peaks) >= 0.98            # sensitivity
    assert matched / len(det) >= 0.98                   # positive predictivity


def test_detect_r_peaks_after_resampling_from_250hz():
    sig360, true_peaks = synthetic_ecg(seconds=20)
    sig250 = sps.resample(sig360, int(len(sig360) * 250 / 360))
    det = detect_r_peaks(prepare_signal(sig250, 250))
    tol = int(0.05 * TARGET_FS)
    assert sum(np.min(np.abs(det - p)) <= tol for p in true_peaks) / len(true_peaks) >= 0.95


def test_rr_features_regular_rhythm():
    peaks = np.arange(0, 360 * 20, 288)                  # 0.8 s RR -> 75 bpm
    rr = compute_rr_features(peaks)
    assert rr.shape == (len(peaks), 4)
    np.testing.assert_allclose(rr, 1.0, atol=1e-6)       # every ratio is 1 for a perfectly regular rhythm


def test_rr_features_are_heart_rate_independent():
    """Same rhythm pattern at 60 and 120 bpm must give identical features (no patient-identifying absolute RR)."""
    pattern = [1.0] * 10 + [0.6, 1.4] + [1.0] * 10
    slow = compute_rr_features(np.cumsum([0] + [int(360 * x) for x in pattern]))
    fast = compute_rr_features(np.cumsum([0] + [int(180 * x) for x in pattern]))
    np.testing.assert_allclose(slow, fast, atol=0.02)


def test_rr_features_flag_premature_beat():
    peaks = np.cumsum([0] + [288] * 10 + [173] + [403] + [288] * 10)   # early beat then compensatory pause
    rr = compute_rr_features(peaks)
    early = 11
    assert rr[early, 0] < 0.75      # short pre-RR relative to local mean
    assert rr[early, 1] > 1.3       # long post-RR (compensatory pause)
    assert rr[early, 2] < 0.7       # short relative to the recording median
    assert rr[early, 3] < 0.5       # pre/post ratio


def test_rr_features_degenerate_inputs():
    assert compute_rr_features(np.array([100])).shape == (1, 4)
    assert compute_rr_features(np.array([], dtype=int)).shape == (0, 4)


def test_segment_beats_skips_edges():
    sig = np.random.randn(2000).astype(np.float32)
    peaks = np.array([10, 500, 1000, 1990])
    beats, kept = segment_beats(sig, peaks)
    assert beats.shape == (2, BEAT_LEN)
    np.testing.assert_array_equal(kept, [1, 2])


def test_segment_beats_empty():
    beats, kept = segment_beats(np.zeros(100, dtype=np.float32), np.array([50]))
    assert beats.shape == (0, BEAT_LEN) and len(kept) == 0
