"""MIT-BIH Arrhythmia Database helpers (download, load, build beat datasets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from app.ecg.preprocessing import (
    SYMBOL_TO_AAMI,
    TARGET_FS,
    bandpass,
    compute_rr_features,
    segment_beats,
)

ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234",
]
# de Chazal et al. (2004) inter-patient split; paced records 102, 104, 107, 217 excluded.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]


def download(data_dir: Path) -> Path:
    """Download MIT-BIH from PhysioNet (~100 MB) if not already present."""
    data_dir = Path(data_dir)
    if not (data_dir / "100.dat").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", dl_dir=str(data_dir))
    return data_dir


def load_record(data_dir: Path, record: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (MLII signal, annotation sample indices, annotation symbols, fs)."""
    rec = wfdb.rdrecord(str(Path(data_dir) / record))
    ann = wfdb.rdann(str(Path(data_dir) / record), "atr")
    ch = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    return rec.p_signal[:, ch].astype(np.float32), np.asarray(ann.sample), np.asarray(ann.symbol), int(rec.fs)


def record_beats(data_dir: Path, record: str) -> dict[str, np.ndarray]:
    """Beats, RR features and AAMI labels for one record (uses annotated R-peaks)."""
    sig, samples, symbols, fs = load_record(data_dir, record)
    assert fs == TARGET_FS, f"MIT-BIH is expected at {TARGET_FS} Hz, got {fs}"
    sig = bandpass(sig, fs)

    is_beat = np.isin(symbols, list(SYMBOL_TO_AAMI))
    peaks, syms = samples[is_beat], symbols[is_beat]
    rr = compute_rr_features(peaks, fs)
    beats, kept = segment_beats(sig, peaks)
    return {
        "beats": beats,
        "rr": rr[kept],
        "labels": np.array([SYMBOL_TO_AAMI[s] for s in syms[kept]]),
        "record": np.full(len(kept), record),
        "peak": peaks[kept],
    }


def build_dataset(data_dir: Path, records: list[str]) -> dict[str, np.ndarray]:
    parts = [record_beats(data_dir, r) for r in records]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
