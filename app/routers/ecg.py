import csv
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.deps import AppState, get_state, require_classifier
from app.ecg.classifier import ECGClassifier
from app.schemas import ECGUploadRequest, ECGUploadResponse

router = APIRouter(prefix="/ecg", tags=["ecg"])


@router.post("/upload", response_model=ECGUploadResponse)
async def upload_ecg(req: ECGUploadRequest, state: AppState = Depends(get_state)):
    rec = state.registry.add(req.signal, req.fs, req.name, req.r_peaks)
    return ECGUploadResponse(ecg_id=rec.id, duration_s=round(len(rec.signal) / rec.fs, 2))


@router.post("/upload-csv", response_model=ECGUploadResponse)
async def upload_ecg_csv(
    file: UploadFile = File(..., description="CSV with one numeric column (or pick one with `column`)."),
    fs: float = Form(...),
    column: int = Form(0),
    state: AppState = Depends(get_state),
):
    text = (await file.read()).decode("utf-8", errors="ignore")
    signal: list[float] = []
    for row in csv.reader(io.StringIO(text)):
        try:
            signal.append(float(row[column]))
        except (ValueError, IndexError):
            continue  # skip header / malformed rows
    if len(signal) < 256:
        raise HTTPException(400, "Need at least 256 numeric samples.")
    rec = state.registry.add(signal, fs, file.filename)
    return ECGUploadResponse(ecg_id=rec.id, duration_s=round(len(signal) / fs, 2))


@router.get("/")
async def list_ecgs(state: AppState = Depends(get_state)):
    return state.registry.list()


@router.post("/{ecg_id}/analyze")
async def analyze_ecg(ecg_id: str, state: AppState = Depends(get_state), clf: ECGClassifier = Depends(require_classifier)):
    rec = state.registry.get(ecg_id)
    if rec is None:
        raise HTTPException(404, f"Unknown ecg_id {ecg_id}")
    return await run_in_threadpool(clf.analyze, rec.signal, rec.fs, rec.r_peaks)
