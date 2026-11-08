from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.deps import AppState, get_state
from app.ingest import chunk_text, extract_text
from app.schemas import IngestResponse, IngestTextRequest, SearchHit, SearchRequest

router = APIRouter(prefix="/documents", tags=["documents"])


async def _ingest(state: AppState, source: str, text: str) -> IngestResponse:
    chunks = chunk_text(text, state.settings.chunk_size, state.settings.chunk_overlap)
    if not chunks:
        raise HTTPException(400, "No text could be extracted.")
    await run_in_threadpool(state.store.delete_source, source)
    n = await run_in_threadpool(state.store.add_chunks, source, chunks)
    return IngestResponse(source=source, chunks=n)


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...), state: AppState = Depends(get_state)):
    data = await file.read()
    try:
        text = extract_text(file.filename or "", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await _ingest(state, file.filename or "upload", text)


@router.post("/text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest, state: AppState = Depends(get_state)):
    return await _ingest(state, req.source, req.text)


@router.post("/search", response_model=list[SearchHit])
async def search(req: SearchRequest, state: AppState = Depends(get_state)):
    return await run_in_threadpool(state.store.search, req.query, req.top_k, req.source)


@router.delete("/{source}")
async def delete_document(source: str, state: AppState = Depends(get_state)):
    await run_in_threadpool(state.store.delete_source, source)
    return {"deleted": source}


@router.get("/count")
async def count(state: AppState = Depends(get_state)):
    return {"collection": state.store.collection, "points": await run_in_threadpool(state.store.count)}
