import logging

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.deps import AppState, get_state
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, state: AppState = Depends(get_state)):
    if req.ecg_id and state.registry.get(req.ecg_id) is None:
        raise HTTPException(404, f"Unknown ecg_id {req.ecg_id}")
    history = [m.model_dump() for m in req.history]
    try:
        return await run_in_threadpool(state.agent.chat, req.message, history, req.ecg_id)
    except openai.APIError as exc:
        logger.warning("LLM request failed: %s", exc)
        raise HTTPException(
            502, f"LLM request failed ({type(exc).__name__}). Check LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in .env."
        ) from exc
