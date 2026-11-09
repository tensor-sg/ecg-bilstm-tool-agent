"""Agent API: the LLM agentic core that decides when to call the tools.

GET  /agent       -> provider, model, fallback, tools and system prompt in use
POST /agent/chat  -> run the agent loop; returns the answer, cited sources and a step-by-step trace
POST /chat        -> same as /agent/chat (kept for existing clients)
"""

import logging

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.deps import AppState, get_state
from app.schemas import AgentConfig, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


@router.get("/agent", response_model=AgentConfig)
async def agent_config(state: AppState = Depends(get_state)):
    return state.agent.config()


@router.post("/agent/chat", response_model=ChatResponse)
@router.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat(req: ChatRequest, state: AppState = Depends(get_state)):
    if req.ecg_id and state.registry.get(req.ecg_id) is None:
        raise HTTPException(404, f"Unknown ecg_id {req.ecg_id}")
    history = [m.model_dump() for m in req.history]
    try:
        return await run_in_threadpool(state.agent.chat, req.message, history, req.ecg_id)
    except openai.APIError as exc:
        logger.warning("LLM request failed: %s", exc)
        raise HTTPException(
            502,
            f"LLM request failed ({type(exc).__name__}). Check LLM_PROVIDER, LLM_MODEL and the matching API key "
            "(OPENAI_API_KEY, GEMINI_API_KEY or LLM_API_KEY) in .env.",
        ) from exc
