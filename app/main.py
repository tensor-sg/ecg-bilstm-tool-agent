import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import Agent, ECGRegistry
from app.config import get_settings
from app.deps import AppState
from app.ecg.classifier import ECGClassifier
from app.embeddings import Embedder
from app.routers import agent, documents, ecg, tools
from app.vector_store import VectorStore

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore(settings, Embedder(settings))
    try:
        classifier = ECGClassifier(settings.resolve(settings.ecg_model_path), settings.resolve(settings.ecg_meta_path))
    except FileNotFoundError as exc:
        logger.warning("%s — /ecg/*/analyze and the analyze_ecg tool are disabled.", exc)
        classifier = None
    registry = ECGRegistry()
    app.state.services = AppState(
        settings=settings,
        store=store,
        registry=registry,
        classifier=classifier,
        agent=Agent(settings, store, registry, classifier),
    )
    yield


DESCRIPTION = """
An OpenAI LLM is the agentic core; the trained ECG LSTM and the Chroma knowledge base are its tools.

* **/agent** – the agent: chat, with a trace of every LLM decision and tool call
* **/tools** – the same tools as plain REST endpoints, callable without an LLM
* **/ecg**, **/documents** – upload ECG recordings and reference documents
"""
app = FastAPI(title="ECG RAG + LSTM Agent", version="0.2.0", description=DESCRIPTION, lifespan=lifespan,
              openapi_tags=[{"name": "agent", "description": "LLM agent (OpenAI main, Gemini fallback)"},
                            {"name": "tools", "description": "Tools the agent can call, exposed directly"},
                            {"name": "ecg", "description": "ECG upload and direct analysis"},
                            {"name": "documents", "description": "Knowledge base (RAG) management"},
                            {"name": "health", "description": "Service status"}])
app.include_router(documents.router)
app.include_router(ecg.router)
app.include_router(tools.router)
app.include_router(agent.router)


@app.get("/health", tags=["health"])
async def health():
    s: AppState = app.state.services
    return {"status": "ok", "vector_collection": s.store.collection, "ecg_model_loaded": s.classifier is not None,
            "llm": s.agent.describe()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
