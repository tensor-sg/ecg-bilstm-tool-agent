import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import Agent, ECGRegistry
from app.config import get_settings
from app.deps import AppState
from app.ecg.classifier import ECGClassifier
from app.embeddings import Embedder
from app.routers import chat, documents, ecg
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


app = FastAPI(title="ECG RAG + LSTM Agent", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(ecg.router)
app.include_router(chat.router)


@app.get("/health", tags=["health"])
async def health():
    s: AppState = app.state.services
    return {"status": "ok", "vector_collection": s.store.collection, "ecg_model_loaded": s.classifier is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=True)
