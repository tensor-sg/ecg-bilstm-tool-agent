"""Application-wide singletons, created once in the FastAPI lifespan."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.agent import Agent, ECGRegistry
from app.config import Settings
from app.ecg.classifier import ECGClassifier
from app.vector_store import VectorStore


@dataclass
class AppState:
    settings: Settings
    store: VectorStore
    registry: ECGRegistry
    classifier: ECGClassifier | None
    agent: Agent


def get_state(request: Request) -> AppState:
    return request.app.state.services


def require_classifier(request: Request) -> ECGClassifier:
    clf = get_state(request).classifier
    if clf is None:
        raise HTTPException(503, "ECG model not loaded. Train it with notebooks/02_lstm_training.ipynb and restart.")
    return clf
