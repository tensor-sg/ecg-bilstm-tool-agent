"""Tool-calling agent: the LLM can search the Chroma knowledge base (RAG) and run the ECG LSTM model as tools."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openai import OpenAI

from app.config import Settings
from app.ecg.classifier import ECGClassifier
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a cardiology research assistant.
You have tools to (1) search a knowledge base of uploaded documents and (2) run an LSTM \
ECG beat classifier on uploaded ECG recordings.
- Use search_knowledge_base for factual / guideline questions and cite the sources you used.
- Use analyze_ecg whenever the user asks about an ECG recording; never guess its contents.
- Explain the beat classes the tool reports (N normal, S supraventricular ectopic, V ventricular ectopic) in plain language and mention model confidence.
- The model does not detect fusion or paced beats, and its S-beat sensitivity is limited; say so when relevant.
- Always state that results are not a medical diagnosis."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Semantic search over the uploaded document knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "top_k": {"type": "integer", "description": "Number of chunks to return (1-10).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_ecg",
            "description": (
                "Run the trained LSTM classifier on an uploaded single-lead ECG recording. "
                "Returns heart rate, RR variability, per-class beat counts (AAMI N/S/V) and abnormal beat times."
            ),
            "parameters": {
                "type": "object",
                "properties": {"ecg_id": {"type": "string", "description": "Id returned by POST /ecg/upload."}},
                "required": ["ecg_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_ecg_recordings",
            "description": "List the ECG recordings currently available for analysis.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class ECGRecording:
    id: str
    signal: list[float]
    fs: float
    name: str | None = None
    r_peaks: list[int] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ECGRegistry:
    """In-memory store of uploaded ECG strips (swap for Redis/DB in production)."""

    def __init__(self) -> None:
        self._items: dict[str, ECGRecording] = {}

    def add(self, signal: list[float], fs: float, name: str | None = None, r_peaks: list[int] | None = None) -> ECGRecording:
        rec = ECGRecording(id=uuid.uuid4().hex[:12], signal=signal, fs=fs, name=name, r_peaks=r_peaks)
        self._items[rec.id] = rec
        return rec

    def get(self, ecg_id: str) -> ECGRecording | None:
        return self._items.get(ecg_id)

    def list(self) -> list[dict]:
        return [
            {"ecg_id": r.id, "name": r.name, "fs": r.fs, "duration_s": round(len(r.signal) / r.fs, 2), "created_at": r.created_at}
            for r in self._items.values()
        ]


class Agent:
    def __init__(self, settings: Settings, store: VectorStore, registry: ECGRegistry, classifier: ECGClassifier | None):
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model
        self.store = store
        self.registry = registry
        self.classifier = classifier
        self.default_top_k = settings.top_k

    # ---------- tools ----------
    def _search_knowledge_base(self, query: str, top_k: int | None = None) -> dict:
        k = max(1, min(int(top_k or self.default_top_k), 10))
        return {"results": self.store.search(query, top_k=k)}

    def _analyze_ecg(self, ecg_id: str) -> dict:
        if self.classifier is None:
            return {"error": "ECG model is not loaded. Train it with notebooks/02_lstm_training.ipynb."}
        rec = self.registry.get(ecg_id)
        if rec is None:
            return {"error": f"Unknown ecg_id '{ecg_id}'.", "available": self.registry.list()}
        return {"ecg_id": ecg_id, "name": rec.name, **self.classifier.analyze(rec.signal, rec.fs, rec.r_peaks)}

    def _list_ecg_recordings(self) -> dict:
        return {"recordings": self.registry.list()}

    def run_tool(self, name: str, args: dict) -> dict:
        handlers = {
            "search_knowledge_base": self._search_knowledge_base,
            "analyze_ecg": self._analyze_ecg,
            "list_ecg_recordings": self._list_ecg_recordings,
        }
        if name not in handlers:
            return {"error": f"Unknown tool {name}"}
        try:
            return handlers[name](**args)
        except Exception as exc:  # surface tool failures to the model instead of crashing the request
            logger.exception("Tool %s failed", name)
            return {"error": str(exc)}

    # ---------- loop ----------
    def chat(self, message: str, history: list[dict] | None = None, ecg_id: str | None = None, max_steps: int = 6) -> dict:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history or []
        user_content = message if not ecg_id else f"{message}\n\n[Attached ECG recording: ecg_id={ecg_id}]"
        messages.append({"role": "user", "content": user_content})

        trace: list[dict] = []
        sources: list[dict] = []
        for _ in range(max_steps):
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS, tool_choice="auto")
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return {"answer": msg.content, "tool_calls": trace, "sources": sources}

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.run_tool(tc.function.name, args)
                trace.append({"tool": tc.function.name, "arguments": args})
                if tc.function.name == "search_knowledge_base":
                    sources += [{"source": r["source"], "chunk_index": r["chunk_index"], "score": r["score"]} for r in result.get("results", [])]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

        return {"answer": "Stopped: reached the maximum number of tool steps.", "tool_calls": trace, "sources": sources}
