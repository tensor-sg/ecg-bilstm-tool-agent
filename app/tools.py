"""Tool layer: the functions the LLM agent can call, with their JSON schemas.

Independent of any LLM. The agent (app/agent.py) offers these tools to the model, and the /tools
API exposes them directly, so the LSTM and the knowledge base can be called by other agents,
scripts or services in exactly the same way the LLM calls them.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.ecg.classifier import ECGClassifier
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

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

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


class UnknownToolError(KeyError):
    pass


class ToolArgumentError(ValueError):
    pass


class ToolResourceNotFound(LookupError):
    """The tool ran but what it was asked about does not exist (e.g. an unknown ecg_id)."""

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details


class ToolUnavailable(RuntimeError):
    """The tool cannot run right now (e.g. the ECG model is not loaded)."""


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


class ToolBox:
    def __init__(self, store: VectorStore, registry: ECGRegistry, classifier: ECGClassifier | None, default_top_k: int = 5):
        self.store = store
        self.registry = registry
        self.classifier = classifier
        self.default_top_k = default_top_k
        self._handlers = {
            "search_knowledge_base": self._search_knowledge_base,
            "analyze_ecg": self._analyze_ecg,
            "list_ecg_recordings": self._list_ecg_recordings,
        }
        self._schemas = {t["function"]["name"]: t for t in TOOLS}

    # ---------- catalogue ----------
    @property
    def schemas(self) -> list[dict]:
        return TOOLS

    def schema(self, name: str) -> dict:
        if name not in self._schemas:
            raise UnknownToolError(name)
        return self._schemas[name]

    def validate(self, name: str, args: dict) -> None:
        """Check arguments against the tool's JSON schema (required, unknown and type of each field)."""
        params = self.schema(name)["function"]["parameters"]
        props = params.get("properties", {})
        if not isinstance(args, dict):
            raise ToolArgumentError("arguments must be a JSON object")
        missing = [k for k in params.get("required", []) if k not in args]
        unknown = [k for k in args if k not in props]
        wrong = [k for k, v in args.items() if k in props and not (
            isinstance(v, _JSON_TYPES[props[k]["type"]]) and not (props[k]["type"] in ("integer", "number") and isinstance(v, bool)))]
        problems = ([f"missing: {', '.join(missing)}"] if missing else []) + ([f"unknown: {', '.join(unknown)}"] if unknown else []) + \
                   [f"{k} must be {props[k]['type']}" for k in wrong]
        if problems:
            raise ToolArgumentError("; ".join(problems))

    # ---------- execution ----------
    def invoke(self, name: str, args: dict) -> dict:
        """Strict call for API clients: raises on unknown tools or invalid arguments."""
        self.validate(name, args)
        return self._handlers[name](**args)

    def run(self, name: str, args: dict) -> dict:
        """Forgiving call for the LLM: every failure becomes {"error": ...} so the model can recover."""
        try:
            return self.invoke(name, args)
        except UnknownToolError:
            return {"error": f"Unknown tool {name}", "available": list(self._handlers)}
        except ToolArgumentError as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}
        except ToolResourceNotFound as exc:
            return {"error": str(exc), **exc.details}
        except ToolUnavailable as exc:
            return {"error": str(exc)}
        except Exception as exc:  # surface tool failures to the model instead of crashing the request
            logger.exception("Tool %s failed", name)
            return {"error": str(exc)}

    def timed_run(self, name: str, args: dict) -> tuple[dict, float]:
        t = time.perf_counter()
        result = self.run(name, args)
        return result, round((time.perf_counter() - t) * 1000, 1)

    # ---------- tools ----------
    def _search_knowledge_base(self, query: str, top_k: int | None = None) -> dict:
        k = max(1, min(int(top_k or self.default_top_k), 10))
        return {"results": self.store.search(query, top_k=k)}

    def _analyze_ecg(self, ecg_id: str) -> dict:
        if self.classifier is None:
            raise ToolUnavailable("ECG model is not loaded. Train it with notebooks/02_lstm_training.ipynb.")
        rec = self.registry.get(ecg_id)
        if rec is None:
            raise ToolResourceNotFound(f"Unknown ecg_id '{ecg_id}'.", available=self.registry.list())
        return {"ecg_id": ecg_id, "name": rec.name, **self.classifier.analyze(rec.signal, rec.fs, rec.r_peaks)}

    def _list_ecg_recordings(self) -> dict:
        return {"recordings": self.registry.list()}
