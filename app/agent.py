"""Agentic core: an OpenAI-compatible LLM that decides which tools to call (app/tools.py) and writes the answer.

Flow of one request: system prompt + history + question -> LLM -> tool calls -> tool results -> LLM -> ...
-> final answer. Every step is recorded in a trace returned to the client.
"""

from __future__ import annotations

import json
import logging
import time

import openai
from openai import OpenAI

from app.config import LLMEndpoint, Settings
from app.ecg.classifier import ECGClassifier
from app.tools import TOOLS, ECGRecording, ECGRegistry, ToolBox  # noqa: F401  (re-exported)
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a cardiology research assistant.
You have tools to (1) search a knowledge base of uploaded documents and (2) run an LSTM \
ECG beat classifier on uploaded ECG recordings.
- Use search_knowledge_base for factual / guideline questions and cite the sources you used.
- Use analyze_ecg whenever the user asks about an ECG recording; never guess its contents.
- Explain the beat classes the tool reports (N normal, S supraventricular ectopic, V ventricular ectopic) in plain language and mention model confidence.
- The model does not detect fusion or paced beats, and its S-beat sensitivity is limited; say so when relevant.
- If analyze_ecg returns signal_quality.ok = false, say the recording looks like noise or artefact and do not interpret the beat counts.
- Always state that results are not a medical diagnosis."""

MAX_STEPS = 6


class Agent:
    def __init__(self, settings: Settings, store: VectorStore, registry: ECGRegistry, classifier: ECGClassifier | None):
        main = settings.llm_endpoint()
        self.client, self.model, self.provider = _client(main), main.model, main.provider
        fallback = settings.llm_fallback_endpoint()
        self.fallback = (_client(fallback), fallback.model, fallback.provider) if fallback else None
        self.temperature = settings.llm_temperature
        self.tools = ToolBox(store, registry, classifier, settings.top_k)

    def run_tool(self, name: str, args: dict) -> dict:
        return self.tools.run(name, args)

    # ---------- LLM ----------
    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model,
                "fallback": f"{self.fallback[2]}:{self.fallback[1]}" if self.fallback else None}

    def config(self) -> dict:
        return {**self.describe(), "temperature": self.temperature, "max_steps": MAX_STEPS,
                "tools": [t["function"]["name"] for t in TOOLS], "system_prompt": SYSTEM_PROMPT}

    def _complete(self, messages: list[dict], used: dict):
        """One chat-completions call; on a provider error (auth, quota, outage) retry once on the fallback."""
        kwargs = {"messages": messages, "tools": TOOLS, "tool_choice": "auto"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        targets = [(self.client, self.model, self.provider)] + ([self.fallback] if self.fallback else [])
        if used.get("switched"):
            targets = targets[1:]
        for i, (client, model, provider) in enumerate(targets):
            try:
                resp = client.chat.completions.create(model=model, **kwargs)
                used.update(provider=provider, model=model)
                return resp
            except openai.APIError as exc:
                if i == len(targets) - 1:
                    raise
                logger.warning("LLM %s:%s failed (%s); switching to fallback %s:%s",
                               provider, model, type(exc).__name__, targets[i + 1][2], targets[i + 1][1])
                used["switched"] = True

    # ---------- loop ----------
    def chat(self, message: str, history: list[dict] | None = None, ecg_id: str | None = None, max_steps: int = MAX_STEPS) -> dict:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history or []
        user_content = message if not ecg_id else f"{message}\n\n[Attached ECG recording: ecg_id={ecg_id}]"
        messages.append({"role": "user", "content": user_content})

        trace: list[dict] = []
        sources: list[dict] = []
        steps: list[dict] = []
        used: dict = {}
        for _ in range(max_steps):
            t = time.perf_counter()
            resp = self._complete(messages, used)
            msg = resp.choices[0].message
            steps.append({"type": "llm", "provider": used.get("provider"), "model": used.get("model"),
                          "duration_ms": round((time.perf_counter() - t) * 1000, 1), "tokens": _tokens(resp),
                          "decision": "answer" if not msg.tool_calls else "call_tools",
                          "tool_calls": [tc.function.name for tc in msg.tool_calls or []]})
            if not msg.tool_calls:
                return {"answer": msg.content, "tool_calls": trace, "sources": sources, "llm": _llm_info(used), "steps": steps}

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
                result, ms = self.tools.timed_run(tc.function.name, args)
                trace.append({"tool": tc.function.name, "arguments": args})
                steps.append({"type": "tool", "tool": tc.function.name, "arguments": args, "duration_ms": ms,
                              "ok": "error" not in result, "error": result.get("error")})
                if tc.function.name == "search_knowledge_base":
                    sources += [{"source": r["source"], "chunk_index": r["chunk_index"], "score": r["score"]} for r in result.get("results", [])]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

        return {"answer": "Stopped: reached the maximum number of tool steps.", "tool_calls": trace, "sources": sources,
                "llm": _llm_info(used), "steps": steps}


def _client(ep: LLMEndpoint) -> OpenAI:
    return OpenAI(api_key=ep.api_key, base_url=ep.base_url, max_retries=1, timeout=60)


def _tokens(resp) -> dict | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    return {"prompt": getattr(usage, "prompt_tokens", None), "completion": getattr(usage, "completion_tokens", None)}


def _llm_info(used: dict) -> dict:
    return {"provider": used.get("provider"), "model": used.get("model"), "fallback_used": bool(used.get("switched"))}
