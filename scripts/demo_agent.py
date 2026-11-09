"""End-to-end demo against a running server: OpenAI is the agentic core and calls the LSTM and the
knowledge base as tools. Shows the tools API (direct call, no LLM) and the agent API with its step trace.

Usage:
    uvicorn app.main:app            # in another terminal
    python scripts/demo_agent.py    # [--api http://localhost:8000] [--record 233] [--seconds 60]

Uploads the reference document, uploads a held-out MIT-BIH recording, then asks the agent three
questions and prints which tools it called, the sources it cited and its answers.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ecg import mitbih  # noqa: E402


def show(title: str, body: dict) -> None:
    print(f"\n=== {title}")
    llm = body.get("llm") or {}
    print(f"LLM: {llm.get('provider')}:{llm.get('model')}" + ("  (fallback used)" if llm.get("fallback_used") else ""))
    for st in body.get("steps", []):
        if st["type"] == "llm":
            what = "answer" if st["decision"] == "answer" else "call " + ", ".join(st["tool_calls"])
            tok = st.get("tokens") or {}
            print(f"  [LLM  {st['duration_ms']:>7.0f} ms] {st['provider']}:{st['model']} decides to {what}"
                  + (f"  ({tok.get('prompt')}+{tok.get('completion')} tokens)" if tok else ""))
        else:
            args = ", ".join(f"{k}={v!r}" for k, v in st["arguments"].items())
            print(f"  [TOOL {st['duration_ms']:>7.0f} ms] {st['tool']}({args})" + ("" if st["ok"] else f"  ERROR: {st['error']}"))
    for s in body["sources"]:
        print(f"source: {s['source']} chunk {s['chunk_index']} (score {s['score']:.2f})")
    for line in (body["answer"] or "").splitlines():
        print(textwrap.fill(line, 100, initial_indent="  ", subsequent_indent="    ") if line.strip() else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--record", default="233", help="MIT-BIH record (DS2 records were never used in training)")
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()

    with httpx.Client(base_url=args.api, timeout=180) as http:
        print("health:", http.get("/health").json())
        doc = ROOT / "docs" / "ecg_reference.md"
        print("ingested:", http.post("/documents/upload", files={"file": (doc.name, doc.read_bytes(), "text/markdown")}).json())

        sig, _, _, fs = mitbih.load_record(ROOT / "data" / "mitdb", args.record)
        ecg_id = http.post("/ecg/upload", json={"signal": sig[: args.seconds * fs].tolist(), "fs": fs,
                                                "name": f"mitdb_{args.record}"}).json()["ecg_id"]
        print(f"uploaded record {args.record} ({args.seconds} s) as {ecg_id}")
        print("agent:", http.get("/agent").json() | {"system_prompt": "..."})
        print("tools:", [t["name"] for t in http.get("/tools").json()])
        direct = http.post("/tools/analyze_ecg/invoke", json={"arguments": {"ecg_id": ecg_id}}).json()
        print(f"direct tool call (no LLM): {direct['result']['beat_counts']} in {direct['duration_ms']:.0f} ms")

        q1 = "Is anything abnormal in this ECG? Give the numbers."
        a1 = http.post("/agent/chat", json={"message": q1, "ecg_id": ecg_id}).json()
        show(q1, a1)

        q2 = "According to the reference documents, what are premature ventricular contractions and when are they significant?"
        show(q2, http.post("/agent/chat", json={"message": q2}).json())

        q3 = "Given those results and the references, what should a clinician check next?"
        history = [{"role": "user", "content": q1}, {"role": "assistant", "content": a1["answer"]}]
        show(q3, http.post("/agent/chat", json={"message": q3, "ecg_id": ecg_id, "history": history}).json())


if __name__ == "__main__":
    main()
