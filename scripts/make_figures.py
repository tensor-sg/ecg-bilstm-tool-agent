"""Figures and diagrams for the project report (docs/figures/*.png).

Usage:
    python scripts/make_figures.py

Reads models/training_log.csv, models/ecg_lstm_meta.json, reports/evaluation.json and
reports/performance.json; diagrams are drawn with Graphviz (`dot` must be on PATH).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "unused")  # settings need a key; no LLM is called

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = ROOT / "docs" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLOR = {"N": "#2a78d6", "S": "#eb6834", "V": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


# ---------------------------------------------------------------- charts
def training_curves(meta):
    rows = list(csv.DictReader(open(ROOT / "models" / "training_log.csv")))
    ep = np.array([int(r["epoch"]) + 1 for r in rows])
    get = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    best = meta["best_epoch"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3))
    a.plot(ep, get("loss"), color=SERIES[0], lw=2, marker="o", ms=4, label="train")
    a.plot(ep, get("val_loss"), color=SERIES[1], lw=2, marker="o", ms=4, label="validation")
    a.set(title="Loss (weighted cross-entropy)", xlabel="epoch")
    a.legend()
    b.plot(ep, get("val_macro_f1"), color=SERIES[1], lw=2, marker="o", ms=4, label="validation macro-F1")
    b.plot(ep, get("val_balanced_acc"), color=SERIES[2], lw=2, marker="o", ms=4, label="validation balanced acc.")
    b.axvline(best, color=MUTED, ls="--", lw=1)
    b.text(best + 0.1, 0.31, f"checkpoint kept (epoch {best})", color=MUTED, fontsize=8)
    b.set(title="Validation metrics (model selection)", xlabel="epoch", ylim=(0.3, 0.75))
    b.legend(loc="center right")
    for ax in (a, b):
        ax.set_xticks(ep)
    save(fig, "training_curves.png")


def confusion(ev):
    cm = np.array(ev["annotated"]["confusion_matrix"])
    classes = ev["classes"]
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else INK)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=classes, yticklabels=classes, xlabel="predicted", ylabel="true",
           title="Confusion matrix, DS2 test patients\n(row-normalised; counts in brackets)")
    save(fig, "confusion_matrix.png")


def reliability(perf):
    bins = perf["probability"]["reliability"]
    conf = [b["confidence"] for b in bins]
    acc = [b["accuracy"] for b in bins]
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfect calibration")
    ax.plot(conf, acc, color=SERIES[0], lw=2, marker="o", ms=5, label="model")
    ax.set(xlim=(0.3, 1), ylim=(0.3, 1), xlabel="mean confidence in bin", ylabel="observed accuracy",
           title=f"Reliability diagram (ECE = {perf['probability']['ece']:.3f})")
    ax.legend(loc="upper left")
    save(fig, "reliability.png")


def baselines(meta):
    names = ["Majority\n(always N)", "RR-only\nlogistic regr.", "Conv-BiLSTM\n(this work)"]
    keys = ["majority (always N)", "RR-only logistic regression", "Conv-BiLSTM (this model)"]
    f1 = [meta["baselines_test"][k]["macro_f1"] for k in keys]
    lo, hi = meta["test_ci95"]["macro_f1"]
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.bar(range(3), f1, color=[GRID, GRID, SERIES[0]], edgecolor=[MUTED, MUTED, SERIES[0]], width=0.55)
    ax.errorbar(2, f1[2], yerr=[[f1[2] - lo], [hi - f1[2]]], color=INK, capsize=4, lw=1)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.015 if i < 2 else hi + 0.015, f"{v:.3f}", ha="center", color=INK, fontsize=9)
    ax.set(xticks=range(3), xticklabels=names, ylabel="macro-F1 (DS2)", ylim=(0, 0.9),
           title="Held-out macro-F1 vs baselines\n(error bar: patient-bootstrap 95% CI)")
    ax.grid(axis="x", visible=False)
    save(fig, "baselines.png")


def ecg_strip():
    from app.ecg import mitbih
    from app.ecg.classifier import ECGClassifier
    from app.ecg.preprocessing import SYMBOL_TO_AAMI, TARGET_FS, prepare_signal

    clf = ECGClassifier(ROOT / "models" / "ecg_lstm.keras", ROOT / "models" / "ecg_lstm_meta.json")
    sig, samples, symbols, fs = mitbih.load_record(ROOT / "data" / "mitdb", "233")
    n = int(60 * fs)
    res = clf.analyze(sig[:n], fs)
    clean = prepare_signal(sig[:n], fs)
    t0, t1 = 20, 30
    t = np.arange(len(clean)) / TARGET_FS
    w = (t >= t0) & (t <= t1)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(t[w], clean[w], color=INK, lw=0.8)
    ax.grid(False)
    top = clean[w].max()
    for s, p in zip(symbols, samples):
        if t0 <= p / fs <= t1 and s in SYMBOL_TO_AAMI:
            ax.text(p / fs, top * 1.35, SYMBOL_TO_AAMI[s], ha="center", fontsize=7, color=MUTED)
    for b in res["abnormal_beats"]:
        if t0 <= b["time_s"] <= t1:
            ax.axvspan(b["time_s"] - 0.12, b["time_s"] + 0.12, color=CLASS_COLOR[b["label"]], alpha=0.25, lw=0)
            ax.text(b["time_s"], top * 1.03, f"{b['label']}\n{b['confidence']:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color=INK, linespacing=0.9)
    ax.set(xlabel="time (s)", ylabel="filtered ECG (a.u.)", xlim=(t0, t1))
    ax.set_ylim(top=top * 1.5)
    ax.set_title("Record 233 (held-out): cardiologist labels (top row) vs model flags (shaded, with confidence)", fontsize=10)
    save(fig, "ecg_strip_233.png")


# ---------------------------------------------------------------- diagrams (Graphviz)
def dot(name, src):
    path = FIG / f"{name}.dot"
    path.write_text(src)
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", str(path), "-o", str(FIG / f"{name}.png")], check=True)
    path.unlink()
    print("wrote", FIG / f"{name}.png")


FONT = 'fontname="Liberation Serif"'


def architecture():
    dot("architecture", f"""digraph G {{
rankdir=TB; nodesep=0.25; ranksep=0.35; {FONT}; node [{FONT}, shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#52514e", fontsize=14];
edge [{FONT}, fontsize=12, color="#52514e"];
user [label="Client\\n(browser, notebook,\\nscript, other agent)", fillcolor="#ffffff"];
subgraph cluster_api {{ label="FastAPI service (app/)"; labelloc="b"; style="rounded"; color="#2a78d6"; fontcolor="#2a78d6";
  agentapi [label="Agent API\\nGET /agent · POST /agent/chat", fillcolor="#fdeee7"];
  toolapi [label="Tools API\\nGET /tools · POST /tools/{{name}}/invoke", fillcolor="#e8f6f0"];
  data [label="/ecg/*  ·  /documents/*\\nupload, analyze, ingest, search"];
  agent [label="Agent (agent.py)\\nloop · fallback · step trace", fillcolor="#fdeee7"];
  tb [label="ToolBox (tools.py)\\nJSON schemas · validation", fillcolor="#e8f6f0"];
  clf [label="analyze_ecg\\nECGClassifier → Conv-BiLSTM"];
  kb [label="search_knowledge_base\\nMiniLM embeddings"];
  reg [label="list_ecg_recordings\\nECG registry (memory)"];
}}
openai [label="OpenAI gpt-4o-mini\\n(agentic core)", fillcolor="#fbdccf"];
gemini [label="Gemini 2.5 Flash\\n(fallback)", fillcolor="#f4f3ef", style="rounded,filled,dashed"];
chroma [label="Chroma\\n(chroma_db/ on disk)", shape=cylinder, fillcolor="#e8f0fb"];
model [label="models/ecg_lstm.keras", shape=note, fillcolor="#e8f6f0"];
user -> agentapi; user -> toolapi; user -> data;
agentapi -> agent;
agent -> openai [dir=both, label="messages,\\ntool calls"];
agent -> gemini [dir=both, style=dashed, label="on error"];
agent -> tb [label="run tool"]; toolapi -> tb [label="invoke"];
tb -> clf; tb -> kb; tb -> reg;
clf -> model [style=dashed, label="loads"]; kb -> chroma; data -> chroma; data -> reg;
}}""")


def model_diagram():
    dot("model", f"""digraph G {{
rankdir=TB; nodesep=0.3; ranksep=0.25; {FONT}; node [{FONT}, shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#52514e", fontsize=10, width=2.6];
edge [color="#52514e", arrowsize=0.6];
b0 [label="Beat waveform 256 × 1\\n(0.25 s before, 0.46 s after R, z-scored)", fillcolor="#e8f0fb"];
b1 [label="GaussianNoise σ = 0.1 (training only)"];
b2 [label="Conv1D 32 filters, k = 7 → BatchNorm → MaxPool 2"];
b3 [label="Conv1D 64 filters, k = 5 → BatchNorm → MaxPool 2"];
b4 [label="Bidirectional LSTM 32 (sequences) → Dropout 0.3"];
b5 [label="Bidirectional LSTM 16"];
r0 [label="RR features × 4\\n(patient-relative)", fillcolor="#e8f6f0"];
r1 [label="GaussianNoise σ = 0.03"];
r2 [label="BatchNorm → Dense 16 (ReLU)"];
c [label="Concatenate (32 + 16)"];
d [label="Dense 64 (ReLU) → Dropout 0.4"];
o [label="Dense 3, softmax → P(N), P(S), P(V)", fillcolor="#fdeee7"];
b0 -> b1 -> b2 -> b3 -> b4 -> b5 -> c; r0 -> r1 -> r2 -> c; c -> d -> o;
}}""")


def erd():
    def table(name, color, rows):
        cells = "".join(
            f'<TR><TD ALIGN="LEFT">{"<B>" if k == "PK" else ""}{f}{"</B>" if k == "PK" else ""}</TD>'
            f'<TD ALIGN="LEFT"><FONT COLOR="#52514e">{k or " "}</FONT></TD></TR>' for f, k in rows)
        return (f'{name} [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" COLOR="#52514e">'
                f'<TR><TD COLSPAN="2" BGCOLOR="{color}"><B>{name}</B></TD></TR>{cells}</TABLE>>];')

    blue, green, orange, grey = "#d6e4f7", "#d3efe4", "#fbdccf", "#ebeae6"
    ents = [
        table("PATIENT_RECORD", grey, [("record_id", "PK"), ("split (train/val/test)", ""), ("fs = 360 Hz", ""), ("lead = MLII", ""), ("duration_s", "")]),
        table("ANNOTATION", grey, [("record_id", "FK"), ("sample", ""), ("symbol (MIT-BIH)", ""), ("aami_class", "")]),
        table("BEAT", grey, [("beat_id", "PK"), ("record_id", "FK"), ("r_peak_sample", ""), ("waveform[256]", ""), ("rr_features[4]", ""), ("label (N/S/V)", "")]),
        table("MODEL_VERSION", green, [("model_id", "PK"), ("artefact_path", ""), ("classes", ""), ("hyperparameters", ""), ("best_epoch", ""), ("val/test metrics", "")]),
        table("ECG_UPLOAD", blue, [("ecg_id", "PK"), ("name", ""), ("fs", ""), ("signal[]", ""), ("r_peaks[] (optional)", "")]),
        table("ANALYSIS_RESULT", blue, [("ecg_id", "FK"), ("model_id", "FK"), ("heart_rate_bpm", ""), ("rr_sdnn_ms", ""), ("beat_counts", ""), ("signal_quality", "")]),
        table("ABNORMAL_BEAT", blue, [("ecg_id", "FK"), ("beat_index", ""), ("time_s", ""), ("label (S/V)", ""), ("confidence", "")]),
        table("COLLECTION", orange, [("name", "PK"), ("embedding_model", ""), ("space = cosine", "")]),
        table("DOCUMENT", orange, [("source", "PK"), ("collection", "FK"), ("file_type", "")]),
        table("CHUNK", orange, [("chunk_id = source::i", "PK"), ("source", "FK"), ("chunk_index", ""), ("text", ""), ("embedding[384]", "")]),
        table("CHAT_TURN", "#f3e3ec", [("turn", "PK"), ("message", ""), ("history[]", ""), ("ecg_id", "FK, optional"), ("answer", "")]),
        table("TOOL_CALL", "#f3e3ec", [("turn", "FK"), ("tool", ""), ("arguments", ""), ("result", "")]),
        table("AGENT_STEP", "#f3e3ec", [("turn", "FK"), ("step_no", ""), ("type (llm / tool)", ""), ("provider, model", "FK"), ("duration_ms", ""), ("tokens", "")]),
        table("LLM_PROVIDER", "#f3e3ec", [("provider", "PK"), ("model", ""), ("role (main / fallback)", ""), ("api_key (from .env)", "")]),
    ]
    rel = [
        ("PATIENT_RECORD", "ANNOTATION", "1", "N", "has"),
        ("PATIENT_RECORD", "BEAT", "1", "N", "segmented into"),
        ("BEAT", "MODEL_VERSION", "N", "1", "trains / evaluates"),
        ("ECG_UPLOAD", "ANALYSIS_RESULT", "1", "N", "analysed as"),
        ("MODEL_VERSION", "ANALYSIS_RESULT", "1", "N", "produces"),
        ("ANALYSIS_RESULT", "ABNORMAL_BEAT", "1", "N", "lists"),
        ("COLLECTION", "DOCUMENT", "1", "N", "stores"),
        ("DOCUMENT", "CHUNK", "1", "N", "split into"),
        ("CHAT_TURN", "TOOL_CALL", "1", "N", "makes"),
        ("CHAT_TURN", "AGENT_STEP", "1", "N", "traced as"),
        ("AGENT_STEP", "LLM_PROVIDER", "N", "1", "answered by"),
        ("TOOL_CALL", "CHUNK", "N", "N", "retrieves"),
        ("TOOL_CALL", "ANALYSIS_RESULT", "N", "1", "runs"),
        ("CHAT_TURN", "ECG_UPLOAD", "N", "1", "refers to"),
    ]
    arrow = {"1": "tee", "N": "crow"}
    edges = "\n".join(f'{a} -> {b} [arrowtail={arrow[ca]}, arrowhead={arrow[cb]}, dir=both, label="{lab}"];' for a, b, ca, cb, lab in rel)
    dot("erd", f"""digraph ERD {{
rankdir=TB; nodesep=0.25; ranksep=0.45; ratio=compress; size="10,15"; {FONT}; node [{FONT}, shape=plaintext, fontsize=14]; edge [{FONT}, fontsize=12, color="#52514e", fontcolor="#52514e"];
{chr(10).join(ents)}
{edges}
}}""")


def sequence():
    dot("chat_flow", f"""digraph G {{
rankdir=TB; nodesep=0.3; ranksep=0.3; {FONT}; node [{FONT}, shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#52514e", fontsize=10, width=3.6];
edge [{FONT}, fontsize=9, color="#52514e"];
a [label="1. Client: POST /ecg/upload-csv → ecg_id", fillcolor="#e8f0fb"];
b [label="2. Client: POST /agent/chat {{message, ecg_id}}", fillcolor="#e8f0fb"];
c [label="3. Agent sends system prompt, history, message + tool schemas\\nto OpenAI gpt-4o-mini (Gemini if OpenAI fails)"];
d [label="4. OpenAI returns tool calls, e.g. analyze_ecg(ecg_id),\\nsearch_knowledge_base(\\"premature ventricular contraction\\")", fillcolor="#fdeee7"];
e [label="5. ToolBox validates arguments and runs the tools: Conv-BiLSTM\\non the stored ECG; Chroma similarity search over embedded chunks", fillcolor="#e8f6f0"];
f [label="6. Tool results (JSON) appended as tool messages; OpenAI called again"];
g [label="7. OpenAI writes the answer grounded in tool output and cited sources", fillcolor="#fdeee7"];
h [label="8. Response: answer, tool calls, sources, model used, step trace", fillcolor="#e8f0fb"];
a -> b -> c -> d -> e -> f -> g -> h; f -> d [label=" more tool calls\\n (≤ 6 steps)", style=dashed, constraint=false];
}}""")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "models" / "ecg_lstm_meta.json").read_text())
    ev = json.loads((ROOT / "reports" / "evaluation.json").read_text())
    perf = json.loads((ROOT / "reports" / "performance.json").read_text())
    training_curves(meta)
    confusion(ev)
    reliability(perf)
    baselines(meta)
    ecg_strip()
    architecture()
    model_diagram()
    erd()
    sequence()


if __name__ == "__main__":
    main()
