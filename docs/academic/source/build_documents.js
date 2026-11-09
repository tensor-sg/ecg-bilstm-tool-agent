// Builds the SASTRA synopsis and the project report (.docx) from the project's committed results.
// Usage: node build.js <project_root>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, LevelFormat, HeadingLevel, PageBreak, Footer, PageNumber, TabStopType,
} = require("docx");

const ROOT = process.argv[2];
const OUT = path.join(ROOT, "docs", "academic");
fs.mkdirSync(OUT, { recursive: true });
const J = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const meta = J("models/ecg_lstm_meta.json");
const ev = J("reports/evaluation.json");
const perf = J("reports/performance.json");

// ------------------------------------------------------------------ student details
const S = {
  name: "Priyatosh Satpati",
  reg: "25113170096",
  email: "priyatoshsatpatiofficial@gmail.com",
  guide: "Mr. G. Swaminathan",
  rmSap: "C2385",
  company: "SASTRA DEEMED TO BE UNIVERSITY, TIRUMALAISAMUDRAM, THANJAVUR – 613 401, TAMIL NADU, INDIA",
  programme: "Master of Science in Data Science",
  semester: "Fourth Semester",
  university: "SASTRA Deemed to be University",
  uniPlace: "Thanjavur, Tamil Nadu, India",
  month: "November 2026",
  title: "ECG Arrhythmia Beat Classification Using a Convolutional Bidirectional LSTM Deployed as a Tool for a Retrieval-Augmented LLM Agent",
};

const FONT = "Times New Roman";
const f3 = (x) => Number(x).toFixed(3);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const ci = (v) => `[${f3(v[0])}–${f3(v[1])}]`;

// ------------------------------------------------------------------ building blocks
function runs(text, base = {}) {
  // **bold**, *italic* inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ alignment: o.align ?? AlignmentType.JUSTIFIED, spacing: { line: 360, after: o.after ?? 120, before: o.before ?? 0 }, indent: o.indent, keepNext: o.keepNext, children: runs(text, o.run || {}) });
const C = (text, o = {}) => P(text, { ...o, align: AlignmentType.CENTER });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(text)] });
const H1nb = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 60 }, children: runs(text) });
const N = (text, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 60 }, children: runs(text) });
const EQ = (text) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 }, children: [new TextRun({ text, italics: true })] });
const BR = () => new Paragraph({ children: [new PageBreak()] });

let figNo = 0, tabNo = 0;
const figures = [], tables = [];
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function FIG(file, caption, widthIn = 5.7) {
  widthIn = Math.min(widthIn, 5.7);
  const full = path.join(ROOT, "docs", "figures", file);
  const { w, h } = pngSize(full);
  const wpx = Math.round(widthIn * 96), hpx = Math.round((wpx * h) / w);
  figNo += 1;
  figures.push(`Figure ${figNo}: ${caption}`);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width: wpx, height: hpx }, altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: `Figure ${figNo}: `, bold: true, size: 20 }), new TextRun({ text: caption, size: 20 })] }),
  ];
}
const figRef = () => `Figure ${figNo + 1}`;
const tabRef = () => `Table ${tabNo + 1}`;

const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function TBL(caption, headers, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (!o.nocap) { tabNo += 1; tables.push(`Table ${tabNo}: ${caption}`); }
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: (o.align && o.align[i]) || AlignmentType.LEFT, children: runs(String(t), { size: o.size || 20, bold: head || undefined }) })],
  });
  return [
    o.nocap ? new Paragraph({ children: [] }) : new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 }, children: [new TextRun({ text: `Table ${tabNo}: `, bold: true, size: 20 }), new TextRun({ text: caption, size: 20 })] }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }), ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const R = AlignmentType.RIGHT, L = AlignmentType.LEFT, CEN = AlignmentType.CENTER;

const styles = {
  default: { document: { run: { font: FONT, size: 24 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 240, after: 240 }, alignment: AlignmentType.CENTER, keepNext: true, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
  ],
};
const numbering = {
  config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    ...["numbers", "obj", "contrib", "future"].map((reference) => ({ reference, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] })),
  ],
};
const A4 = { width: 11906, height: 16838 };

// ================================================================== SYNOPSIS
const ABSTRACT = [
  "Cardiac arrhythmias are usually identified from long ambulatory electrocardiogram (ECG) recordings, where manual beat-by-beat review is slow and error-prone. Many published deep-learning heartbeat classifiers report near-perfect accuracy, but they often split beats randomly, so beats from the same patient appear in both training and test data and the reported performance does not carry over to new patients. This project builds a heartbeat classifier that is evaluated on unseen patients and makes it usable through a conversational assistant.",
  "Single-lead recordings from the MIT-BIH Arrhythmia Database are band-pass filtered, segmented around each R-peak and grouped into the AAMI classes normal (N), supraventricular ectopic (S) and ventricular ectopic (V). A compact convolutional bidirectional long short-term memory (Conv-BiLSTM) network of about 50,000 parameters combines beat morphology with rhythm features expressed relative to each patient's own heart rate. Model selection uses patients held out from training, and final testing uses 22 patients never seen during development, reported with patient-level bootstrap confidence intervals and compared against baselines.",
  `On the unseen patients the model reaches ${pct(meta.test_metrics.accuracy)} accuracy and a macro-F1 score of ${meta.test_metrics.macro_f1.toFixed(2)}, against ${meta.baselines_test["RR-only logistic regression"].macro_f1.toFixed(2)} for a rhythm-only baseline. It detects ${pct(meta.test_metrics.per_class.V.sensitivity, 0)} of ventricular beats with ${pct(meta.test_metrics.per_class.V.ppv, 0)} precision, and its probabilities are well calibrated. Supraventricular beats remain difficult, and this limitation is reported explicitly.`,
  "The model is deployed in a FastAPI web service with an ECG signal-quality check. An OpenAI language model acts as the agentic core: it decides when to call the classifier as a tool and when to retrieve reference documents from a persistent Chroma vector database (retrieval-augmented generation), with a Gemini model as automatic fallback. Answers about an uploaded ECG therefore rest on both the measurement and cited sources. The system is verified by 119 automated tests, including live tests against the real language-model APIs.",
];
const words = ABSTRACT.join(" ").split(/\s+/).filter(Boolean).length;
if (words > 300) throw new Error(`abstract has ${words} words (max 300)`);

function synopsis() {
  const f14 = { font: FONT, size: 28 };
  const line = (label, value, o = {}) => new Paragraph({
    tabStops: [{ type: TabStopType.LEFT, position: 3600 }],
    indent: { left: 3780, hanging: 3780 },
    spacing: { after: o.after ?? 0 },
    children: [new TextRun({ text: label, ...f14 }), new TextRun({ text: "\t: ", ...f14 }),
      new TextRun({ text: value, ...f14, highlight: o.highlight })],
  });
  const children = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: S.title, bold: true, ...f14 })] }),
    line("Student Name", S.name.toUpperCase()),
    line("Student Registration Number", S.reg, { after: 160 }),
    line("Approved Guide Name", S.guide.toUpperCase(), { after: 160 }),
    line("Name of the Company with address", S.company, { after: 120 }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 }, children: [new TextRun({ text: "Abstract / Problem Statement", bold: true, ...f14 })] }),
    ...ABSTRACT.map((t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 0 }, children: [new TextRun({ text: t, font: FONT, size: 24 })] })),
    new Paragraph({ spacing: { before: 360 }, tabStops: [{ type: TabStopType.RIGHT, position: 9026 }], children: [
      new TextRun({ text: "Signature of the Student", font: FONT, size: 24 }), new TextRun({ text: "\tSignature of the Reporting Manager", font: FONT, size: 24 })] }),
    new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: 9026 }], children: [
      new TextRun({ text: `(${S.name.toUpperCase()})`, font: FONT, size: 24 }), new TextRun({ text: `\t(${S.guide.toUpperCase()})`, font: FONT, size: 24 })] }),
    new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: 9026 }], children: [
      new TextRun({ text: `Reg. No. ${S.reg}`, font: FONT, size: 24 }), new TextRun({ text: `\tSAP ID: ${S.rmSap}`, font: FONT, size: 24 })] }),
  ];
  return new Document({
    creator: S.name, title: S.title, description: "Project synopsis", styles: { default: { document: { run: { font: FONT, size: 24 } } } },
    sections: [{ properties: { page: { size: A4, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children }],
  });
}

// ================================================================== REPORT
const T = meta.test_metrics, CI = meta.test_ci95, BL = meta.baselines_test;
const E2E = ev.end_to_end, DET = E2E.detection, ANN = ev.annotated;
const PQ = perf.probability, SP = perf.speed, ER = perf.errors;

function front() {
  const c = (text, size, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 120, before: o.before ?? 0 }, children: [new TextRun({ text, size, bold: o.bold, italics: o.italics, font: FONT })] });
  return [
    c(S.title.toUpperCase(), 32, { bold: true, after: 480, before: 600 }),
    c("A project report submitted in partial fulfilment of the requirements", 24),
    c("for the award of the degree of", 24),
    c(S.programme.toUpperCase(), 28, { bold: true, after: 480 }),
    c("Submitted by", 24, { italics: true }),
    c(S.name.toUpperCase(), 28, { bold: true }),
    c(`Registration Number: ${S.reg}`, 24),
    c(S.email, 24, { after: 480 }),
    c("Under the guidance of", 24, { italics: true }),
    c(S.guide.toUpperCase(), 28, { bold: true, after: 720 }),
    c(S.university.toUpperCase(), 28, { bold: true }),
    c(S.uniPlace, 24),
    c(`${S.semester} · ${S.month}`, 24),

    H1("BONAFIDE CERTIFICATE"),
    P(`This is to certify that the project report titled **“${S.title}”** is a bonafide record of the work carried out by **${S.name.toUpperCase()}** (Registration Number **${S.reg}**) in partial fulfilment of the requirements for the award of the degree of ${S.programme} of ${S.university}, during the ${S.semester.toLowerCase()} of the programme, under my guidance and supervision.`),
    new Paragraph({ spacing: { before: 1200 }, tabStops: [{ type: TabStopType.RIGHT, position: 8306 }], children: [new TextRun("Signature of the Guide"), new TextRun("\tDate:")] }),
    new Paragraph({ children: [new TextRun({ text: S.guide.toUpperCase(), bold: true })] }),
    new Paragraph({ children: [new TextRun("Project Guide and Reporting Manager")] }),
    new Paragraph({ children: [new TextRun(`SAP ID: ${S.rmSap}`)] }),

    H1("DECLARATION"),
    P(`I, **${S.name.toUpperCase()}** (Registration Number **${S.reg}**), declare that the project report titled **“${S.title}”** submitted to ${S.university} is a record of original work done by me under the guidance of ${S.guide}. The source code, experiments, results and text are my own work. All datasets, publications and software used are acknowledged and cited in the References. The project was carried out at ${S.university}. This report does not contain any confidential information or the name of any customer, and it has not been submitted for the award of any other degree or diploma.`),
    new Paragraph({ spacing: { before: 1200 }, tabStops: [{ type: TabStopType.RIGHT, position: 8306 }], children: [new TextRun("Place:"), new TextRun("\tSignature of the Student")] }),
    new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: 8306 }], children: [new TextRun("Date:"), new TextRun({ text: `\t${S.name.toUpperCase()}`, bold: true })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun(S.email)] }),

    H1("ACKNOWLEDGEMENT"),
    P(`I thank my guide and reporting manager, **${S.guide}**, for reviewing the scope, methodology and results of this project and for his guidance throughout. I thank the faculty of the ${S.programme} programme at ${S.university} for the coursework this project builds on.`),
    P("I acknowledge the creators and maintainers of the MIT-BIH Arrhythmia Database and PhysioNet, whose openly available, expert-annotated recordings made this work possible, and the developers of the open-source libraries listed in the References."),
    new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 480 }, children: [new TextRun({ text: S.name.toUpperCase(), bold: true })] }),

    H1("ABSTRACT"),
    ...ABSTRACT.map((t) => P(t)),
    P("**Keywords:** electrocardiogram, arrhythmia, heartbeat classification, Conv-BiLSTM, inter-patient evaluation, MIT-BIH, retrieval-augmented generation, LLM tool calling, Chroma, FastAPI."),
  ];
}

function contents() {
  const chapters = [
    "1. Introduction", "2. Literature Review", "3. Dataset and Data Understanding", "4. Methodology",
    "5. System Design", "6. Implementation", "7. Testing and Validation", "8. Results and Discussion",
    "9. Limitations and Ethical Considerations", "10. Conclusion and Future Work", "References",
    "Appendix A: How to Reproduce", "Appendix B: Complete Test Case Catalogue",
  ];
  return [
    H1("TABLE OF CONTENTS"),
    ...chapters.map((c) => new Paragraph({ spacing: { after: 80 }, children: [new TextRun(c)] })),
    H1("LIST OF FIGURES"),
    ...figures.map((f) => new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: f, size: 22 })] })),
    H1("LIST OF TABLES"),
    ...tables.map((t) => new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: t, size: 22 })] })),
    H1("LIST OF ABBREVIATIONS"),
    ...TBL("Abbreviations", ["Abbreviation", "Meaning"], [
      ["AAMI", "Association for the Advancement of Medical Instrumentation"], ["API", "Application Programming Interface"],
      ["AUC", "Area Under the ROC Curve"], ["BiLSTM", "Bidirectional Long Short-Term Memory"], ["CI", "Confidence Interval"],
      ["CNN / Conv1D", "(One-dimensional) Convolutional Neural Network"], ["DS1 / DS2", "Training / test record sets of de Chazal et al. [3]"],
      ["ECE", "Expected Calibration Error"], ["ECG", "Electrocardiogram"], ["ERD", "Entity Relationship Diagram"],
      ["F / Q", "Fusion / unclassifiable (paced) beats"], ["LLM", "Large Language Model"], ["MLII", "Modified limb lead II"],
      ["N / S / V", "Normal / supraventricular ectopic / ventricular ectopic beats"], ["PPV", "Positive Predictive Value (precision)"],
      ["PVC", "Premature Ventricular Contraction"], ["RAG", "Retrieval-Augmented Generation"], ["ROC", "Receiver Operating Characteristic"],
      ["RR interval", "Time between consecutive R-peaks"], ["Se", "Sensitivity (recall)"], ["SDNN", "Standard deviation of normal-to-normal RR intervals"],
    ], [2400, 5906], { nocap: true }),
  ];
}

function body() {
  const out = [];
  const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? out.push(...x) : out.push(x)));

  // ---------------------------------------------------------------- 1
  add(H1("CHAPTER 1: INTRODUCTION"), H2("1.1 Background"),
    P("The electrocardiogram (ECG) records the electrical activity of the heart and is the primary tool for detecting arrhythmias, i.e. disturbances of the heart's rhythm. Many arrhythmias are intermittent, so patients are monitored with ambulatory (Holter) recorders for 24 hours or longer. A single day contains roughly 100,000 heartbeats, and reviewing them manually is slow, tiring and prone to error. Automatic heartbeat classification therefore has clear clinical value as a screening aid."),
    P("Two developments shape this project. First, deep learning models that combine convolutional and recurrent layers can learn beat morphology directly from the raw signal [8]–[11]. Second, large language models (LLMs) can call external tools and ground their answers in retrieved documents [14]–[16]. Combining the two allows a clinician or researcher to ask questions in natural language about an uploaded ECG and receive an answer based on an actual measurement and on cited reference material, rather than on the LLM's general knowledge."),
    H2("1.2 Problem Statement"),
    P("Published ECG classifiers frequently report accuracies of 98–99%, but many of them split heartbeats randomly into training and test sets. Because beats from one patient are highly similar, the model is then tested on people it has already seen, and the result does not describe its performance on a new patient [3], [13]. In addition, a trained classifier on its own is difficult for non-programmers to use and gives no context for its output."),
    P("The problem addressed is therefore: **to build an ECG heartbeat classifier whose performance is measured on patients it has never seen, to report that performance honestly with its uncertainty and limitations, and to make the classifier usable through a conversational agent whose answers are grounded in the model's output and in a searchable knowledge base.**"),
    H2("1.3 Objectives"),
    N("Explore the MIT-BIH Arrhythmia Database and build a preprocessing pipeline (filtering, R-peak detection, beat segmentation, rhythm features) shared by training and deployment.", "obj"),
    N("Train a compact Conv-BiLSTM classifier for the AAMI classes N, S and V that runs on an ordinary laptop CPU.", "obj"),
    N("Evaluate it with a strict inter-patient protocol, baselines, patient-level confidence intervals, end-to-end detection, calibration and speed measurements.", "obj"),
    N("Deploy the model in a FastAPI service with a persistent Chroma vector store for retrieval-augmented generation (RAG).", "obj"),
    N("Expose the classifier and the knowledge base as tools to an LLM agent whose agentic core is an OpenAI model, with a Gemini fallback, and as a separate tools API usable without an LLM.", "obj"),
    N("Verify the system with automated unit, API, integration, model-accuracy, model-performance and live LLM tests.", "obj"),
    H2("1.4 Scope"),
    P("The system classifies individual beats of a single-lead ECG into N, S and V. Fusion (F) and paced or unclassifiable (Q) beats are outside the trained classes; how the model labels them is reported. The system is a research and education tool and is **not** a medical device. The LLM (OpenAI gpt-4o-mini, with Gemini 2.5 Flash as fallback) is an external service and is not part of the trained model."),
    H2("1.5 Organisation of the Report"),
    P("Chapter 2 reviews related work. Chapter 3 describes the dataset. Chapter 4 presents the methodology, Chapter 5 the system design including the entity relationship diagram, and Chapter 6 the implementation. Chapter 7 describes the testing strategy and test cases, and Chapter 8 the results. Chapter 9 discusses limitations and ethics, and Chapter 10 concludes."),
  );

  // ---------------------------------------------------------------- 2
  add(H1("CHAPTER 2: LITERATURE REVIEW"), H2("2.1 Data and Evaluation Standards"),
    P("The MIT-BIH Arrhythmia Database [1], distributed through PhysioNet [2], is the most widely used benchmark for beat classification. The AAMI EC57 standard [4] groups its beat annotations into five classes (N, S, V, F, Q) and defines how sensitivity and positive predictivity are reported. De Chazal et al. [3] proposed splitting the 44 non-paced records into two sets of 22 patients (DS1 for training, DS2 for testing) so that no patient contributes to both. This *inter-patient* protocol has become the reference for realistic evaluation."),
    H2("2.2 Feature-Based Methods"),
    P("Early systems detected the QRS complex with the Pan–Tompkins algorithm [5] and classified beats from hand-crafted morphology and RR-interval features with linear discriminants [3] or ensembles of classifiers [12]. Llamedo and Martínez [13] showed that selecting features for generalisation across databases improves performance on unseen patients, and that RR-interval features normalised to the patient's own rhythm are among the most robust inputs."),
    H2("2.3 Deep Learning Methods"),
    P("Kiranyaz et al. [8] used one-dimensional CNNs on raw beats, Acharya et al. [9] trained a nine-layer CNN, and Yildirim [11] used bidirectional LSTMs [6], [7] on wavelet sequences. Hannun et al. [10] trained a deep network on a very large single-lead dataset and reported cardiologist-level performance on rhythm classes. Many deep-learning papers on MIT-BIH, however, use random beat-level splits; their very high accuracies are not comparable to inter-patient results, which are substantially lower, especially for supraventricular beats [3], [13]."),
    H2("2.4 Calibration and Uncertainty"),
    P("Guo et al. [21] showed that modern neural networks are often over-confident and proposed the expected calibration error (ECE) to measure the gap between confidence and accuracy. The Brier score [24] measures the overall quality of probabilistic predictions. Because beats from one patient are correlated, confidence intervals should be computed by resampling patients rather than beats, using the bootstrap [20]."),
    H2("2.5 LLM Agents, Tool Use and RAG"),
    P("Retrieval-augmented generation [14] grounds an LLM's answer in passages retrieved from an external store, reducing unsupported statements and allowing sources to be cited. ReAct [15] interleaves reasoning with tool calls, and Toolformer [16] showed that language models can learn when to call external APIs. Dense retrieval relies on sentence embeddings such as Sentence-BERT [17]; MiniLM [18] provides a small embedding model that runs on a CPU."),
    H2("2.6 Research Gap"),
    P("Few works combine (i) a beat classifier evaluated with a strict inter-patient protocol and patient-level uncertainty, (ii) deployment as a service with tests that guard its accuracy, and (iii) integration as a tool for an LLM agent with retrieval over reference documents. This project addresses that combination with modest hardware and fully open data."),
  );

  // ---------------------------------------------------------------- 3
  const tr = { N: 36436, S: 686, V: 2936 }, va = { N: 9412, S: 258, V: 852 }, te = { N: 44240, S: 1837, V: 3220 };
  add(H1("CHAPTER 3: DATASET AND DATA UNDERSTANDING"), H2("3.1 MIT-BIH Arrhythmia Database"),
    P("The database contains 48 half-hour two-channel ambulatory ECG recordings from 47 subjects, digitised at 360 Hz with 11-bit resolution over a 10 mV range [1]. Every beat was annotated independently by two or more cardiologists. The database is available on PhysioNet [35]. This project uses the modified limb lead II (MLII), which is available in all non-paced records. The data is downloaded from PhysioNet [2] with the WFDB Python package and is distributed under the Open Data Commons Attribution License."),
    H2("3.2 AAMI Class Mapping"),
    ...TBL("Mapping of MIT-BIH beat symbols to AAMI EC57 classes [4]", ["AAMI class", "MIT-BIH symbols", "Meaning"], [
      ["N", "N, L, R, e, j", "Normal, left/right bundle branch block, atrial/nodal escape"],
      ["S", "A, a, J, S", "Atrial, aberrated atrial, nodal and supraventricular premature beats"],
      ["V", "V, E", "Premature ventricular contraction, ventricular escape"],
      ["F", "F", "Fusion of ventricular and normal beat (not modelled)"],
      ["Q", "/, f, Q", "Paced, fusion of paced and normal, unclassifiable (not modelled)"],
    ], [1300, 2000, 5006]),
    H2("3.3 Patient-Wise Data Split"),
    P(`The four paced records (102, 104, 107, 217) are excluded. The remaining records follow the inter-patient split of de Chazal et al. [3]. Five DS1 patients are held out as a validation set for early stopping and model selection, so the DS2 test patients are used only once, for the final evaluation (${tabRef()}).`),
    ...TBL("Patient-wise split and beat counts per class", ["Set", "Records", "Patients", "N", "S", "V", "Purpose"], [
      ["Train", meta.train_records.join(", "), "17", tr.N.toLocaleString(), tr.S.toLocaleString(), tr.V.toLocaleString(), "Fit weights"],
      ["Validation", meta.val_records.join(", "), "5", va.N.toLocaleString(), va.S.toLocaleString(), va.V.toLocaleString(), "Early stopping, model selection"],
      ["Test (DS2)", meta.test_records.join(", "), "22", te.N.toLocaleString(), te.S.toLocaleString(), te.V.toLocaleString(), "Final evaluation only"],
    ], [1000, 3006, 800, 800, 600, 600, 1500], { size: 18, align: [L, L, R, R, R, R, L] }),
    H2("3.4 Observations from Exploratory Analysis"),
    B("**Severe class imbalance.** Normal beats make up about 90% of all beats, so a model that always predicts N already reaches about 90% accuracy. Accuracy alone is therefore misleading, and macro-F1 is used for model selection."),
    B("**Patient concentration.** Rare classes are concentrated in few patients. In the test set, 1,382 of the 1,837 S beats come from a single record (232), and 373 of the 415 fusion beats available for training come from record 208."),
    B("**Rhythm is informative.** Premature beats arrive early relative to the patient's own rhythm and are often followed by a compensatory pause, which motivates the patient-relative RR features of Section 4.1."),
    B("**Noise and baseline wander.** The power spectrum shows low-frequency baseline wander and high-frequency noise, removed by a 0.5–40 Hz band-pass filter (notebook 01)."),
  );

  // ---------------------------------------------------------------- 4
  add(H1("CHAPTER 4: METHODOLOGY"), H2("4.1 Preprocessing Pipeline"),
    P("The same module (app/ecg/preprocessing.py) is used by the training notebooks and by the deployed service, so the beats seen at inference are built exactly like the training beats."),
    N("**Resampling** to 360 Hz, so recordings from devices sampling at, for example, 250 Hz are accepted."),
    N("**Band-pass filtering** at 0.5–40 Hz with a third-order zero-phase Butterworth filter."),
    N("**R-peak detection**: annotated peaks during training; at inference a Pan–Tompkins-style detector [5] (derivative, squaring, 150 ms moving-window integration, adaptive threshold, 250 ms refractory period, refinement to the local maximum) or peaks supplied by the caller."),
    N("**Segmentation**: a 256-sample window per beat, 90 samples (0.25 s) before and 166 samples (0.46 s) after the R-peak, z-score normalised per beat."),
    N("**Rhythm features**, all relative to the patient's own rhythm: previous and next RR interval divided by the local mean RR of about 10 beats, previous RR divided by the recording's median RR, and the ratio of previous to next RR. Absolute RR values in seconds are deliberately excluded, because they identify individual patients and caused over-fitting in an earlier version."),
    N("**Signal-quality index**: the excess kurtosis of each 10-second window of the filtered signal. QRS complexes make a clean ECG strongly peaked (median kurtosis about 10 on MIT-BIH), while noise and flat or periodic artefacts score near or below 0 [28]. A recording with more than half of its windows below 1.5 is flagged as unreliable. This check was added after an integration test showed that the detector reports beats in pure noise (Section 7.6)."),
    H2("4.2 Model Architecture"),
    P(`The classifier has two inputs (${figRef()}). The beat waveform passes through two convolutional blocks that learn local morphology (QRS width, T-wave shape) and shorten the sequence from 256 to 64 steps, followed by two bidirectional LSTM layers [6], [7] that model the temporal order of the waves. The four rhythm features pass through batch normalisation [22] and a dense layer. Both branches are concatenated and classified by a dense layer with dropout [23] and a three-way softmax. Gaussian noise layers act as data augmentation during training only. The network has 49,571 parameters (0.71 MB).`),
    ...FIG("model.png", "Conv-BiLSTM architecture with waveform and rhythm branches", 3.6),
    H2("4.3 Training Procedure"),
    ...TBL("Training configuration", ["Setting", "Value", "Reason"], [
      ["Optimiser", "Adam [19], learning rate 0.001, halved on plateau", "Standard, robust default"],
      ["Batch size / max epochs", `${meta.hyperparameters.batch_size} / ${meta.hyperparameters.max_epochs}`, "Fits a 2-core laptop CPU (about 60 s per epoch)"],
      ["Loss", "Categorical cross-entropy with balanced class weights", "Counteracts the 90% majority of N beats"],
      ["Model selection", `Validation macro-F1, early stopping (patience ${meta.hyperparameters.patience})`, "Accuracy is dominated by N; loss does not reflect rare classes"],
      ["Augmentation", "Gaussian noise σ = 0.1 on beats, 0.03 on RR features", "Robustness to noise and unseen morphology"],
      ["Seed", String(meta.hyperparameters.seed), "Reproducibility"],
    ], [2100, 3300, 2906]),
    P(`Training stopped after ${meta.epochs_run} epochs; the checkpoint from epoch ${meta.best_epoch} had the best validation macro-F1 (${meta.val_metrics.macro_f1.toFixed(3)}) and was kept (${figRef()}). Validation macro-F1 fluctuates between epochs because the validation set contains only five patients; selecting on it, rather than on training loss, avoids keeping a model that has started to over-fit.`),
    ...FIG("training_curves.png", "Training and validation curves; the dashed line marks the kept checkpoint", 6.0),
    H2("4.4 Evaluation Design"),
    P("For each class *c*, with true positives TP, false negatives FN, false positives FP and true negatives TN:"),
    EQ("Sensitivity Se = TP / (TP + FN)        PPV = TP / (TP + FP)        Specificity = TN / (TN + FP)"),
    EQ("F1 = 2 · Se · PPV / (Se + PPV)        macro-F1 = mean of F1 over N, S, V        balanced accuracy = mean of Se over N, S, V"),
    B("**Baselines**: a majority-class model (always N) and a logistic regression on the four RR features only, fitted on the same training patients."),
    B("**Confidence intervals**: 1,000 bootstrap resamples of the 22 test *patients* [20]; beat-level resampling would understate the uncertainty."),
    B("**End-to-end evaluation**: the deployed R-peak detector replaces the annotations; detected beats are matched to annotations within ±150 ms, as in AAMI EC57 [4]."),
    B("**Probability quality**: one-vs-rest ROC AUC, Brier score [24], ECE with 10 bins [21], and selective accuracy at confidence thresholds."),
    B("**Significance**: a paired patient-level bootstrap comparing the macro-F1 of the model and the RR-only baseline on the same resampled patients."),
    H2("4.5 Model Development History"),
    ...TBL("Model versions and why each was kept or rejected", ["Version", "Change", "DS2 macro-F1", "Outcome"], [
      ["v1", "4 classes, random beat-level validation split, absolute and relative RR features", "not evaluated", "Discarded: validation contained training patients (leakage)"],
      ["v2", "Patient-wise validation, macro-F1 selection, 4 classes (N/S/V/F)", "0.458", "Rejected: about 11% of normal beats labelled F (memorised patient 208); S Se 0.07"],
      ["v3", "N/S/V only, patient-relative RR features, balanced weights, noise augmentation", T.macro_f1.toFixed(3), "Final model"],
    ], [900, 3500, 1200, 2706], { align: [L, L, R, L] }),
  );

  // ---------------------------------------------------------------- 5
  add(H1("CHAPTER 5: SYSTEM DESIGN"), H2("5.1 Architecture"),
    P(`${figRef()} shows the components. The design separates the **agentic core** from the **tools**. The agent (agent.py) holds the conversation with the LLM: OpenAI gpt-4o-mini decides which tools to call and writes the answer, and Gemini 2.5 Flash takes over automatically if the OpenAI call fails. The tools (tools.py) are plain functions with JSON schemas: ECG analysis by the Conv-BiLSTM, semantic search of the Chroma knowledge base, and listing of uploaded recordings. They know nothing about the LLM.`),
    P("The FastAPI application exposes this separation as four groups of endpoints: the **Agent API** (/agent) runs the agent loop and returns a step-by-step trace; the **Tools API** (/tools) lists the tool schemas and invokes any tool directly, so another program or agent can use the LSTM exactly as the LLM does; the **ECG** and **Document** endpoints upload recordings and reference documents. All services are created once at start-up (FastAPI lifespan) and shared by the requests."),
    ...FIG("architecture.png", "System architecture", 6.2),
    H2("5.2 Entity Relationship Diagram"),
    P(`${figRef()} shows the logical data model. Three groups of entities can be distinguished:`),
    B("**Training data (grey and green)**: a PATIENT_RECORD has many ANNOTATIONs and is segmented into many BEATs; beats train and evaluate a MODEL_VERSION, whose metadata (classes, hyperparameters, split, metrics) is stored in ecg_lstm_meta.json."),
    B("**Runtime ECG analysis (blue)**: an ECG_UPLOAD is analysed by a MODEL_VERSION into an ANALYSIS_RESULT (heart rate, SDNN, beat counts, signal quality), which lists ABNORMAL_BEATs with time and confidence."),
    B("**Knowledge base and agent (orange and pink)**: a Chroma COLLECTION stores DOCUMENTs, each split into CHUNKs with a 384-dimensional embedding and the identifier source::index. A CHAT_TURN may refer to an ECG_UPLOAD and makes TOOL_CALLs, which retrieve chunks or run an analysis. Each turn is traced as AGENT_STEPs (LLM decisions and tool runs with duration and tokens), each LLM step answered by an LLM_PROVIDER (OpenAI as main, Gemini as fallback)."),
    ...FIG("erd.png", "Entity relationship diagram (crow's foot notation)", 6.2),
    P("Physically, chunks, embeddings and collection metadata live in Chroma's on-disk store (chroma_db/), the model and its metadata in models/, and uploaded ECGs in an in-memory registry. The collection records the name of the embedding model that built it, and the service refuses to mix embeddings from different models."),
    H2("5.3 Request Flow of the Agent"),
    P(`${figRef()} shows a typical question about an uploaded ECG. The agent sends the conversation and the tool schemas to the LLM; the LLM requests tool calls; the agent executes them against the real classifier and vector store and returns the JSON results; the LLM then writes an answer grounded in those results. The loop is limited to six LLM calls. Every LLM decision and tool execution is recorded in a trace with its duration and token usage, which the client receives with the answer. Failures of both providers are returned as HTTP 502 with a configuration hint.`),
    ...FIG("chat_flow.png", "Request flow for a question about an uploaded ECG", 4.2),
    H2("5.4 API Endpoints"),
    ...TBL("REST endpoints", ["Method", "Path", "Purpose"], [
      ["GET", "/health", "Status, vector collection, model loaded, LLM in use"],
      ["GET", "/agent", "Agent configuration: provider, model, fallback, tools, system prompt"],
      ["POST", "/agent/chat", "Run the agent: answer, tool calls, sources, model used, step trace"],
      ["GET", "/tools", "Tool catalogue with JSON schemas (what the LLM is offered)"],
      ["GET", "/tools/{name}", "Schema of one tool"],
      ["POST", "/tools/{name}/invoke", "Run a tool directly without an LLM (404, 422, 503 on errors)"],
      ["POST", "/documents/upload", "Upload PDF, TXT or Markdown; chunk, embed and store"],
      ["POST", "/documents/text", "Ingest raw text under a source name"],
      ["POST", "/documents/search", "Semantic search with optional source filter"],
      ["DELETE", "/documents/{source}", "Remove all chunks of a document"],
      ["GET", "/documents/count", "Number of stored chunks"],
      ["POST", "/ecg/upload", "Upload an ECG as JSON (signal, sampling rate, optional R-peaks)"],
      ["POST", "/ecg/upload-csv", "Upload an ECG as CSV"],
      ["GET", "/ecg/", "List uploaded recordings"],
      ["POST", "/ecg/{ecg_id}/analyze", "Run the classifier; returns heart rate, counts, abnormal beats, signal quality"],
      ["POST", "/chat", "Alias of /agent/chat for existing clients"],
    ], [1000, 2700, 4606]),
    H2("5.5 RAG Design"),
    P("Documents are split with a recursive character splitter into chunks of 800 characters with 120 characters of overlap. Chunks are embedded locally with all-MiniLM-L6-v2 [17], [18] (ONNX runtime, no API key) or optionally with an OpenAI-compatible embedding endpoint. Chroma indexes the vectors with HNSW under cosine distance; search results are returned with a similarity score of 1 − distance. Re-uploading a document replaces its chunks instead of duplicating them."),
    H2("5.6 Agent and LLM Providers"),
    P("Providers are selected in the configuration file. LLM_PROVIDER=openai uses OPENAI_API_KEY and gpt-4o-mini; gemini uses GEMINI_API_KEY through Google's OpenAI-compatible endpoint; custom accepts any OpenAI-compatible server such as Ollama or vLLM. LLM_FALLBACK_PROVIDER names the provider used when the main one raises an API error (authentication, quota, outage); once switched, the rest of the conversation stays on the fallback. The response states which provider and model produced the answer."),
    P("Before a tool runs, its arguments are validated against the tool's JSON schema (required fields, unknown fields, types). For the LLM, every failure becomes a JSON error message the model can react to; for direct API callers, the same failures become HTTP status codes. The system prompt instructs the model to call analyze_ecg instead of guessing, to cite knowledge-base sources, to mention model confidence and limitations, to ignore results flagged by the signal-quality check, and to state that the output is not a diagnosis."),
  );

  // ---------------------------------------------------------------- 6
  add(H1("CHAPTER 6: IMPLEMENTATION"), H2("6.1 Technology Stack"),
    ...TBL("Software used", ["Component", "Library (version)"], [
      ["Language", "Python 3.12"], ["Deep learning", "TensorFlow 2.21 / Keras 3.15 [26]"],
      ["Signal processing, data", "NumPy 2.5 [30], SciPy 1.18 [29], WFDB 4.3 [27], pandas 3.0"],
      ["Classical ML and metrics", "scikit-learn 1.9 [25]"], ["Web service", "FastAPI 0.139 [34], Uvicorn 0.51, Pydantic 2.13"],
      ["Vector database", "ChromaDB 1.5 (persistent client) [32]"], ["Embeddings", "all-MiniLM-L6-v2 via ONNX (Chroma default) [18]"],
      ["LLM (agentic core)", "OpenAI gpt-4o-mini via openai 2.45 [31]; fallback Gemini 2.5 Flash [36]"], ["Document parsing", "pypdf 6.14, langchain-text-splitters 1.1"],
      ["Testing", "pytest 9.2, FastAPI TestClient, psutil"], ["Notebooks and plots", "Jupyter, Matplotlib 3.11"],
      ["Hardware", "Intel Core i5-7200U (2 cores, 4 threads), 16 GB RAM, no GPU"],
    ], [2600, 5706]),
    H2("6.2 Project Structure"),
    ...TBL("Main modules and files", ["Path", "Responsibility"], [
      ["app/ecg/preprocessing.py", "Filtering, R-peak detection, segmentation, RR features, signal quality"],
      ["app/ecg/mitbih.py", "Download, loading and patient splits of MIT-BIH"],
      ["app/ecg/classifier.py", "Inference wrapper used by the API and the agent tool"],
      ["app/ecg/evaluation.py", "AAMI metrics, bootstrap CIs, end-to-end and probability metrics"],
      ["app/vector_store.py, embeddings.py, ingest.py", "Chroma store, embeddings, text extraction and chunking"],
      ["app/agent.py", "Agentic core: LLM loop, provider fallback, step trace, system prompt"],
      ["app/tools.py", "Tool layer: JSON schemas, argument validation, handlers, ECG registry"],
      ["app/routers/agent.py, tools.py", "Agent API and Tools API"],
      ["app/routers/ecg.py, documents.py, main.py, config.py", "ECG and document endpoints, start-up, provider settings"],
      ["notebooks/01–03", "Exploratory analysis, training and validation, LSTM as an LLM tool"],
      ["scripts/evaluate_model.py, benchmark_model.py", "Accuracy report and performance benchmark (reports/)"],
      ["scripts/test_report.py, make_figures.py", "Test report and report figures"],
      ["scripts/demo_agent.py", "Live demonstration of the tools API and the agent against a running server"],
      ["tests/", "119 automated tests (Chapter 7)"],
    ], [3500, 4806]),
    H2("6.3 Configuration and Notebooks"),
    P("All settings (Chroma path and collection, LLM provider, model, fallback and keys, embedding provider, chunking, model paths) are read from a .env file; a template (.env.example) is versioned while the real .env is excluded from version control. Notebook 01 explores the data, notebook 02 trains, validates and tests the model and writes the model, its metadata and the training log, and notebook 03 demonstrates the model as an LLM tool, with and without RAG, both directly and through the REST API."),
  );

  // ---------------------------------------------------------------- 7
  const junit = parseJunit(path.join(ROOT, "reports", "junit.xml"));
  const suiteRows = [
    ["test_preprocessing.py", "Unit", "Signal processing, R-peak detection, RR features, signal quality"],
    ["test_ingest.py", "Unit", "Text extraction (txt, md, pdf) and chunking"],
    ["test_vector_store.py", "Component", "Real Chroma store on disk"],
    ["test_agent.py", "Unit", "LLM tool-calling loop with a scripted LLM"],
    ["test_api.py", "API", "Every endpoint with fakes; validation and error codes"],
    ["test_model.py", "Model", "Correctness, robustness, accuracy floors on DS2"],
    ["test_model_performance.py", "Performance", "Speed, memory, calibration, AUC, error rates, significance"],
    ["test_api_integration.py", "Integration", "Real app: trained model, Chroma, local embeddings"],
    ["test_llm_live.py", "Live", "Real OpenAI agent (Gemini fallback) with LSTM and RAG tools"],
  ].map(([f, lvl, scope]) => { const s = junit.byFile[f.replace(".py", "")] || []; return [f, lvl, scope, String(s.length), String(s.filter((x) => x.status === "passed").length)]; });
  add(H1("CHAPTER 7: TESTING AND VALIDATION"), H2("7.1 Strategy"),
    P("Testing follows a layered approach. Fast unit and API tests use fakes for the LLM, the vector store and the classifier, so they run in seconds without a network or a trained model. Model, performance and integration tests run against the trained model and real MIT-BIH data, using only the 22 held-out DS2 patients. Thresholds are set a few points below the measured values, so the tests fail on real regressions rather than on normal variation between machines. In the offline suites the LLM is scripted, so the agent loop can be tested deterministically; the scripted LLM issues real tool calls that run against the real classifier and vector store. A separate live suite, enabled with RUN_LLM_TESTS=1, runs the complete system against the real OpenAI and Gemini APIs. Its assertions do not depend on wording: they check which tools were called with which arguments, that the cited sources exist, and that the numbers in the answer match the classifier's output."),
    ...TBL(`Test suites and results (${junit.passed} of ${junit.total} passed)`, ["Suite", "Level", "Scope", "Tests", "Passed"], suiteRows, [2500, 1100, 3306, 700, 700], { size: 18, align: [L, L, L, R, R] }),
    H2("7.2 Model Test Cases"),
    ...TBL("Selected model accuracy and robustness test cases", ["ID", "Test case", "Expected result", "Status"], [
      ["TC-M01", "Train, validation and test record lists are disjoint; test set equals DS2", "No patient in two sets", "Pass"],
      ["TC-M02", "API inference path vs training pipeline on record 233", "≤ 2% of beat labels differ", "Pass"],
      ["TC-M03", "Predictions repeated on the same beats", "Identical; probabilities sum to 1", "Pass"],
      ["TC-M04", "Signal scaled ×3 and offset +0.5", "Beat counts unchanged (±1)", "Pass"],
      ["TC-M05", "Record resampled to 250 Hz", "Beats, V count and heart rate within 2%, 10%, 2 bpm", "Pass"],
      ["TC-M06", "White noise added at 25 dB SNR", "V count within 10%, heart rate within 2 bpm", "Pass"],
      ["TC-M07", "Record 100 (sinus rhythm)", "Heart rate 65–85 bpm, ≥ 90% N", "Pass"],
      ["TC-M08", "Record 233 (frequent PVCs)", "≥ 70% of annotated V beats reported", "Pass"],
      ["TC-M09", "Accuracy floors on DS2", "Acc ≥ 0.92, macro-F1 ≥ 0.60, Se N ≥ 0.97, Se V ≥ 0.75, PPV V ≥ 0.78", "Pass"],
      ["TC-M10", "Comparison with baselines", "Macro-F1 above majority and RR-only baselines", "Pass"],
      ["TC-M11", "End-to-end detection on DS2", "Detection Se ≥ 0.97, PPV ≥ 0.99", "Pass"],
      ["TC-M12", "All 22 DS2 records through signal-quality check", "None flagged", "Pass"],
      ["TC-M13", "Flat and 3-second inputs", "No crash; valid response", "Pass"],
    ], [900, 3300, 3206, 900], { size: 18 }),
    H2("7.3 Model Performance Test Cases"),
    ...TBL("Model performance test cases (measured value in brackets)", ["ID", "Test case", "Expected result", "Status"], [
      ["TC-P01", "Model size", `< 100k parameters, < 2 MB (49,571; 0.71 MB)`, "Pass"],
      ["TC-P02", "Batched inference speed (2,048 beats)", `< 5 ms per beat (${SP.predict_beats["2048"].ms_per_beat} ms)`, "Pass"],
      ["TC-P03", "Full 30-minute record through analyze()", `< 30 s (${SP.analyze_full_record.wall_s} s)`, "Pass"],
      ["TC-P04", "Batch of 40 vs 40 single predictions", "Equal within 1e-4", "Pass"],
      ["TC-P05", "Ten repeated analyses of a 5-minute strip", "Memory growth < 200 MB (no leak)", "Pass"],
      ["TC-P06", "Calibration on DS2", `ECE < 0.05, Brier < 0.15 (${PQ.ece}; ${PQ.brier})`, "Pass"],
      ["TC-P07", "ROC AUC per class", `V ≥ 0.95, N ≥ 0.88, S ≥ 0.70 (${PQ.roc_auc.V}; ${PQ.roc_auc.N}; ${PQ.roc_auc.S})`, "Pass"],
      ["TC-P08", "Selective accuracy at rising confidence", "Monotonic; ≥ 0.95 accuracy at ≥ 85% coverage", "Pass"],
      ["TC-P09", "Normal beats predicted as V", `< 1% (${pct(ER.normal_called_V_rate, 2)})`, "Pass"],
      ["TC-P10", "V specificity", `≥ 0.98 (${f3(T.per_class.V.specificity)})`, "Pass"],
      ["TC-P11", "S sensitivity without record 232", `≥ 0.30 (${f3(ER.without_record_232.S.sensitivity)})`, "Pass"],
      ["TC-P12", "V sensitivity per patient", "≥ 0.7 in ≥ 75% of patients; none < 0.3", "Pass"],
      ["TC-P13", "Bootstrap CI of macro-F1 vs majority baseline", `Lower bound > ${f3(BL["majority (always N)"].macro_f1)} (${f3(CI.macro_f1[0])})`, "Pass"],
      ["TC-P14", "Paired patient bootstrap vs RR-only baseline", "Model better in ≥ 90% of resamples", "Pass"],
    ], [900, 3000, 3506, 900], { size: 18 }),
    H2("7.4 API and Integration Test Cases"),
    ...TBL("Selected API and integration test cases", ["ID", "Test case", "Expected result", "Status"], [
      ["TC-A01", "GET /health", "200, collection name, model loaded flag", "Pass"],
      ["TC-A02", "Upload, search and delete a document", "Chunks stored, relevant hit first, count back to 0", "Pass"],
      ["TC-A03", "Re-ingest a document with the same name", "Old chunks replaced, not duplicated", "Pass"],
      ["TC-A04", "Unsupported file type or empty text", "400", "Pass"],
      ["TC-A05", "Signal shorter than 256 samples, fs = 0, top_k = 0", "422", "Pass"],
      ["TC-A06", "Analyse an unknown ecg_id", "404", "Pass"],
      ["TC-A07", "Analyse while the model is not loaded", "503", "Pass"],
      ["TC-A08", "Chat with the LLM provider unreachable", "502 with a configuration hint", "Pass"],
      ["TC-A09", "Chat history containing a system role", "422 (prompt-injection guard)", "Pass"],
      ["TC-A10", "GET /tools", "Same names and schemas the LLM receives", "Pass"],
      ["TC-A11", "POST /tools/analyze_ecg/invoke", "LSTM result returned with timing", "Pass"],
      ["TC-A12", "Invoke unknown tool / bad arguments / unknown ecg_id / no model", "404 / 422 / 404 / 503", "Pass"],
      ["TC-A13", "POST /agent/chat with tool calls", "Trace: LLM → tool → LLM", "Pass"],
      ["TC-A14", "OpenAI fails, Gemini configured (scripted)", "Answer from fallback; flagged in response", "Pass"],
      ["TC-I01", "Start-up with real lifespan", "Model loaded, Chroma collection created", "Pass"],
      ["TC-I02", "Semantic search with real embeddings", "ECG question → ECG document; baking question → cooking text", "Pass"],
      ["TC-I03", "Restart the application", "Knowledge base persists on disk", "Pass"],
      ["TC-I04", "Record 233 uploaded as 250 Hz CSV", "Beat count within 3%, V count within 15% of cardiologist labels", "Pass"],
      ["TC-I05", "Upload with supplied R-peaks (record 100)", "All peaks classified except edges; ≥ 95% N", "Pass"],
      ["TC-I06", "Analyse response contract", "All fields present; counts and percentages consistent", "Pass"],
      ["TC-I07", "Upload and analyse a 5-minute strip over HTTP", "< 5 s (measured ≈ 0.45 s)", "Pass"],
      ["TC-I08", "Pure noise uploaded as an ECG", "signal_quality.ok = false with a warning", "Pass"],
      ["TC-I09", "Agent calls analyze_ecg and search on real services", "Tool results contain the real model output and cited chunks", "Pass"],
    ], [900, 3300, 3206, 900], { size: 18 }),
    H2("7.5 Live LLM Test Cases"),
    P("These tests run the complete system with the real OpenAI gpt-4o-mini agent (temperature 0) and real Gemini 2.5 Flash calls, on held-out patients."),
    ...TBL("Live end-to-end test cases with real LLMs", ["ID", "Test case", "Expected result", "Status"], [
      ["TC-L01", "GET /health", "Main openai:gpt-4o-mini, fallback gemini:gemini-2.5-flash", "Pass"],
      ["TC-L02", "Question about record 233", "Agent calls analyze_ecg with the right id; answer quotes the model's V count or share; states not a diagnosis", "Pass"],
      ["TC-L03", "Noise uploaded as ECG", "Agent calls the tool and reports noise / poor quality", "Pass"],
      ["TC-L04", "Which recordings are uploaded?", "list_ecg_recordings called; both names in the answer", "Pass"],
      ["TC-L05", "Multi-turn follow-up", "Exact V count given in the second turn", "Pass"],
      ["TC-L06", "Question answered from the knowledge base", "search tool called; sources cited from the reference document", "Pass"],
      ["TC-L07", "ECG and guideline question together", "Both tools called; sources cited", "Pass"],
      ["TC-L08", "Tools API vs agent on the same ECG", "Answer contains the N and V counts from the direct tool call", "Pass"],
      ["TC-L09", "Step trace", "LLM decides to call the tool, tool succeeds, LLM answers; token counts present", "Pass"],
      ["TC-L10", "Gemini as main agent", "Gemini calls analyze_ecg and answers", "Pass"],
      ["TC-L11", "OpenAI key rejected", "Gemini fallback answers; fallback_used = true", "Pass"],
    ], [900, 2700, 3806, 900], { size: 18 }),
    H2("7.6 Mutation Testing"),
    P("To check that the tests detect real faults, bugs were injected deliberately into the code [33] and the suite was re-run; every mutation was detected (Table below)."),
    ...TBL("Injected faults and the tests that detected them", ["Injected fault", "Tests failed"], [
      ["RR features passed in the wrong order", "Unit tests on RR features"],
      ["Similarity score computed as distance instead of 1 − distance", "Vector-store ranking tests"],
      ["Tool result not returned to the LLM", "Agent loop tests"],
      ["API path ignores RR features", "2 model tests"],
      ["Signal-quality check disabled", "4 tests (3 unit, 1 integration)"],
      ["Rhythm features replaced by zeros at inference", "7 of 14 performance tests (calibration, AUC, specificity, S detection, significance)"],
    ], [4300, 4006]),
    H2("7.7 Defects Found by Testing"),
    B("**Patient leakage in validation (v1)**: a random beat split put training patients in the validation set. Fixed with patient-wise validation."),
    B("**Over-prediction of fusion beats (v2)**: found by the held-out evaluation; fusion was removed from the trained classes."),
    B("**Unhandled LLM errors**: an invalid key produced an HTTP 500; it now returns 502 with a hint."),
    B("**Beats reported in pure noise**: an integration test showed that 20 s of Gaussian noise produced 39 \"beats\". A kurtosis-based signal-quality index was added; it flags all noise inputs and none of the 22 test recordings."),
    B("**Edge-beat misalignment**: a consistency test exposed that segmentation drops edge beats; the training loader now records the R-peak of each kept beat."),
  );

  // ---------------------------------------------------------------- 8
  const pc = T.per_class, cpc = CI.per_class;
  add(H1("CHAPTER 8: RESULTS AND DISCUSSION"), H2("8.1 Overall Performance on Unseen Patients"),
    ...TBL("Performance on the 22 DS2 test patients at annotated R-peaks (95% patient-bootstrap CI)", ["Model", "Accuracy", "Macro-F1", "Balanced accuracy"], [
      ["Majority class (always N)", f3(BL["majority (always N)"].accuracy), f3(BL["majority (always N)"].macro_f1), f3(BL["majority (always N)"].balanced_accuracy)],
      ["RR-only logistic regression", f3(BL["RR-only logistic regression"].accuracy), f3(BL["RR-only logistic regression"].macro_f1), f3(BL["RR-only logistic regression"].balanced_accuracy)],
      ["**Conv-BiLSTM (this work)**", `**${f3(T.accuracy)}** ${ci(CI.accuracy)}`, `**${f3(T.macro_f1)}** ${ci(CI.macro_f1)}`, `**${f3(T.balanced_accuracy)}** ${ci(CI.balanced_accuracy)}`],
    ], [2700, 1900, 1900, 1806], { align: [L, R, R, R] }),
    ...TBL("Per-class results on DS2", ["Class", "Beats", "Sensitivity [95% CI]", "PPV [95% CI]", "Specificity", "F1"],
      ["N", "S", "V"].map((c) => [c, pc[c].support.toLocaleString(), `${f3(pc[c].sensitivity)} ${ci(cpc[c].sensitivity)}`, `${f3(pc[c].ppv)} ${ci(cpc[c].ppv)}`, f3(pc[c].specificity), f3(pc[c].f1)]),
      [800, 1000, 2200, 2200, 1100, 1006], { align: [L, R, R, R, R, R] }),
    P(`The model beats both baselines on macro-F1, and the lower bound of its confidence interval (${f3(CI.macro_f1[0])}) is well above the majority baseline (${figRef()}). Normal beats are recognised almost perfectly (Se ${f3(pc.N.sensitivity)}), and ventricular beats are detected with sensitivity ${f3(pc.V.sensitivity)} and precision ${f3(pc.V.ppv)}. Supraventricular beats are the main weakness, discussed in Section 8.5.`),
    ...FIG("baselines.png", "Held-out macro-F1 of the model and the baselines", 4.0),
    ...FIG("confusion_matrix.png", "Confusion matrix on the DS2 test patients", 3.4),
    H2("8.2 End-to-End Performance"),
    P(`With the deployed R-peak detector instead of the annotations, ${DET.matched.toLocaleString()} of ${DET.reference_beats.toLocaleString()} annotated beats are found (detection sensitivity ${f3(DET.sensitivity)}, PPV ${f3(DET.ppv)}, ±${DET.tolerance_ms} ms). Classification of the detected beats reaches accuracy ${f3(E2E.accuracy)} and macro-F1 ${f3(E2E.macro_f1)}, with V sensitivity ${f3(E2E.per_class.V.sensitivity)} and PPV ${f3(E2E.per_class.V.ppv)}. Performance therefore barely changes when the model runs on its own detected beats, which is the situation of a real user.`),
    H2("8.3 Calibration"),
    P(`The predicted probabilities are well calibrated: ECE is ${PQ.ece}, the Brier score ${PQ.brier}, and the reliability curve lies close to the diagonal (${figRef()}). The ROC AUC is ${PQ.roc_auc.N} for N, ${PQ.roc_auc.S} for S and ${PQ.roc_auc.V} for V. Confidence is useful in practice: on the ${pct(PQ.selective["0.9"].coverage, 1)} of beats with confidence ≥ 0.9 the accuracy is ${pct(PQ.selective["0.9"].accuracy, 1)}, so low-confidence beats can be highlighted for human review. The agent receives the confidence of every abnormal beat.`),
    ...FIG("reliability.png", "Reliability diagram of the predicted-class confidence on DS2", 3.4),
    H2("8.4 Computational Performance"),
    ...TBL("Speed and resources on an Intel i5-7200U laptop CPU without GPU", ["Measurement", "Result"], [
      ["Inference, batch of 2,048 beats", `${SP.predict_beats["2048"].ms_per_beat} ms per beat (${SP.predict_beats["2048"].beats_per_s.toLocaleString()} beats/s)`],
      ["Inference, single beat", `${SP.predict_beats["1"].latency_ms} ms (dominated by framework overhead)`],
      ["Full analysis of a 30-minute record", `${SP.analyze_full_record.wall_s} s, about ${SP.analyze_full_record.faster_than_real_time.toLocaleString()}× faster than real time`],
      ["API upload + analysis, 1-minute strip", `${perf.api["1_min_strip"].median_ms} ms (median of 5)`],
      ["API upload + analysis, 5-minute strip", `${perf.api["5_min_strip"].median_ms} ms (median of 5)`],
      ["Model size", "49,571 parameters, 0.71 MB"],
      ["Training time", `${meta.epochs_run} epochs, about 9 minutes`],
    ], [3600, 4706]),
    H2("8.5 Error Analysis"),
    P(`S beats are mostly missed (Se ${f3(pc.S.sensitivity)}). ${ER.S_beats_in_232.toLocaleString()} of the ${ER.S_beats_total.toLocaleString()} S beats in DS2 come from record 232, a patient with sinus bradycardia and long pauses, in which atrial premature beats are not early relative to the patient's own rhythm, the main cue the model relies on. Without this record, accuracy is ${f3(ER.without_record_232.accuracy)}, macro-F1 ${f3(ER.without_record_232.macro_f1)}, and S sensitivity ${f3(ER.without_record_232.S.sensitivity)}. The patient-level confidence interval for S sensitivity (${ci(cpc.S.sensitivity)}) reflects this dependence on individual patients.`),
    P(`V sensitivity also varies between patients: per record it ranges from ${Math.min(...Object.values(ER.V_sensitivity_per_record)).toFixed(2)} to ${Math.max(...Object.values(ER.V_sensitivity_per_record)).toFixed(2)}. Only ${pct(ER.normal_called_V_rate, 2)} of normal beats are wrongly reported as V, so false ventricular alarms are rare. Of the ${ANN.excluded_classes_predicted_as.F ? Object.values(ANN.excluded_classes_predicted_as.F).reduce((a, b) => a + b, 0) : 0} fusion beats in DS2, ${ANN.excluded_classes_predicted_as.F ? ANN.excluded_classes_predicted_as.F.N : 0} are labelled N, as expected for a class that was not trained.`),
    P(`${figRef()} shows ten seconds of record 233, a held-out patient. The model flags every ventricular beat with confidence above 0.95 and leaves the normal beats unflagged; the single S beat at 20.8 s is missed, illustrating the weakness described above.`),
    ...FIG("ecg_strip_233.png", "Model output on record 233 compared with cardiologist labels", 6.2),
    H2("8.6 Agent Behaviour with a Real LLM"),
    P("With the real OpenAI model, a question such as “Is anything abnormal in this ECG? Give the numbers.” about 30 seconds of record 233 produced the following trace: the LLM decided to call analyze_ecg (about 2.0 s, 361 prompt and 24 completion tokens); the LSTM tool ran in about 0.1 s; the LLM then wrote the answer (about 7.2 s, 1,009 prompt and 531 completion tokens). The answer reported 36 normal and 13 ventricular beats, identical to a direct call of the tool through the Tools API, listed the abnormal beats with their times and confidences, and ended with the disclaimer. Most of the response time is spent in the LLM, not in the classifier."),
    P("Across the 11 live tests the agent always called the ECG tool instead of guessing, quoted the classifier's numbers, cited the knowledge base when asked factual questions, and recognised noise input from the signal-quality block. When the OpenAI key was invalid, Gemini answered the same question with the same tool calls."),
    H2("8.7 Discussion"),
    P("The results show that a small model trained on a laptop can reach useful performance on unseen patients for normal and ventricular beats, the most frequent and clinically most relevant ectopic class. They also confirm what the literature reports for the inter-patient setting [3], [13]: supraventricular beats are much harder to recognise on new patients than in intra-patient experiments. Reporting patient-level confidence intervals, baselines and a per-record breakdown makes this visible, where a single accuracy figure (94.5%) would hide it."),
    P("Integrating the model as an LLM tool keeps the measurement and the explanation separate. The classifier produces the numbers, the knowledge base supplies the reference text, and the LLM only combines them, with instructions to state the model's limitations, to mention its confidence and to disregard results flagged by the signal-quality check."),
  );

  // ---------------------------------------------------------------- 9
  add(H1("CHAPTER 9: LIMITATIONS AND ETHICAL CONSIDERATIONS"), H2("9.1 Limitations"),
    B("Low sensitivity for supraventricular beats; the model must not be used to rule out atrial ectopy."),
    B("Fusion and paced beats are not recognised; recordings from patients with pacemakers are out of scope."),
    B("Trained on a single lead (MLII) of a single database of 47 subjects recorded in 1975–1979; other leads, wearable devices and populations are untested."),
    B("RR features use the recording's own rhythm as reference, so strips shorter than about 10 beats give less reliable results."),
    B("Uploaded ECGs are kept in memory and are lost on restart; the service has no authentication."),
    B("The quality of the agent's answers depends on the external LLM. It was exercised in 11 live tests with OpenAI and Gemini, but its explanations have not been evaluated by clinicians, and LLM providers may change model behaviour over time."),
    H2("9.2 Ethical Considerations"),
    B("**Not a medical device.** Every analysis includes a disclaimer, and the agent is instructed to state that results are not a diagnosis."),
    B("**Data privacy.** The MIT-BIH recordings are de-identified and openly licensed. ECGs uploaded by users are kept in memory only. The question, the conversation history and the tool results (heart rate, beat counts, abnormal beat times, retrieved document chunks) are sent to the LLM provider (OpenAI, or Google for the fallback); the raw ECG signal is not. A local model can be configured (LLM_PROVIDER=custom) to keep all data on-premises."),
    B("**Transparency.** Limitations, failure cases and rejected model versions are documented in the model card and in this report."),
    B("**Confidentiality.** The project uses only public data and open-source software and contains no confidential or customer information."),
  );

  // ---------------------------------------------------------------- 10
  add(H1("CHAPTER 10: CONCLUSION AND FUTURE WORK"), H2("10.1 Conclusion"),
    P(`This project delivered a complete, tested system for ECG beat classification and its use by an LLM agent. A Conv-BiLSTM of 49,571 parameters, trained on a laptop CPU, reaches ${pct(T.accuracy)} accuracy and a macro-F1 of ${f3(T.macro_f1)} on 22 unseen patients, beating a rhythm-only baseline in a paired patient-level bootstrap, with well-calibrated probabilities and an end-to-end pipeline that analyses a 30-minute recording in about a second. The model is served by a FastAPI application with a persistent Chroma knowledge base. An OpenAI model is the agentic core that calls the classifier and the knowledge base as tools, with Gemini as automatic fallback, and the same tools are available through a separate REST API. ${junit.total} automated tests, checked by mutation testing, guard preprocessing, APIs, integration, accuracy, performance and the live agent.`),
    H2("10.2 Contributions"),
    N("A shared preprocessing pipeline with patient-relative rhythm features and a kurtosis-based signal-quality check.", "contrib"),
    N("A strict inter-patient evaluation with baselines, patient-bootstrap confidence intervals, end-to-end detection, calibration and error analysis.", "contrib"),
    N("A deployable service in which an OpenAI agent, with a Gemini fallback, uses the model and a RAG knowledge base as tools, and a separate tools API that exposes the same functions without an LLM.", "contrib"),
    N("A test suite that includes accuracy floors, performance tests and integration tests on held-out patients.", "contrib"),
    H2("10.3 Future Work"),
    N("Improve S-beat detection with longer rhythm context (sequence-of-beats models) and patient-adaptive fine-tuning on a few labelled beats.", "future"),
    N("Train and validate on additional databases (e.g. INCART, the European ST-T database) and on wearable single-lead devices.", "future"),
    N("Add fusion and paced classes with more varied training data.", "future"),
    N("Add persistent storage, authentication and audit logging for the uploaded recordings.", "future"),
    N("Evaluate the agent's answers with clinicians and with automatic groundedness metrics.", "future"),
  );
  return { out, junit };
}

const REFS = [
  "G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,” IEEE Engineering in Medicine and Biology Magazine, vol. 20, no. 3, pp. 45–50, 2001.",
  "A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng and H. E. Stanley, “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,” Circulation, vol. 101, no. 23, pp. e215–e220, 2000.",
  "P. de Chazal, M. O’Dwyer and R. B. Reilly, “Automatic classification of heartbeats using ECG morphology and heartbeat interval features,” IEEE Transactions on Biomedical Engineering, vol. 51, no. 7, pp. 1196–1206, 2004.",
  "ANSI/AAMI EC57:2012, Testing and Reporting Performance Results of Cardiac Rhythm and ST Segment Measurement Algorithms. Arlington, VA: Association for the Advancement of Medical Instrumentation, 2012.",
  "J. Pan and W. J. Tompkins, “A real-time QRS detection algorithm,” IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3, pp. 230–236, 1985.",
  "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
  "M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.",
  "S. Kiranyaz, T. Ince and M. Gabbouj, “Real-time patient-specific ECG classification by 1-D convolutional neural networks,” IEEE Transactions on Biomedical Engineering, vol. 63, no. 3, pp. 664–675, 2016.",
  "U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, M. Adam, A. Gertych and R. S. Tan, “A deep convolutional neural network model to classify heartbeats,” Computers in Biology and Medicine, vol. 89, pp. 389–396, 2017.",
  "A. Y. Hannun, P. Rajpurkar, M. Haghpanahi, G. H. Tison, C. Bourn, M. P. Turakhia and A. Y. Ng, “Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network,” Nature Medicine, vol. 25, no. 1, pp. 65–69, 2019.",
  "Ö. Yildirim, “A novel wavelet sequence based on deep bidirectional LSTM network model for ECG signal classification,” Computers in Biology and Medicine, vol. 96, pp. 189–202, 2018.",
  "V. Mondéjar-Guerra, J. Novo, J. Rouco, M. G. Penedo and M. Ortega, “Heartbeat classification fusing temporal and morphological information of ECGs via ensemble of classifiers,” Biomedical Signal Processing and Control, vol. 47, pp. 41–48, 2019.",
  "G. Llamedo and J. P. Martínez, “Heartbeat classification using feature selection driven by database generalization criteria,” IEEE Transactions on Biomedical Engineering, vol. 58, no. 3, pp. 616–625, 2011.",
  "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel and D. Kiela, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Advances in Neural Information Processing Systems 33 (NeurIPS), 2020, pp. 9459–9474.",
  "S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations (ICLR), 2023.",
  "T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” in Advances in Neural Information Processing Systems 36 (NeurIPS), 2023.",
  "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proceedings of EMNLP-IJCNLP, 2019, pp. 3982–3992.",
  "W. Wang, F. Wei, L. Dong, H. Bao, N. Yang and M. Zhou, “MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,” in Advances in Neural Information Processing Systems 33 (NeurIPS), 2020.",
  "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations (ICLR), 2015.",
  "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall/CRC, 1993.",
  "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning (ICML), 2017, pp. 1321–1330.",
  "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015, pp. 448–456.",
  "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” Journal of Machine Learning Research, vol. 15, pp. 1929–1958, 2014.",
  "G. W. Brier, “Verification of forecasts expressed in terms of probability,” Monthly Weather Review, vol. 78, no. 1, pp. 1–3, 1950.",
  "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
  "M. Abadi et al., “TensorFlow: A system for large-scale machine learning,” in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2016, pp. 265–283.",
  "C. Xie, L. McCullum, A. Johnson, T. Pollard, B. Gow and B. Moody, “Waveform Database Software Package (WFDB) for Python,” PhysioNet. [Online]. Available: https://github.com/MIT-LCP/wfdb-python",
  "Q. Li, R. G. Mark and G. D. Clifford, “Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter,” Physiological Measurement, vol. 29, no. 1, pp. 15–32, 2008.",
  "P. Virtanen et al., “SciPy 1.0: Fundamental algorithms for scientific computing in Python,” Nature Methods, vol. 17, pp. 261–272, 2020.",
  "C. R. Harris et al., “Array programming with NumPy,” Nature, vol. 585, pp. 357–362, 2020.",
  "OpenAI, “Chat Completions API and function calling,” API reference. [Online]. Available: https://platform.openai.com/docs/api-reference",
  "Chroma, “Chroma: the open-source AI application database,” documentation. [Online]. Available: https://docs.trychroma.com",
  "Y. Jia and M. Harman, “An analysis and survey of the development of mutation testing,” IEEE Transactions on Software Engineering, vol. 37, no. 5, pp. 649–678, 2011.",
  "S. Ramírez, “FastAPI,” documentation. [Online]. Available: https://fastapi.tiangolo.com",
  "G. B. Moody and R. G. Mark, “MIT-BIH Arrhythmia Database” (version 1.0.0), PhysioNet, 2005. [Online]. Available: https://physionet.org/content/mitdb/1.0.0/",
  "Google, “Gemini API: OpenAI compatibility,” documentation. [Online]. Available: https://ai.google.dev/gemini-api/docs/openai",
];

function parseJunit(file) {
  const xml = fs.readFileSync(file, "utf8");
  const cases = [];
  const re = /<testcase classname="([^"]+)" name="([^"]+)" time="([^"]+)"(\s*\/>|>([\s\S]*?)<\/testcase>)/g;
  let m;
  while ((m = re.exec(xml))) {
    const inner = m[5] || "";
    const status = /<failure|<error/.test(inner) ? "failed" : /<skipped/.test(inner) ? "skipped" : "passed";
    cases.push({ file: m[1].split(".").pop(), name: m[2].replace(/&quot;/g, '"').replace(/&amp;/g, "&"), time: +m[3], status });
  }
  const byFile = {};
  cases.forEach((c) => (byFile[c.file] = byFile[c.file] || []).push(c));
  return { cases, byFile, total: cases.length, passed: cases.filter((c) => c.status === "passed").length };
}

function report() {
  const { out: chapters, junit } = body();
  const refs = [H1("REFERENCES"), ...REFS.map((r, i) => new Paragraph({ spacing: { after: 100, line: 300 }, indent: { left: 600, hanging: 600 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: `[${i + 1}]\t${r}`, size: 22 })], tabStops: [{ type: TabStopType.LEFT, position: 600 }] }))];
  const appA = [
    H1("APPENDIX A: HOW TO REPRODUCE"),
    P("All commands are run from the project root on Linux with Python 3.12."),
    ...[
      "python3 -m venv env && source env/bin/activate && pip install -r requirements.txt",
      "cp .env.example .env    # set OPENAI_API_KEY (and GEMINI_API_KEY for the fallback)",
      "jupyter nbconvert --to notebook --execute notebooks/01_ecg_analysis.ipynb",
      "jupyter nbconvert --to notebook --execute notebooks/02_lstm_training.ipynb",
      "python scripts/evaluate_model.py      # reports/evaluation.md",
      "python scripts/benchmark_model.py     # reports/performance.md",
      "python scripts/test_report.py         # runs all tests, reports/test_report.md",
      "python scripts/make_figures.py        # docs/figures/",
      "RUN_LLM_TESTS=1 pytest -m llm         # live tests with the real OpenAI and Gemini APIs",
      "uvicorn app.main:app                  # API docs at http://localhost:8000/docs",
      "python scripts/demo_agent.py          # tools API and agent trace against the running server",
    ].map((c) => new Paragraph({ spacing: { after: 40 }, indent: { left: 360 }, children: [new TextRun({ text: c, font: "Courier New", size: 18 })] })),
    P("Model artefacts (models/), MIT-BIH data (data/) and the vector store (chroma_db/) are generated by these steps and are not stored in version control. The full API reference with request and response examples is in docs/API.md of the project repository."),
  ];
  const appB = [
    H1("APPENDIX B: COMPLETE TEST CASE CATALOGUE"),
    P(`Generated from reports/junit.xml: ${junit.passed} passed, ${junit.total - junit.passed} failed or skipped, of ${junit.total} tests.`),
    ...TBL("All automated test cases", ["#", "Suite", "Test case", "Time (s)", "Result"],
      junit.cases.map((c, i) => [String(i + 1), c.file.replace("test_", ""), c.name.replace(/^test_/, "").replace(/_/g, " "), c.time.toFixed(2), c.status === "passed" ? "Pass" : c.status]),
      [500, 1700, 4506, 800, 800], { size: 16, align: [R, L, L, R, L] }),
  ];
  // front matter lists need figures/tables collected first, so build body before contents
  const doc = new Document({
    creator: S.name, title: S.title, description: `Project report, ${S.programme}, ${S.university}`,
    styles, numbering,
    sections: [{
      properties: { page: { size: A4, margin: { top: 1440, bottom: 1440, left: 2160, right: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
      children: [...front(), ...contents(), ...chapters, ...refs, ...appA, ...appB],
    }],
  });
  return doc;
}

(async () => {
  const syn = path.join(OUT, `${S.reg}.docx`);
  fs.writeFileSync(syn, await Packer.toBuffer(synopsis()));
  const rep = path.join(OUT, `Project_Report_${S.reg}.docx`);
  fs.writeFileSync(rep, await Packer.toBuffer(report()));
  console.log(`abstract words: ${words}\nwrote ${syn}\nwrote ${rep}\nfigures ${figNo}, tables ${tabNo}`);
})();
