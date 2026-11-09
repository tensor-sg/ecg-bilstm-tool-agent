# Academic documents

M.Sc. Data Science, 4th semester project, SASTRA Deemed to be University.
Student: Priyatosh Satpati (Reg. No. 25113170096, priyatoshsatpatiofficial@gmail.com) ·
Guide and reporting manager: Mr. G. Swaminathan (SAP ID C2385).

| File | What it is |
|---|---|
| `25113170096.pdf` / `.docx` | Project synopsis in the SASTRA template: Times New Roman, title 14 pt bold centred, details in capitals 14 pt, abstract 12 pt justified at 1.5 spacing, ≤ 300 words (currently 291), one page |
| `Project_Report_25113170096.pdf` / `.docx` | Full report (43 pages): certificate, declaration, abstract, 10 chapters, ERD and diagrams, test-case tables, 36 IEEE-style references, appendices |
| `source/build_documents.js` | Generator for both `.docx` files; numbers and tables are read from the committed results |

## Submission checklist (synopsis)

1. Check the synopsis with the reporting manager; edit the `.docx` in Word if he asks for changes, or edit
   `source/build_documents.js` and rebuild.
2. After approval, get his e-signature or signature on the document.
3. Save as PDF named exactly `25113170096.pdf`.
4. Upload it with the link sent by email, entering the reporting manager's full name (G. Swaminathan) and SAP ID (C2385).
5. Remind him to check his email.

The synopsis and report contain no employer or customer names and no confidential information.

## Rebuilding

The generator reads `models/ecg_lstm_meta.json`, `reports/evaluation.json`, `reports/performance.json`,
`reports/junit.xml` and `docs/figures/*.png`, so regenerate those first if results change:

```bash
python scripts/evaluate_model.py && python scripts/benchmark_model.py
RUN_LLM_TESTS=1 python scripts/test_report.py      # writes reports/junit.xml
python scripts/make_figures.py

cd docs/academic/source
npm install                                       # docx 9.5.1
node build_documents.js ../../..                  # writes ../25113170096.docx and ../Project_Report_25113170096.docx
cd .. && soffice --headless --convert-to pdf *.docx
```
