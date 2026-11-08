import pytest
from fpdf import FPDF

from app.ingest import chunk_text, extract_text


def test_extract_text_txt_and_md():
    assert extract_text("a.txt", b"hello") == "hello"
    assert extract_text("NOTES.MD", "# Title\nbody".encode()) == "# Title\nbody"


def test_extract_text_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Premature ventricular contraction")
    text = extract_text("doc.pdf", bytes(pdf.output()))
    assert "Premature ventricular contraction" in text


def test_extract_text_rejects_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("image.png", b"\x89PNG")


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 10
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share text because of the overlap
    assert chunks[0].split()[-1] in chunks[1]


def test_chunk_text_drops_blank_input():
    assert chunk_text("   \n\n  ", 100, 10) == []
