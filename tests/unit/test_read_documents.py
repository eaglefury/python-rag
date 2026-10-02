from pathlib import Path

import pytest

from python_rag.ingest.read_documents import read_document
from python_rag.types.types import FileData


def test_reads_txt(data_file):
    result = read_document(data_file("notes.txt"))

    assert result == FileData(content="Hello from a text file.\nSecond line.", name="notes.txt")


def test_reads_docx_paragraphs_on_separate_lines(data_file):
    result = read_document(data_file("report.docx"))

    assert result.name == "report.docx"
    assert result.content.splitlines() == ["First paragraph.", "Second paragraph."]


def test_reads_pdf(data_file):
    result = read_document(data_file("manual.pdf"))

    assert result.name == "manual.pdf"
    assert "The region number of this player is 2." in result.content


def test_pdf_pages_are_separated_by_a_newline(data_file):
    content = read_document(data_file("two-pages.pdf")).content

    # Without a separator the last word of one page is glued to the next page.
    assert "oneStart" not in content
    assert content.splitlines() == ["End of page one", "Start of page two"]


def test_reads_xref_stream_pdf(data_file):
    assert "Modern PDF." in read_document(data_file("xref-stream.pdf")).content


@pytest.mark.parametrize("name", ["junk-prefix-xref-table.pdf", "junk-prefix-xref-stream.pdf"])
def test_reads_pdf_with_junk_before_pdf_header(data_file, name):
    # Regression: the CD player manual (xref-stream PDF) had HTML injected before
    # %PDF by the download page, which made PyPDF2 crash with KeyError: '/Root'.
    assert "Still readable." in read_document(data_file(name)).content


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        read_document(str(tmp_path / "nope.pdf"))


def test_unsupported_extension_raises(data_file):
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_document(data_file("data.csv"))


def test_non_string_path_raises(data_file):
    with pytest.raises(ValueError, match="Invalid file path"):
        read_document(Path(data_file("notes.txt")))
