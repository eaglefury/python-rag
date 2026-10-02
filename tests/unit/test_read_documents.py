from pathlib import Path

import pytest

from python_rag.ingest.read_documents import read_document
from python_rag.types.types import FileData


def test_reads_txt(make_txt):
    path = make_txt("Hello from a text file.\nSecond line.", name="notes.txt")

    result = read_document(path)

    assert result == FileData(content="Hello from a text file.\nSecond line.", name="notes.txt")


def test_reads_docx_paragraphs_on_separate_lines(make_docx):
    path = make_docx(["First paragraph.", "Second paragraph."], name="report.docx")

    result = read_document(path)

    assert result.name == "report.docx"
    assert result.content.splitlines() == ["First paragraph.", "Second paragraph."]


def test_reads_pdf(make_pdf):
    path = make_pdf(["The region number of this player is 2."], name="manual.pdf")

    result = read_document(path)

    assert result.name == "manual.pdf"
    assert "The region number of this player is 2." in result.content


def test_pdf_pages_are_separated_by_a_newline(make_pdf):
    path = make_pdf(["End of page one", "Start of page two"])

    content = read_document(path).content

    # Without a separator the last word of one page is glued to the next page.
    assert "oneStart" not in content
    assert content.splitlines() == ["End of page one", "Start of page two"]


@pytest.mark.parametrize("xref_stream", [False, True], ids=["xref-table", "xref-stream"])
def test_reads_pdf_with_junk_before_pdf_header(make_pdf, xref_stream):
    # Regression: the CD player manual (xref-stream PDF) had HTML injected before
    # %PDF by the download page, which made PyPDF2 crash with KeyError: '/Root'.
    path = make_pdf(["Still readable."], prefix=b"<!==  template/download.html ==>",
                    xref_stream=xref_stream)

    assert "Still readable." in read_document(path).content


def test_reads_xref_stream_pdf(make_pdf):
    assert "Modern PDF." in read_document(make_pdf(["Modern PDF."], xref_stream=True)).content


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        read_document(str(tmp_path / "nope.pdf"))


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="Unsupported file type"):
        read_document(str(path))


def test_non_string_path_raises(make_txt):
    with pytest.raises(ValueError, match="Invalid file path"):
        read_document(Path(make_txt("text")))
