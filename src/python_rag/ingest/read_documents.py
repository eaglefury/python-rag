from typing import Dict

# pypdf is the maintained successor to PyPDF2; it also tolerates PDFs with junk
# before the %PDF header (e.g. HTML injected by download pages), which PyPDF2 cannot open.
import pypdf
import os
from docx import Document
from pathlib import Path

from python_rag.types.types import FileData


def read_txt(file_path: str) -> FileData:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return FileData(content=text, name=Path(file_path).name)


def read_pdf(file_path: str) -> FileData:
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        # Join pages with a newline so the last word of one page isn't glued to
        # the first word of the next.
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return FileData(content=text, name=Path(file_path).name)


def read_docx(file_path: str) -> FileData:
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return FileData(content=text, name=Path(file_path).name)


def read_document(file_path: str) -> FileData:

    #  check that the valid file path is provided
    if not isinstance(file_path, str):
        raise ValueError(f"Invalid file path: {file_path}")
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    if not file_path.endswith((".txt", ".pdf", ".docx")):
        raise ValueError(f"Unsupported file type: {file_path}")

    if file_path.endswith(".txt"):
        return read_txt(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".docx"):
        return read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
