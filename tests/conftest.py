from pathlib import Path
import struct

import pytest
from docx import Document as DocxDocument


def build_pdf(pages: list[str], xref_stream: bool = False) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page.

    Hand-written so the tests don't need a PDF-writing dependency.
    `xref_stream=True` writes the cross-reference index as a stream object
    (PDF 1.5+, used by most modern PDFs including the CD player manual)
    instead of a plain `xref` table.
    """
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
           f"/Count {len(pages)} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for text, page_id in zip(pages, page_ids):
        content_id = page_id + 1
        stream = f"BT /F1 12 Tf 72 720 Td ({escape(text)}) Tj ET".encode()
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.5\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n"
    xref_offset = len(out)

    if xref_stream:
        # Uncompressed xref stream: per object 1 byte type, 4 bytes offset, 2 bytes generation.
        xref_id = max(objects) + 1
        size = xref_id + 1
        offsets[xref_id] = xref_offset
        rows = b"\x00" + struct.pack(">IH", 0, 65535) + b"".join(
            b"\x01" + struct.pack(">IH", offsets[i], 0) for i in range(1, size))
        out += (f"{xref_id} 0 obj\n<< /Type /XRef /Size {size} /W [1 4 2] /Root 1 0 R "
                f"/Length {len(rows)} >>\nstream\n").encode() + rows + b"\nendstream\nendobj\n"
        out += f"startxref\n{xref_offset}\n%%EOF\n".encode()
        return bytes(out)

    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for obj_id in range(1, size):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n").encode()
    return bytes(out)


@pytest.fixture
def make_pdf(tmp_path: Path):
    """Factory: write a PDF to tmp_path and return its path as a str.

    `prefix` prepends raw bytes before %PDF, like the HTML that some download
    pages inject (the cause of the PyPDF2 crash on the CD player manual).
    """
    def _make(pages: list[str], name: str = "doc.pdf", prefix: bytes = b"",
              xref_stream: bool = False) -> str:
        path = tmp_path / name
        path.write_bytes(prefix + build_pdf(pages, xref_stream=xref_stream))
        return str(path)
    return _make


@pytest.fixture
def make_docx(tmp_path: Path):
    """Factory: write a .docx with one paragraph per string and return its path."""
    def _make(paragraphs: list[str], name: str = "doc.docx") -> str:
        doc = DocxDocument()
        for paragraph in paragraphs:
            doc.add_paragraph(paragraph)
        path = tmp_path / name
        doc.save(str(path))
        return str(path)
    return _make


@pytest.fixture
def make_txt(tmp_path: Path):
    """Factory: write a UTF-8 text file and return its path."""
    def _make(text: str, name: str = "doc.txt") -> str:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)
    return _make
