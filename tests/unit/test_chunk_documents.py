import tiktoken

from python_rag.ingest.chunk_documents import (
    chunk_document,
    get_chunks_with_metadata_for_document,
)
from python_rag.types.types import AccessRole, DocumentMetadata
from python_rag.vector_store import EMBEDDING_ENCODING

ENCODING = tiktoken.get_encoding(EMBEDDING_ENCODING)


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def long_text(paragraphs: int = 40) -> str:
    """Varied multi-paragraph text of a few thousand tokens."""
    return "\n\n".join(
        f"Section {i}. The player stores setting number {i} and shows message {i * 7} "
        f"when button {i % 9} is pressed during playback of disc {i * 3}. " * 6
        for i in range(paragraphs)
    )


def metadata(source: str = "manual.pdf") -> DocumentMetadata:
    return DocumentMetadata(source=source, tag=[".pdf"])


def test_short_text_is_a_single_chunk():
    assert chunk_document("Just one short sentence.") == ["Just one short sentence."]


def test_chunks_never_exceed_default_token_limit():
    chunks = chunk_document(long_text())

    assert len(chunks) > 1
    assert max(token_count(c) for c in chunks) <= 800


def test_chunks_respect_custom_size():
    chunks = chunk_document(long_text(), chunk_size=100, chunk_overlap=10)

    assert max(token_count(c) for c in chunks) <= 100


def test_ids_are_stable_across_runs():
    # Stable IDs are what make re-ingesting a file overwrite instead of duplicate.
    first = get_chunks_with_metadata_for_document(long_text(), metadata())
    second = get_chunks_with_metadata_for_document(long_text(), metadata())

    assert [d.id for d in first] == [d.id for d in second]


def test_repeated_text_in_one_file_gets_unique_ids():
    paragraph = "Identical safety notice repeated on many pages. " * 120  # ~700 tokens
    docs = get_chunks_with_metadata_for_document("\n\n".join([paragraph] * 3), metadata())

    assert len({d.page_content for d in docs}) < len(docs), "test needs duplicate chunks"
    assert len({d.id for d in docs}) == len(docs)


def test_same_text_in_different_files_gets_different_ids():
    a = get_chunks_with_metadata_for_document("Same content.", metadata("a.pdf"))
    b = get_chunks_with_metadata_for_document("Same content.", metadata("b.pdf"))

    assert a[0].id != b[0].id


def test_metadata_fields():
    meta = DocumentMetadata(source="guide.docx", tag=[".docx", "setup"],
                            access_role=AccessRole.ADMIN)

    doc = get_chunks_with_metadata_for_document("Some text.", meta)[0]

    assert doc.metadata == {"source": "guide.docx", "tag": ".docx,setup", "access_role": "admin"}


def test_empty_tag_list_is_stored_as_empty_string():
    doc = get_chunks_with_metadata_for_document(
        "Some text.", DocumentMetadata(source="x.txt", tag=[]))[0]

    assert doc.metadata["tag"] == ""
