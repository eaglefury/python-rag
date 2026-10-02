from pathlib import Path

import pytest

# Committed sample documents (regenerate with tests/data/make_test_documents.py).
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def data_file():
    """Return the path (as str, which read_document requires) of a file in tests/data."""
    def _path(name: str) -> str:
        path = DATA_DIR / name
        assert path.exists(), f"missing test document {path}; run make_test_documents.py"
        return str(path)
    return _path


@pytest.fixture
def fake_vector_store(tmp_path: Path, monkeypatch):
    """A real Chroma store in tmp_path with offline fake embeddings.

    DeterministicFakeEmbedding gives the same vector for the same text, so a
    query equal to a stored chunk finds that chunk first; no API key needed.
    Patched at both import sites because each module imports the name directly.
    """
    from langchain_chroma import Chroma
    from langchain_core.embeddings import DeterministicFakeEmbedding

    store = Chroma(
        collection_name="test",
        embedding_function=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "chroma"),
    )
    monkeypatch.setattr("python_rag.ingest.ingest_document.get_vector_store", lambda: store)
    monkeypatch.setattr("python_rag.retreive.retreive_chunks.get_vector_store", lambda: store)
    return store
