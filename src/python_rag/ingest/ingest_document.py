from langchain_core.documents import Document

from python_rag.ingest.chunk_documents import get_chunks_with_metadata_for_document
from pathlib import Path
from python_rag.ingest.read_documents import read_document
from python_rag.types.types import DocumentMetadata
from python_rag.vector_store import get_vector_store


def ingest_document(path: str):
    # Shared vector store (see vector_store.py)
    chroma = get_vector_store()
    document = read_document(path)

    metadata = DocumentMetadata(
        source=document.name, tag=[Path(path).suffix.lower()])

    ingest_documents = get_chunks_with_metadata_for_document(
        text=document.content, document_metadata=metadata
    )

    chroma.add_documents(ingest_documents,)
