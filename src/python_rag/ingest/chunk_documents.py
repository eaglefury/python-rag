from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

from python_rag.types.types import DocumentMetadata
from python_rag.vector_store import EMBEDDING_ENCODING


# Sizes are in tokens, counted locally with tiktoken using the embedding model's own
# encoding (no API calls). 800 tokens is far below the model's 8191-token limit and
# in the 512-1024 range that keeps each embedding focused on one topic.
def chunk_document(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=EMBEDDING_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_chunks_with_metadata_for_document(text: str, document_metadata: DocumentMetadata,) -> list[Document]:
    documents = []

    chunks = chunk_document(text)
    for i, c in enumerate(chunks):
        # Deterministic ID: the same file ingested again yields the same IDs, so
        # Chroma upserts (overwrites) existing chunks instead of adding duplicates.
        # Source and index are included so identical text in different files, or
        # repeated within one file, still gets a unique ID.
        chunk_hash = hashlib.sha256(
            f"{document_metadata.source}:{i}:{c}".encode('utf-8')).hexdigest()

        documents.append(Document(id=chunk_hash, page_content=c, metadata={
                         "source": document_metadata.source,
                         "tag": ",".join(document_metadata.tag) if document_metadata.tag else "",
                         "access_role": document_metadata.access_role.value}))
    return documents
