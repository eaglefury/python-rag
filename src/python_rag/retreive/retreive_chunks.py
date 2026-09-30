from python_rag.types.types import DocumentMetadata
from python_rag.vector_store import get_vector_store


def get_documents(query: str, top_k: int = 5):
    # Shared Gemini-backed store (see vector_store.py)
    chroma = get_vector_store()
    results = chroma.similarity_search(query, k=top_k)
    return results
