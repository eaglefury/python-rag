from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# GoogleGenerativeAIEmbeddings reads GOOGLE_API_KEY from the environment, so load
# .env here rather than relying on answer.py having been imported first.
load_dotenv()

# Single place for the embedding model: ingestion and retrieval must use the same
# model, otherwise query vectors are compared against incompatible document vectors.
EMBEDDING_MODEL = "gemini-embedding-001"
# Anchored to the project root (src/python_rag/vector_store.py -> parents[2]) rather
# than the current working directory, so ingest and retrieval always use the same
# database no matter where the command is run from. CHROMA_DIR overrides it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = str(
    Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / ".chroma")).expanduser().resolve())


# Shared by ingestion and retrieval; cached so the client is created once per process.
@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    # Gemini embeds documents with task_type RETRIEVAL_DOCUMENT and queries with
    # RETRIEVAL_QUERY by default, which is what we want for search.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    # Collection named after the model so vectors from different models (which have
    # different dimensions) never end up in the same collection.
    return Chroma(
        collection_name=EMBEDDING_MODEL,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
