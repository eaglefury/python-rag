from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# OpenAIEmbeddings reads OPENAI_API_KEY from the environment, so load
# .env here rather than relying on answer.py having been imported first.
load_dotenv()

# Single place for the embedding model: ingestion and retrieval must use the same
# model, otherwise query vectors are compared against incompatible document vectors.
# text-embedding-3-small: 1536-dim vectors, up to 8191 input tokens.
EMBEDDING_MODEL = "text-embedding-3-small"
# Tokenizer used by the text-embedding-3 models; chunk_documents.py sizes chunks with it.
EMBEDDING_ENCODING = "cl100k_base"
# Anchored to the project root (src/python_rag/vector_store.py -> parents[2]) rather
# than the current working directory, so ingest and retrieval always use the same
# database no matter where the command is run from. CHROMA_DIR overrides it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = str(
    Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / ".chroma")).expanduser().resolve())


# Shared by ingestion and retrieval; cached so the client is created once per process.
@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    # Collection named after the model so vectors from different models (which have
    # different dimensions) never end up in the same collection.
    return Chroma(
        collection_name=EMBEDDING_MODEL,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
