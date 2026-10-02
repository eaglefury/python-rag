import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "resources" / "user-manual-cd-player.pdf"

load_dotenv(ROOT / ".env")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
# Reasoning models can exceed DeepEval's default ~88s per-call timeout; a timeout
# fails the test and its retry repeats the paid call.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")


@pytest.fixture(scope="session")
def ingested_manual(tmp_path_factory):
    """Ingest the manual once per session into a temporary Chroma DB.

    Uses real OpenAI embeddings (~28k tokens, well under $0.01). The user's
    project .chroma is never touched.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (.env or environment)")
    if not MANUAL.exists():
        pytest.skip(f"eval document missing: {MANUAL}")

    import python_rag.vector_store as vector_store
    from python_rag.ingest.ingest_document import ingest_document

    # get_vector_store() reads PERSIST_DIRECTORY at call time, so patching the
    # module attribute and clearing its cache redirects it to the temp DB.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vector_store, "PERSIST_DIRECTORY", str(tmp_path_factory.mktemp("chroma")))
        vector_store.get_vector_store.cache_clear()
        ingest_document(str(MANUAL))
        yield
    vector_store.get_vector_store.cache_clear()


@pytest.fixture(scope="session")
def run_rag(ingested_manual):
    """answer_with_context, memoized so each question hits OpenAI only once per session."""
    from python_rag.retreive.answer import answer_with_context

    cache = {}

    def _run(question: str):
        if question not in cache:
            cache[question] = answer_with_context(question)
        return cache[question]

    return _run
