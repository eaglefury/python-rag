import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from python_rag.ingest.ingest_document import ingest_document
from python_rag.retreive import answer
from python_rag.retreive.answer import CHAT_MODEL, answer_with_context, build_prompt, get_answer

DOCS = [
    Document(page_content="The region number of this player is 2.", metadata={"source": "m.pdf"}),
    Document(page_content="Press ZOOM during playback.", metadata={"source": "m.pdf"}),
]


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace ChatOpenAI with a stub that records prompts and returns a fixed answer."""
    calls = []

    class StubChat:
        def __init__(self, model):
            calls.append({"model": model})

        def invoke(self, prompt):
            calls[-1]["prompt"] = prompt
            return AIMessage(content="stub answer")

    monkeypatch.setattr(answer, "ChatOpenAI", StubChat)
    return calls


@pytest.fixture
def fixed_retrieval(monkeypatch):
    """Make retrieval return DOCS and record the requested top_k."""
    requested = {}

    def fake_get_rag_documents(query, top_k=5):
        requested["top_k"] = top_k
        return DOCS[:top_k]

    monkeypatch.setattr(answer, "get_rag_documents", fake_get_rag_documents)
    return requested


def test_prompt_contains_question_context_and_fallback_instruction():
    prompt = build_prompt("Which region is it?", DOCS)

    assert "Question: Which region is it?" in prompt
    assert all(d.page_content in prompt for d in DOCS)
    assert "say you do not know" in prompt


def test_answer_with_context_returns_answer_and_documents(fake_chat, fixed_retrieval):
    text, documents = answer_with_context("Which region is it?")

    assert text == "stub answer"
    assert documents == DOCS
    assert fake_chat[0]["model"] == CHAT_MODEL
    assert fake_chat[0]["prompt"] == build_prompt("Which region is it?", DOCS)


def test_top_k_is_passed_to_retrieval(fake_chat, fixed_retrieval):
    _, documents = answer_with_context("q", top_k=1)

    assert fixed_retrieval["top_k"] == 1
    assert documents == DOCS[:1]


def test_get_answer_prints_retrieved_count_and_returns_text(fake_chat, fixed_retrieval, capsys):
    assert get_answer("Which region is it?") == "stub answer"
    assert "Retrieved 2 documents" in capsys.readouterr().out


def test_end_to_end_offline(fake_chat, fake_vector_store, make_txt):
    # Real ingest + retrieval (fake embeddings), stubbed LLM: the ingested text
    # must reach the prompt.
    ingest_document(make_txt("The screen saver starts after 20 minutes.", name="m.txt"))

    _, documents = answer_with_context("The screen saver starts after 20 minutes.")

    assert documents[0].page_content == "The screen saver starts after 20 minutes."
    assert "The screen saver starts after 20 minutes." in fake_chat[0]["prompt"]
