import pytest

import python_rag
from python_rag.ingest import ingest_document as ingest_module
from python_rag.retreive import answer as answer_module


@pytest.fixture
def run_cli(monkeypatch):
    """Run python_rag.main() with scripted answers to its input() prompts."""
    def _run(*replies: str):
        replies_iter = iter(replies)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(replies_iter))
        python_rag.main()
    return _run


@pytest.fixture
def ingested(monkeypatch):
    # main() imports ingest_document at call time, so patching the module attribute works.
    calls = []
    monkeypatch.setattr(ingest_module, "ingest_document", calls.append)
    return calls


@pytest.fixture
def asked(monkeypatch):
    calls = []

    def fake_get_answer(question):
        calls.append(question)
        return "stub answer"

    monkeypatch.setattr(answer_module, "get_answer", fake_get_answer)
    return calls


@pytest.mark.parametrize("command", ["ingest", "INGEST"])
def test_ingest_existing_file(run_cli, ingested, data_file, command):
    path = data_file("notes.txt")

    run_cli(command, path)

    assert ingested == [path]


def test_ingest_missing_file_reports_and_skips(run_cli, ingested, tmp_path, capsys):
    run_cli("ingest", str(tmp_path / "missing.pdf"))

    assert ingested == []
    assert "does not exist" in capsys.readouterr().out


def test_question_prints_answer(run_cli, asked, capsys):
    run_cli("question", "Which region is it?")

    assert asked == ["Which region is it?"]
    assert "Answer: stub answer" in capsys.readouterr().out
