"""Cheap eval tier: `uv run pytest -m eval`.

No LLM judge. Costs only the app's own calls: embedding the manual once
(< $0.01) and one gpt-5-mini answer per question (14 questions).
"""
import re

import pytest

from .golden_dataset import IN_SCOPE, KNOWN_RETRIEVAL_MISSES, OUT_OF_SCOPE

pytestmark = pytest.mark.eval

# Phrases the prompt's "say you do not know" instruction typically produces.
DECLINE_PATTERN = re.compile(
    r"do not know|don't know|not (contain|include|mention|provide|cover)|"
    r"no information|not available|cannot (answer|find|determine)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return " ".join(text.split())


def with_known_misses(cases):
    return [
        pytest.param(c, id=c.id, marks=pytest.mark.xfail(reason=KNOWN_RETRIEVAL_MISSES[c.id]))
        if c.id in KNOWN_RETRIEVAL_MISSES else pytest.param(c, id=c.id)
        for c in cases
    ]


@pytest.mark.parametrize("case", with_known_misses(IN_SCOPE))
def test_retrieval_finds_answer(case, run_rag):
    """The chunk containing the answer is among the retrieved chunks."""
    _, documents = run_rag(case.question)

    assert any(case.answer_phrase in normalize(d.page_content) for d in documents), (
        f"'{case.answer_phrase}' not in any of the {len(documents)} retrieved chunks")


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c.id)
def test_answer_contains_key_facts(case, run_rag):
    answer, _ = run_rag(case.question)

    missing = [p for p in case.must_match if not re.search(p, answer, re.IGNORECASE)]
    assert not missing, f"answer lacks {missing}: {answer!r}"


@pytest.mark.parametrize("case", OUT_OF_SCOPE, ids=lambda c: c.id)
def test_declines_questions_the_documents_cannot_answer(case, run_rag):
    answer, _ = run_rag(case.question)

    assert DECLINE_PATTERN.search(answer), f"expected a 'do not know' answer, got: {answer!r}"
