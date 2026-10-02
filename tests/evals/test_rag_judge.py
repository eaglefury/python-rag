"""Paid eval tier: `uv run pytest -m eval_judge`. Run deliberately (before a
release, or after changing prompts, models or chunking).

LLM judge on a small subset (`judged=True` in golden_dataset.py), faithfulness
only: did the answer make anything up? Retrieval and key facts are covered for
free by test_rag_checks.py. Roughly 4 questions x ~3 judge calls.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

from .golden_dataset import IN_SCOPE

pytestmark = pytest.mark.eval_judge

JUDGED = [c for c in IN_SCOPE if c.judged]


@pytest.fixture(scope="session")
def judge():
    # gpt-5-mini at low reasoning effort: ~5x cheaper per token than gpt-5 and far
    # fewer hidden reasoning tokens (gpt-5 at default effort timed out at 88s).
    # It's the app's own model, but faithfulness only checks the answer's claims
    # against the retrieved text, which is less prone to self-grading bias.
    return OpenAIModel(model="gpt-5-mini", generation_kwargs={"reasoning_effort": "low"})


@pytest.mark.parametrize("case", JUDGED, ids=lambda c: c.id)
def test_answer_is_faithful_to_retrieved_chunks(case, run_rag, judge):
    answer, documents = run_rag(case.question)
    test_case = LLMTestCase(
        input=case.question,
        actual_output=answer,
        expected_output=case.expected_output,
        retrieval_context=[d.page_content for d in documents],
    )

    assert_test(test_case, [FaithfulnessMetric(threshold=0.8, model=judge)])
