"""Tests for L1 output-quality scoring. Hermetic: a stub scorer replaces Ragas,
so no Gemini, no network. Verifies aggregation, input construction, and that the
real golden set loads and matches the agent's Decision enum.
"""

import math

from eval_engine.agent.schema import Decision
from eval_engine.eval import (
    AgentOutput,
    GoldenLabel,
    OutputQualityScores,
    ScoreInput,
    load_golden_set,
    run_l1,
)
from eval_engine.eval.l1_output_quality import answer_to_text, label_to_reference


class StubScorer:
    """Returns fixed scores, recording inputs so tests can assert construction."""

    def __init__(self, scores: OutputQualityScores) -> None:
        self._scores = scores
        self.seen: list[ScoreInput] = []

    async def score(self, item: ScoreInput) -> OutputQualityScores:
        self.seen.append(item)
        return self._scores


def _label(cid: str) -> GoldenLabel:
    return GoldenLabel(
        claim_id=cid,
        decision=Decision.PARTIAL,
        payable_amount=30000,
        governing_clauses=[],
        expected_tool_calls=["lookup_policy"],
        reasoning="capped by sub-limit",
        reviewed=True,
    )


def _output(cid: str) -> AgentOutput:
    return AgentOutput(
        claim_id=cid,
        decision="partial",
        payable_amount=30000,
        reasoning="cap applied",
        retrieved_clause_texts=["some clause text"],
        policy_terms_text="policy terms json",
    )


def test_answer_and_reference_flattening() -> None:
    a = answer_to_text("partial", 30000, "cap applied")
    assert "partial" in a and "30000" in a and "cap applied" in a
    r = label_to_reference("partial", 30000, "capped")
    assert "partial" in r and "30000" in r


def test_run_l1_aggregates_means() -> None:
    import asyncio

    fixed = OutputQualityScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)
    scorer = StubScorer(fixed)
    outputs = [_output("C01"), _output("C02")]
    labels = {"C01": _label("C01"), "C02": _label("C02")}
    claim_texts = {"C01": "claim one", "C02": "claim two"}

    result = asyncio.run(run_l1(outputs, labels, claim_texts, scorer))

    assert math.isclose(result.mean_faithfulness, 0.9)
    assert math.isclose(result.mean_answer_relevancy, 0.8)
    assert math.isclose(result.mean_context_precision, 0.7)
    assert set(result.per_claim) == {"C01", "C02"}


def test_run_l1_builds_correct_inputs() -> None:
    import asyncio

    scorer = StubScorer(OutputQualityScores(1.0, 1.0, 1.0))
    outputs = [_output("C01")]
    labels = {"C01": _label("C01")}
    claim_texts = {"C01": "the claim text"}

    asyncio.run(run_l1(outputs, labels, claim_texts, scorer))

    assert len(scorer.seen) == 1
    item = scorer.seen[0]
    assert item.claim_text == "the claim text"
    assert "partial" in item.answer_text
    assert item.retrieved_contexts == ["some clause text"]
    # evidence base = retrieved clauses + the looked-up policy terms
    assert "some clause text" in item.evidence_contexts
    assert "policy terms json" in item.evidence_contexts
    assert "partial" in item.reference_text


def test_real_golden_set_loads_and_is_reviewed() -> None:
    labels = load_golden_set()
    assert len(labels) == 31
    assert all(isinstance(label.decision, Decision) for label in labels)
    assert all(label.reviewed for label in labels)
    assert all(label.payable_amount == 0 for label in labels if label.decision is Decision.DENIED)
