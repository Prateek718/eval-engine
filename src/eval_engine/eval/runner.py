"""Run L1 output-quality scoring over a set of adjudications.

Builds a ScoreInput per claim from the agent's output, the golden label, and the
clause texts the agent retrieved, then scores and aggregates.
"""

import statistics
from dataclasses import dataclass

from eval_engine.eval.golden import GoldenLabel
from eval_engine.eval.l1_output_quality import (
    OutputQualityScorer,
    OutputQualityScores,
    ScoreInput,
    answer_to_text,
    label_to_reference,
)


@dataclass
class AgentOutput:
    """The agent's result for one claim, flattened for scoring."""

    claim_id: str
    decision: str
    payable_amount: int
    reasoning: str
    retrieved_clause_texts: list[str]
    policy_terms_text: str = ""  # the looked-up policy terms; part of the evidence base


@dataclass
class L1Result:
    per_claim: dict[str, OutputQualityScores]
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float


def _build_input(out: AgentOutput, label: GoldenLabel, claim_text: str) -> ScoreInput:
    evidence = out.retrieved_clause_texts.copy()
    if out.policy_terms_text:
        evidence.append(out.policy_terms_text)
    return ScoreInput(
        claim_text=claim_text,
        answer_text=answer_to_text(out.decision, out.payable_amount, out.reasoning),
        evidence_contexts=evidence,
        retrieved_contexts=out.retrieved_clause_texts,
        reference_text=label_to_reference(
            label.decision.value, label.payable_amount, label.reasoning
        ),
    )


async def run_l1(
    outputs: list[AgentOutput],
    labels: dict[str, GoldenLabel],
    claim_texts: dict[str, str],
    scorer: OutputQualityScorer,
) -> L1Result:
    per_claim: dict[str, OutputQualityScores] = {}
    for out in outputs:
        label = labels[out.claim_id]
        item = _build_input(out, label, claim_texts[out.claim_id])
        try:
            per_claim[out.claim_id] = await scorer.score(item)
        except Exception:  # transient scoring/API errors skip this claim
            continue

    def mean(attr: str) -> float:
        vals = [getattr(s, attr) for s in per_claim.values()]
        return statistics.fmean(vals) if vals else 0.0

    return L1Result(
        per_claim=per_claim,
        mean_faithfulness=mean("faithfulness"),
        mean_answer_relevancy=mean("answer_relevancy"),
        mean_context_precision=mean("context_precision"),
    )
