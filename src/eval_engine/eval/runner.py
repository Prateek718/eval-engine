"""Run L1 output-quality scoring over a set of adjudications.

Builds a ScoreInput per claim from the agent's output, the golden label, and the
clause texts the agent retrieved, then scores and aggregates. When a run carries
a trace_id, its three L1 scores are bonded onto that Langfuse trace.
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
from eval_engine.eval.l2_trajectory import (
    L2Result,
    TrajectoryInput,
    aggregate,
    score_trajectory,
)
from eval_engine.observability.tracing import NullTracer, Score, Tracer


@dataclass
class AgentOutput:
    """The agent's result for one claim, flattened for scoring."""

    claim_id: str
    decision: str
    payable_amount: int
    reasoning: str
    retrieved_clause_texts: list[str]
    policy_terms_text: str = ""  # the looked-up policy terms; part of the evidence base
    trace_id: str | None = None  # the run's Langfuse trace, for bonding scores to it


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
    tracer: Tracer | None = None,
) -> L1Result:
    active_tracer: Tracer = tracer or NullTracer()
    per_claim: dict[str, OutputQualityScores] = {}
    for out in outputs:
        label = labels[out.claim_id]
        item = _build_input(out, label, claim_texts[out.claim_id])
        try:
            scores = await scorer.score(item)
        except Exception:  # transient scoring/API errors skip this claim
            continue
        per_claim[out.claim_id] = scores
        if out.trace_id:
            active_tracer.write_scores(
                out.trace_id,
                [
                    Score("faithfulness", scores.faithfulness),
                    Score("answer_relevancy", scores.answer_relevancy),
                    Score("context_precision", scores.context_precision),
                ],
            )

    def mean(attr: str) -> float:
        vals = [getattr(s, attr) for s in per_claim.values()]
        return statistics.fmean(vals) if vals else 0.0

    return L1Result(
        per_claim=per_claim,
        mean_faithfulness=mean("faithfulness"),
        mean_answer_relevancy=mean("answer_relevancy"),
        mean_context_precision=mean("context_precision"),
    )


def run_l2(
    inputs: list[TrajectoryInput],
    tracer: Tracer | None = None,
) -> L2Result:
    """Score each run's tool trajectory and bond the scores to its trace.

    Pure and synchronous: trajectory scoring is set arithmetic over tool names,
    no LLM and no network (unlike L1's Ragas path). The only side effect is
    writing scores onto traces when a trace_id is present.
    """
    active_tracer: Tracer = tracer or NullTracer()
    per_claim = {}
    for item in inputs:
        scores = score_trajectory(item)
        per_claim[item.claim_id] = scores
        if item.trace_id:
            active_tracer.write_scores(
                item.trace_id,
                [
                    Score("tool_precision", scores.tool_precision),
                    Score("tool_recall", scores.tool_recall),
                    Score("step_overage", float(scores.step_overage)),
                ],
            )
    return aggregate(per_claim)
