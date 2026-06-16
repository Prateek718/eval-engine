"""Score adjudications across the eval layers and bond results to traces.

Three layers live here, each scoring the same agent runs from a different angle:

- run_l1: output quality via Ragas (faithfulness, answer relevancy, context
  precision). Async -- the metrics are LLM-judged, so it is I/O-bound on the
  evaluator model.
- run_l2: tool-trajectory precision/recall and step overage against the golden
  expected_tool_calls. Pure and synchronous -- set arithmetic, no network.
- run_l4: decision stability under input perturbation. Drives its own agent
  runs over perturbed claims (injected run_agent), so it is heavier than L1/L2
  and lives behind its own script.

Each layer writes its scores onto the run's Langfuse trace when a trace_id is
present, and is a no-op on tracing when it is absent.
"""

import hashlib
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from eval_engine.agent.graph import AgentState
from eval_engine.eval.golden import Claim, GoldenLabel
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
from eval_engine.eval.l3_regression import (
    DriftResult,
    Thresholds,
    build_score_sheet,
    load_sheet,
    save_sheet,
    score_regression,
)
from eval_engine.eval.l4_robustness import AXES, perturb
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


@dataclass
class AxisStability:
    """Per-axis robustness reading over the golden set."""

    decision_stability: float  # fraction of perturbations preserving the golden decision
    amount_stability: float  # fraction preserving decision AND exact amount
    n: int  # perturbations attempted on this axis (one per claim, minus agent failures)


@dataclass
class L4Result:
    per_axis: dict[str, AxisStability]
    mean_decision_stability: float  # over all axes
    mean_amount_stability: float


def run_l4(
    claims: dict[str, Claim],
    labels: dict[str, GoldenLabel],
    run_agent: Callable[[str, str, str], AgentState],
    tracer: Tracer | None = None,
    seed: int = 0,
) -> L4Result:
    """Perturb each claim along each axis, re-run the agent, and check the
    adjudication held against the golden label.

    The agent is injected (not built here): L4 reuses the one rig the script
    built, and the scoring is testable offline with a stub run_agent. The bar is
    strict -- decision AND exact amount must match the golden label -- but
    decision- and amount-stability are reported separately so a payout that
    drifts while the decision holds is visible, not blended into one number.

    seed makes the run reproducible: perturbations are keyed by (claim, axis,
    seed), so the same seed reproduces the same perturbed inputs.
    """
    active_tracer: Tracer = tracer or NullTracer()
    per_axis: dict[str, AxisStability] = {}

    for axis in AXES:
        decision_hits = 0
        amount_hits = 0
        n = 0
        for claim_id, claim in claims.items():
            label = labels[claim_id]
            # stable per-(claim, axis) seed: hashlib (not built-in hash, which is
            # randomized per process) so perturbations reproduce across runs, and
            # distinct across axes so one claim is not perturbed identically by all.
            digest = hashlib.md5(f"{claim_id}:{axis}".encode()).hexdigest()
            claim_seed = seed + int(digest, 16) % 10_000
            perturbed = perturb(claim.claim_text, axis, claim_seed)
            try:
                state = run_agent(perturbed, claim.policy_id, claim_id)
            except Exception:  # transient agent/API error: skip this perturbation
                continue
            if state.result is None:
                continue
            n += 1
            decision_ok = state.result.decision.value == label.decision.value
            amount_ok = decision_ok and state.result.payable_amount == label.payable_amount
            decision_hits += int(decision_ok)
            amount_hits += int(amount_ok)
            if state.trace_id:
                active_tracer.write_scores(
                    state.trace_id,
                    [
                        Score(f"robustness_{axis}_decision", float(decision_ok)),
                        Score(f"robustness_{axis}_amount", float(amount_ok)),
                    ],
                )
        per_axis[axis] = AxisStability(
            decision_stability=decision_hits / n if n else 0.0,
            amount_stability=amount_hits / n if n else 0.0,
            n=n,
        )

    def overall(attr: str) -> float:
        vals = [getattr(a, attr) for a in per_axis.values() if a.n]
        return float(sum(vals) / len(vals)) if vals else 0.0

    return L4Result(
        per_axis=per_axis,
        mean_decision_stability=overall("decision_stability"),
        mean_amount_stability=overall("amount_stability"),
    )


def run_l3(
    decisions: dict[str, tuple[str, int]],
    l1: L1Result,
    l2: L2Result,
    current_path: str | Path,
    baseline_path: str | Path,
    thresholds: Thresholds,
) -> DriftResult | None:
    """Build this run's score sheet, persist it, and diff it against the frozen
    baseline.

    L3 computes no scores of its own: it joins the per-claim L1/L2 results
    (already computed for this run) with the adjudications and compares the
    sheet to a baseline. The current sheet is always written -- so a chosen run
    can later be promoted to baseline -- but drift is scored only once a
    baseline exists. The first run returns None (it has nothing to drift
    against) and simply records itself.
    """
    sheet = build_score_sheet(decisions, l1.per_claim, l2.per_claim)
    save_sheet(sheet, current_path)
    if not Path(baseline_path).exists():
        return None
    baseline = load_sheet(baseline_path)
    return score_regression(baseline, sheet, thresholds)
