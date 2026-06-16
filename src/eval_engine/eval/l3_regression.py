"""L3: regression monitoring -- threshold-delta drift against a frozen baseline.

L1/L2/L4 answer "is this run good?"; L3 answers "has it changed?". It compares a
run's per-claim scores against a frozen baseline sheet and flags columns whose
mean moved past a configured threshold.

Drift here is a threshold delta, not a statistical test. With a 31-claim golden
set and a non-deterministic agent, distributional tests (K-S, chi-square) are
underpowered and trip on run-to-run noise; an explicit, owned threshold is the
honest signal at this scale. The watched columns are the quality and outcome
numbers L1/L2 already produce plus the adjudication itself -- cost is not here;
it is surfaced live from the tracing layer on the dashboard, not diffed against
a golden baseline.

The whole module is pure: two sheets in, a verdict out. No agent, no network,
no external drift library -- so it is fully testable offline.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval_engine.eval.l1_output_quality import OutputQualityScores
from eval_engine.eval.l2_trajectory import TrajectoryScores

# Numerical columns drift-checked by mean delta. payable_amount is checked as a
# fractional shift (see below), the rest as absolute deltas on their native
# [0,1] / small-int scales.
_NUMERIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "tool_precision",
    "tool_recall",
    "step_overage",
)

_DECISION_CLASSES = ("payable", "partial", "denied")


@dataclass
class DriftRow:
    """One claim's row in the score sheet: the columns L3 watches for drift."""

    claim_id: str
    decision: str
    payable_amount: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    tool_precision: float
    tool_recall: float
    step_overage: int


@dataclass
class Thresholds:
    """Per-signal drift thresholds. Starting values tuned against the first
    real baseline; carried in config, not hardcoded at the call site.
    """

    score: float = 0.10  # absolute mean delta on [0,1] score columns
    step_overage: float = 0.10  # absolute mean delta on step_overage
    amount_fraction: float = 0.10  # fractional shift in mean payable_amount
    denial_rate: float = 0.065  # ~2/31: absolute shift in denied share


@dataclass
class DriftResult:
    """The drift verdict for one current-vs-baseline comparison.

    All fields are plain values so the result serialises and logs without any
    external types. ``flags`` is the subset of signals that breached threshold;
    ``n_drifted`` is its length, the single number the dashboard alerts on.
    """

    column_deltas: dict[str, float]  # signed mean delta per numeric column
    denial_rate_delta: float  # signed shift in the denied-decision share
    decision_share_deltas: dict[str, float]  # signed shift per decision class
    flags: list[str] = field(default_factory=list)  # signals over threshold
    n_drifted: int = 0


def build_score_sheet(
    decisions: dict[str, tuple[str, int]],
    l1_per_claim: dict[str, OutputQualityScores],
    l2_per_claim: dict[str, TrajectoryScores],
) -> list[DriftRow]:
    """Join the three per-claim sources into one sheet, on claim_id.

    ``decisions`` maps claim_id -> (decision, payable_amount), unpacked by the
    caller from the run's AgentOutputs so this module need not import the
    runner. A claim earns a row only if it is present in all three sources: L1
    skips claims that error during scoring, so its key set can be a subset, and
    a partial row would silently bias the means. Rows are emitted in sorted
    claim_id order for a stable, diffable sheet.
    """
    common = decisions.keys() & l1_per_claim.keys() & l2_per_claim.keys()
    rows: list[DriftRow] = []
    for claim_id in sorted(common):
        decision, amount = decisions[claim_id]
        l1 = l1_per_claim[claim_id]
        l2 = l2_per_claim[claim_id]
        rows.append(
            DriftRow(
                claim_id=claim_id,
                decision=decision,
                payable_amount=amount,
                faithfulness=l1.faithfulness,
                answer_relevancy=l1.answer_relevancy,
                context_precision=l1.context_precision,
                tool_precision=l2.tool_precision,
                tool_recall=l2.tool_recall,
                step_overage=l2.step_overage,
            )
        )
    return rows


def _column_mean(rows: list[DriftRow], column: str) -> float:
    vals = [float(getattr(r, column)) for r in rows]
    return statistics.fmean(vals) if vals else 0.0


def _decision_share(rows: list[DriftRow], cls: str) -> float:
    return sum(1 for r in rows if r.decision == cls) / len(rows) if rows else 0.0


def score_regression(
    reference: list[DriftRow],
    current: list[DriftRow],
    thresholds: Thresholds,
) -> DriftResult:
    """Compare a current sheet against a frozen baseline and flag drift.

    Numeric columns drift on absolute mean delta against their threshold;
    payable_amount drifts on fractional shift of its mean (an absolute rupee
    threshold would not transfer across baselines of different magnitude);
    decision drifts on per-class share shift, with the denied share called out
    separately as the headline signal. Deltas are signed (current - baseline)
    so direction is legible; flags fire on magnitude.
    """
    column_deltas: dict[str, float] = {}
    flags: list[str] = []

    for column in _NUMERIC_COLUMNS:
        delta = _column_mean(current, column) - _column_mean(reference, column)
        column_deltas[column] = delta
        limit = thresholds.step_overage if column == "step_overage" else thresholds.score
        if abs(delta) > limit:
            flags.append(column)

    # payable_amount: fractional shift of the mean, against a guarded baseline.
    ref_amount = _column_mean(reference, "payable_amount")
    cur_amount = _column_mean(current, "payable_amount")
    amount_delta = cur_amount - ref_amount
    column_deltas["payable_amount"] = amount_delta
    if ref_amount and abs(amount_delta) / ref_amount > thresholds.amount_fraction:
        flags.append("payable_amount")

    # decision: per-class share shift, denied called out as the headline.
    decision_share_deltas = {
        cls: _decision_share(current, cls) - _decision_share(reference, cls)
        for cls in _DECISION_CLASSES
    }
    denial_rate_delta = decision_share_deltas["denied"]
    if abs(denial_rate_delta) > thresholds.denial_rate:
        flags.append("denial_rate")

    return DriftResult(
        column_deltas=column_deltas,
        denial_rate_delta=denial_rate_delta,
        decision_share_deltas=decision_share_deltas,
        flags=flags,
        n_drifted=len(flags),
    )


# --- sheet persistence ---------------------------------------------------------
# A score sheet is a local JSON file, not an external service, so it needs no
# injected seam or null default (unlike the tracer/tracker): it either reads and
# writes or it does not. JSON to match golden_set.json -- DVC-trackable and
# human-diffable, so a frozen baseline reviews like any other versioned artifact.


def save_sheet(rows: list[DriftRow], path: str | Path) -> None:
    """Write a score sheet to JSON. Rows are already in stable claim_id order."""
    payload = {"schema_version": "1", "rows": [asdict(r) for r in rows]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sheet(path: str | Path) -> list[DriftRow]:
    """Read a score sheet written by save_sheet, reconstructing typed rows."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DriftRow(**row) for row in payload["rows"]]
