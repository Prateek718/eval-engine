"""Baseline-integrity gate: the frozen baseline sheet must stay well-formed and
self-consistent, and aligned with the golden set it is diffed against.

This is not an agent-regression gate -- detecting a degraded agent requires
running it, which CI does not. It is the hermetic counterpart: it guards the
*reference* itself, so a corrupted, truncated, or out-of-sync baseline cannot
silently invalidate every L3 comparison built on top of it. Reads two committed
files via their real loaders; no agent, no network.
"""

from __future__ import annotations

import pytest

from eval_engine.eval.golden import load_golden_set
from eval_engine.eval.l3_regression import (
    _DECISION_CLASSES,
    DriftRow,
    load_sheet,
)

_BASELINE_PATH = "data/baseline_scores.json"

# Columns whose values must lie in [0, 1]. step_overage is excluded: it is a
# step count, bounded below by 0 but not above by 1.
_UNIT_INTERVAL_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "tool_precision",
    "tool_recall",
)


@pytest.fixture(scope="module")
def baseline() -> list[DriftRow]:
    """Load the frozen baseline through the same loader the runtime uses, so the
    gate fails if the committed file ever drifts from the DriftRow schema.
    """
    return load_sheet(_BASELINE_PATH)


def test_baseline_loads_nonempty(baseline: list[DriftRow]) -> None:
    assert baseline, "baseline sheet is empty"


def test_claim_ids_unique(baseline: list[DriftRow]) -> None:
    ids = [row.claim_id for row in baseline]
    assert len(ids) == len(set(ids)), "duplicate claim_id in baseline"


def test_claim_set_matches_golden(baseline: list[DriftRow]) -> None:
    """The headline invariant: the baseline grades the same claims the golden
    set defines. A baseline that has drifted out of sync with the golden set
    would make every L3 delta meaningless, so set-equality (not a hardcoded
    count) is asserted -- self-updating if the golden set legitimately changes.
    """
    baseline_ids = {row.claim_id for row in baseline}
    golden_ids = {label.claim_id for label in load_golden_set()}
    assert baseline_ids == golden_ids, (
        f"baseline/golden claim sets diverge: "
        f"only in baseline={baseline_ids - golden_ids}, "
        f"only in golden={golden_ids - baseline_ids}"
    )


def test_decisions_valid(baseline: list[DriftRow]) -> None:
    for row in baseline:
        assert row.decision in _DECISION_CLASSES, (
            f"{row.claim_id}: invalid decision {row.decision!r}"
        )


def test_amounts_nonnegative(baseline: list[DriftRow]) -> None:
    for row in baseline:
        assert row.payable_amount >= 0, (
            f"{row.claim_id}: negative payable_amount {row.payable_amount}"
        )


def test_denied_claims_pay_nothing(baseline: list[DriftRow]) -> None:
    """A denied claim with a payout is internally incoherent."""
    for row in baseline:
        if row.decision == "denied":
            assert row.payable_amount == 0, (
                f"{row.claim_id}: denied but payable_amount {row.payable_amount}"
            )


def test_unit_interval_scores_in_range(baseline: list[DriftRow]) -> None:
    for row in baseline:
        for column in _UNIT_INTERVAL_COLUMNS:
            value = getattr(row, column)
            assert 0.0 <= value <= 1.0, f"{row.claim_id}: {column}={value} outside [0, 1]"


def test_step_overage_nonnegative(baseline: list[DriftRow]) -> None:
    for row in baseline:
        assert row.step_overage >= 0, f"{row.claim_id}: negative step_overage {row.step_overage}"
