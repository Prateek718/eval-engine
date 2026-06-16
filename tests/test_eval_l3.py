"""L3 regression-monitoring tests. Offline: pure sheet arithmetic, no agent,
no network, no external drift library.

Three drift scenarios anchor the suite -- a run identical to baseline flags
nothing, a run with collapsed quality and flipped decisions flags the right
signals, and a comparison with no baseline yet is well-defined -- plus the
sheet-build join rules (intersection only, stable order).
"""

from eval_engine.eval.l1_output_quality import OutputQualityScores
from eval_engine.eval.l2_trajectory import TrajectoryScores
from eval_engine.eval.l3_regression import (
    DriftResult,
    DriftRow,
    Thresholds,
    build_score_sheet,
    load_sheet,
    save_sheet,
    score_regression,
)
from eval_engine.eval.runner import L1Result, L2Result, run_l3


def _healthy_sources() -> tuple[
    dict[str, tuple[str, int]],
    dict[str, OutputQualityScores],
    dict[str, TrajectoryScores],
]:
    """A small, healthy three-source set: two denied, one payable, one partial."""
    decisions = {
        "C01": ("denied", 0),
        "C02": ("payable", 50000),
        "C03": ("partial", 30000),
        "C04": ("denied", 0),
    }
    l1 = {
        "C01": OutputQualityScores(0.22, 0.70, 0.10),
        "C02": OutputQualityScores(0.25, 0.72, 0.09),
        "C03": OutputQualityScores(0.23, 0.71, 0.11),
        "C04": OutputQualityScores(0.21, 0.69, 0.08),
    }
    l2 = {
        "C01": TrajectoryScores(1.0, 1.0, 0),
        "C02": TrajectoryScores(0.5, 1.0, 0),
        "C03": TrajectoryScores(1.0, 1.0, 1),
        "C04": TrajectoryScores(1.0, 1.0, 0),
    }
    return decisions, l1, l2


def test_build_score_sheet_joins_on_claim_id_in_stable_order() -> None:
    decisions, l1, l2 = _healthy_sources()
    sheet = build_score_sheet(decisions, l1, l2)

    assert [r.claim_id for r in sheet] == ["C01", "C02", "C03", "C04"]
    c02 = next(r for r in sheet if r.claim_id == "C02")
    assert c02.decision == "payable"
    assert c02.payable_amount == 50000
    assert c02.tool_precision == 0.5


def test_build_score_sheet_drops_claims_missing_from_any_source() -> None:
    # L1 errored on C03, so it has no L1 row; the joined sheet must drop C03
    # rather than emit a partial row that would bias the means.
    decisions, l1, l2 = _healthy_sources()
    del l1["C03"]
    sheet = build_score_sheet(decisions, l1, l2)
    assert [r.claim_id for r in sheet] == ["C01", "C02", "C04"]


def test_identical_run_flags_no_drift() -> None:
    decisions, l1, l2 = _healthy_sources()
    sheet = build_score_sheet(decisions, l1, l2)

    result = score_regression(sheet, sheet, Thresholds())

    assert result.n_drifted == 0
    assert result.flags == []
    assert result.denial_rate_delta == 0.0


def test_collapsed_quality_and_flipped_decisions_flag_drift() -> None:
    decisions, l1, l2 = _healthy_sources()
    reference = build_score_sheet(decisions, l1, l2)

    # Current run: faithfulness craters across the board, and every claim flips
    # to denied (payouts collapse to zero).
    bad_l1 = {
        cid: OutputQualityScores(0.02, s.answer_relevancy, s.context_precision)
        for cid, s in l1.items()
    }
    denied = {cid: ("denied", 0) for cid in decisions}
    current = build_score_sheet(denied, bad_l1, l2)

    result = score_regression(reference, current, Thresholds())

    assert "faithfulness" in result.flags
    assert "denial_rate" in result.flags
    assert "payable_amount" in result.flags
    assert result.column_deltas["faithfulness"] < 0  # quality dropped
    assert result.denial_rate_delta > 0  # more denials
    assert result.n_drifted == len(result.flags)


def test_subthreshold_movement_does_not_flag() -> None:
    decisions, l1, l2 = _healthy_sources()
    reference = build_score_sheet(decisions, l1, l2)

    # Nudge faithfulness by 0.05 everywhere -- real movement, but under the 0.10
    # threshold. Drift monitoring must not cry wolf on noise this size.
    nudged = {
        cid: OutputQualityScores(s.faithfulness + 0.05, s.answer_relevancy, s.context_precision)
        for cid, s in l1.items()
    }
    current = build_score_sheet(decisions, nudged, l2)

    result = score_regression(reference, current, Thresholds())

    assert "faithfulness" not in result.flags
    assert abs(result.column_deltas["faithfulness"]) < Thresholds().score


def test_empty_baseline_is_well_defined() -> None:
    # Before a baseline exists, comparison against an empty reference must not
    # raise. amount-fraction is guarded on a zero baseline mean, so it cannot
    # flag; the run is simply its own first record.
    decisions, l1, l2 = _healthy_sources()
    current = build_score_sheet(decisions, l1, l2)

    result = score_regression([], current, Thresholds())

    assert "payable_amount" not in result.flags  # zero-baseline guard holds
    assert isinstance(result.n_drifted, int)


def test_score_sheet_round_trips_through_json(tmp_path) -> None:
    decisions, l1, l2 = _healthy_sources()
    sheet = build_score_sheet(decisions, l1, l2)

    path = tmp_path / "baseline_scores.json"
    save_sheet(sheet, path)
    loaded = load_sheet(path)

    assert loaded == sheet
    assert all(isinstance(r, DriftRow) for r in loaded)


def test_run_l3_without_baseline_writes_sheet_and_returns_none(tmp_path) -> None:
    decisions, l1_scores, l2_scores = _healthy_sources()
    l1 = L1Result(
        per_claim=l1_scores,
        mean_faithfulness=0.0,
        mean_answer_relevancy=0.0,
        mean_context_precision=0.0,
    )
    l2 = L2Result(
        per_claim=l2_scores,
        mean_tool_precision=0.0,
        mean_tool_recall=0.0,
        mean_step_overage=0.0,
    )
    current = tmp_path / "latest_scores.json"
    baseline = tmp_path / "baseline_scores.json"

    drift = run_l3(decisions, l1, l2, current, baseline, Thresholds())

    assert drift is None  # no baseline to diff against yet
    assert current.exists()  # but the run recorded itself
    assert len(load_sheet(current)) == len(decisions)


def test_run_l3_with_baseline_returns_drift_result(tmp_path) -> None:
    import shutil

    decisions, l1_scores, l2_scores = _healthy_sources()
    l1 = L1Result(
        per_claim=l1_scores,
        mean_faithfulness=0.0,
        mean_answer_relevancy=0.0,
        mean_context_precision=0.0,
    )
    l2 = L2Result(
        per_claim=l2_scores,
        mean_tool_precision=0.0,
        mean_tool_recall=0.0,
        mean_step_overage=0.0,
    )
    current = tmp_path / "latest_scores.json"
    baseline = tmp_path / "baseline_scores.json"

    run_l3(decisions, l1, l2, current, baseline, Thresholds())  # first run writes current
    shutil.copy(current, baseline)  # promote to baseline

    drift = run_l3(decisions, l1, l2, current, baseline, Thresholds())  # identical re-run
    assert isinstance(drift, DriftResult)
    assert drift.n_drifted == 0
