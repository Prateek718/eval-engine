"""L2 trajectory scoring tests. Pure and offline: no Gemini, no network.

Two halves:
- score_trajectory: the metric over the two golden patterns (policy-only and
  policy+regulation) plus the edges -- over-retrieval, under-use, repeat calls,
  and empty sets. Over- and under-use must read as DIFFERENT numbers; that
  separation is the reason precision/recall was chosen over exact-match.
- run_l2: bonds three trajectory scores to a traced run and writes nothing for
  an untraced one. A recording fake tracer captures the calls, so the bonding
  is asserted without a backend.
"""

from eval_engine.eval.l2_trajectory import TrajectoryInput, score_trajectory
from eval_engine.eval.runner import run_l2
from eval_engine.observability.tracing import Score


def _ti(
    tools_used: list[str],
    tool_calls_made: int,
    expected: list[str],
    *,
    claim_id: str = "C01",
    trace_id: str | None = None,
) -> TrajectoryInput:
    return TrajectoryInput(
        claim_id=claim_id,
        tools_used=tools_used,
        tool_calls_made=tool_calls_made,
        expected_tool_calls=expected,
        trace_id=trace_id,
    )


def test_over_retrieval_drops_precision_not_recall() -> None:
    # policy-only claim, agent also called retrieve_regulations
    s = score_trajectory(_ti(["lookup_policy", "retrieve_regulations"], 2, ["lookup_policy"]))
    assert s.tool_precision == 0.5  # half the tool work was unwarranted
    assert s.tool_recall == 1.0  # nothing needed was missed
    assert s.step_overage == 1  # one call beyond the ceiling of 1


def test_under_use_drops_recall_not_precision() -> None:
    # regulation claim, agent only checked policy
    s = score_trajectory(_ti(["lookup_policy"], 1, ["lookup_policy", "retrieve_regulations"]))
    assert s.tool_precision == 1.0  # what it used was warranted
    assert s.tool_recall == 0.5  # missed the needed regulation lookup
    assert s.step_overage == 0  # under the ceiling is not penalised here


def test_ideal_run_scores_perfect() -> None:
    s = score_trajectory(
        _ti(
            ["lookup_policy", "retrieve_regulations"],
            2,
            ["lookup_policy", "retrieve_regulations"],
        )
    )
    assert s.tool_precision == 1.0
    assert s.tool_recall == 1.0
    assert s.step_overage == 0


def test_repeat_call_keeps_set_metrics_but_flags_overage() -> None:
    # right tool SET, but retrieve_regulations called twice
    s = score_trajectory(
        _ti(
            ["lookup_policy", "retrieve_regulations", "retrieve_regulations"],
            3,
            ["lookup_policy", "retrieve_regulations"],
        )
    )
    assert s.tool_precision == 1.0  # de-duped set is exactly right
    assert s.tool_recall == 1.0
    assert s.step_overage == 1  # the wasted repeat surfaces only here


def test_empty_sets_resolve_vacuously() -> None:
    # no tools used, none expected: nothing unwarranted, nothing missed
    s = score_trajectory(_ti([], 0, []))
    assert s.tool_precision == 1.0
    assert s.tool_recall == 1.0
    assert s.step_overage == 0


class _RecordingTracer:
    def __init__(self) -> None:
        self.writes: list[tuple[str, list[Score]]] = []

    def write_scores(self, trace_id: str, scores: list[Score]) -> None:
        self.writes.append((trace_id, scores))


def test_run_l2_bonds_three_scores_to_the_trace() -> None:
    tracer = _RecordingTracer()
    result = run_l2(
        [_ti(["lookup_policy", "retrieve_regulations"], 2, ["lookup_policy"], trace_id="t-1")],
        tracer=tracer,
    )

    assert len(tracer.writes) == 1
    trace_id, scores = tracer.writes[0]
    assert trace_id == "t-1"
    assert {s.name for s in scores} == {"tool_precision", "tool_recall", "step_overage"}
    assert result.mean_tool_precision == 0.5


def test_run_l2_writes_nothing_without_a_trace_id() -> None:
    tracer = _RecordingTracer()
    run_l2([_ti(["lookup_policy"], 1, ["lookup_policy"], trace_id=None)], tracer=tracer)
    assert tracer.writes == []
