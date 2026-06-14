"""Tracing seam tests. No Langfuse, no credentials, no network: the seam exists
precisely so the bonding logic is testable offline.

- NullTracer is a true no-op: yields no callbacks and a None trace_id, and its
  score/flush calls do nothing and raise nothing. This is what guarantees the
  agent runs identically when tracing is off.
- run_l1 writes exactly the three L1 metrics onto a run's trace when the run
  carries a trace_id, and writes nothing when it doesn't. A recording fake
  Tracer captures the calls, so the bonding is asserted without a backend.
"""

import asyncio

from eval_engine.agent.schema import Decision
from eval_engine.eval.golden import GoldenLabel
from eval_engine.eval.l1_output_quality import OutputQualityScores, ScoreInput
from eval_engine.eval.runner import AgentOutput, run_l1
from eval_engine.observability.tracing import NullTracer, Score


def test_null_tracer_is_a_noop() -> None:
    tracer = NullTracer()
    with tracer.run_span(name="adjudication", claim_id="C01") as (callbacks, trace_id):
        assert callbacks == []
        assert trace_id is None
    tracer.write_scores("anything", [Score("faithfulness", 0.23)])
    tracer.flush()


class _RecordingTracer:
    """Captures write_scores calls so the test can assert what got bonded."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, list[Score]]] = []
        self.flushed = False

    def run_span(self, name: str, claim_id: str):  # not exercised by run_l1
        raise NotImplementedError

    def write_scores(self, trace_id: str, scores: list[Score]) -> None:
        self.writes.append((trace_id, scores))

    def flush(self) -> None:
        self.flushed = True


class _StubScorer:
    """Returns fixed scores; no Ragas, no network."""

    async def score(self, item: ScoreInput) -> OutputQualityScores:
        return OutputQualityScores(faithfulness=0.23, answer_relevancy=0.71, context_precision=0.09)


def _label() -> GoldenLabel:
    return GoldenLabel(
        claim_id="C01",
        decision=Decision.PAYABLE,
        payable_amount=1000,
        governing_clauses=[],
        expected_tool_calls=[],
        reasoning="r",
        reviewed=True,
    )


def _output(trace_id: str | None) -> AgentOutput:
    return AgentOutput(
        claim_id="C01",
        decision="payable",
        payable_amount=1000,
        reasoning="r",
        retrieved_clause_texts=["clause"],
        trace_id=trace_id,
    )


def test_run_l1_bonds_three_scores_to_the_trace() -> None:
    tracer = _RecordingTracer()
    result = asyncio.run(
        run_l1(
            outputs=[_output("trace-abc")],
            labels={"C01": _label()},
            claim_texts={"C01": "claim text"},
            scorer=_StubScorer(),
            tracer=tracer,
        )
    )

    assert len(tracer.writes) == 1
    trace_id, scores = tracer.writes[0]
    assert trace_id == "trace-abc"
    assert {s.name for s in scores} == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
    }
    assert result.mean_faithfulness == 0.23


def test_run_l1_writes_nothing_without_a_trace_id() -> None:
    tracer = _RecordingTracer()
    asyncio.run(
        run_l1(
            outputs=[_output(None)],
            labels={"C01": _label()},
            claim_texts={"C01": "claim text"},
            scorer=_StubScorer(),
            tracer=tracer,
        )
    )
    assert tracer.writes == []
