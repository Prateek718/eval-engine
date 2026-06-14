"""L2: trajectory scoring of the agent's tool use.

L1 grades the answer; L2 grades the path taken to it. The reference is each
golden label's ``expected_tool_calls`` -- the SET of tools a claim warrants:
``{lookup_policy}`` for claims answerable from the policy schedule alone, or
``{lookup_policy, retrieve_regulations}`` for claims that genuinely turn on a
regulatory clause (PED, exclusions, fraud).

Three readings, deliberately separate:

- tool_recall: of the tools the claim needed, how many were used. Low recall is
  UNDER-use -- the agent skipped a lookup it needed (a correctness risk).
- tool_precision: of the tools the agent used, how many were warranted. Low
  precision is OVER-retrieval -- tool work the claim did not justify. This is
  the behaviour the whole eval exists to expose, isolated as its own number.
- step_overage: tool calls made beyond the claim's expected ceiling (1 for a
  policy-only claim, 2 when a regulation lookup is warranted). Catches looping
  inefficiency even when the tool SET is right (e.g. retrieving twice).

Exact-match was rejected on purpose: it collapses over- and under-use into one
pass/fail, and over-retrieval is precisely what this layer must measure.
Precision and recall are computed over the tool SET (names, de-duplicated); a
tool called twice is one tool used, while the repeat surfaces as step_overage.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class TrajectoryScores:
    """Per-claim trajectory reading."""

    tool_precision: float  # warranted tools / tools used  (low = over-retrieval)
    tool_recall: float  # tools used that were needed / tools needed (low = under-use)
    step_overage: int  # tool calls beyond the expected ceiling; 0 is ideal


@dataclass
class TrajectoryInput:
    """One run's trajectory framed for scoring."""

    claim_id: str
    tools_used: list[str]  # tool names the agent called, in order, with repeats
    tool_calls_made: int  # total calls incl. repeats (the efficiency signal)
    expected_tool_calls: list[str]  # golden reference: the warranted tool set
    trace_id: str | None = None  # the run's Langfuse trace, for bonding scores to it


def score_trajectory(item: TrajectoryInput) -> TrajectoryScores:
    used = set(item.tools_used)
    expected = set(item.expected_tool_calls)

    # Precision/recall over the tool set. Empty-set edge cases resolve to the
    # only sensible reading: if the agent used no tools, precision is vacuously
    # 1.0 (it raised no unwarranted calls); if nothing was expected, recall is
    # vacuously 1.0 (nothing to miss).
    correct = used & expected
    tool_precision = len(correct) / len(used) if used else 1.0
    tool_recall = len(correct) / len(expected) if expected else 1.0

    # Efficiency: the ceiling is the count of distinct warranted tools (1 or 2
    # here). Anything beyond it -- a re-call, an extra exploratory loop -- is
    # overage. Never negative: using fewer calls than the ceiling is not a
    # penalty (that is under-use, already caught by recall).
    ceiling = len(expected)
    step_overage = max(0, item.tool_calls_made - ceiling)

    return TrajectoryScores(
        tool_precision=tool_precision,
        tool_recall=tool_recall,
        step_overage=step_overage,
    )


@dataclass
class L2Result:
    per_claim: dict[str, TrajectoryScores]
    mean_tool_precision: float
    mean_tool_recall: float
    mean_step_overage: float


def aggregate(per_claim: dict[str, TrajectoryScores]) -> L2Result:
    def mean(attr: str) -> float:
        vals = [getattr(s, attr) for s in per_claim.values()]
        return statistics.fmean(vals) if vals else 0.0

    return L2Result(
        per_claim=per_claim,
        mean_tool_precision=mean("tool_precision"),
        mean_tool_recall=mean("tool_recall"),
        mean_step_overage=mean("step_overage"),
    )
