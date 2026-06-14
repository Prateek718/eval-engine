"""L4 robustness tests. Offline: perturbation is deterministic, the agent is
stubbed, no Gemini and no network.

Two halves:
- perturbation fact-preservation: across every axis and many seeds, the rupee
  amounts and policy ids in a claim survive untouched. This is the guarantee
  that lets the golden label stay a valid oracle for the perturbed input.
- run_l4 scoring + bonding: a stub agent that is robust on most axes but
  misreads reformatted amounts produces decision_stability 1.0 with
  amount_stability < 1.0 on number_reformat -- proving the two are reported
  separately -- and the per-axis scores bond to traces.
"""

import re

from eval_engine.agent.graph import AgentState
from eval_engine.agent.schema import Adjudication, Decision
from eval_engine.eval.golden import Claim, GoldenLabel
from eval_engine.eval.l4_robustness import AXES, perturb
from eval_engine.eval.runner import run_l4
from eval_engine.observability.tracing import Score

_AMOUNT = re.compile(r"\b(?:Rs\.?|INR)\s*([\d,]+)", re.IGNORECASE)
_POLICY = re.compile(r"POL-\w+", re.IGNORECASE)


def _amounts(text: str) -> set[int]:
    return {int(x.replace(",", "")) for x in _AMOUNT.findall(text)}


def _policies(text: str) -> set[str]:
    return {p.upper() for p in _POLICY.findall(text)}


_CLAIM = (
    "Cataract surgery on the right eye. Hospital billed Rs 50,000. Policy POL-A active for 2 years."
)


def test_perturbation_preserves_amounts_and_policy_ids() -> None:
    orig_amt = _amounts(_CLAIM)
    orig_pol = _policies(_CLAIM)
    for axis in AXES:
        for seed in range(25):
            p = perturb(_CLAIM, axis, seed)
            assert _amounts(p) == orig_amt, f"{axis}/{seed} altered amount: {p}"
            assert _policies(p) == orig_pol, f"{axis}/{seed} altered policy id: {p}"


def test_perturbation_is_deterministic() -> None:
    assert perturb(_CLAIM, "typo", 7) == perturb(_CLAIM, "typo", 7)


def test_number_reformat_changes_surface_but_not_value() -> None:
    surfaces = {perturb(_CLAIM, "number_reformat", s) for s in range(10)}
    assert len(surfaces) > 1
    for s in surfaces:
        assert _amounts(s) == _amounts(_CLAIM)


class _RecordingTracer:
    def __init__(self) -> None:
        self.writes: list[tuple[str, list[Score]]] = []

    def write_scores(self, trace_id: str, scores: list[Score]) -> None:
        self.writes.append((trace_id, scores))


def _claims_and_labels() -> tuple[dict[str, Claim], dict[str, GoldenLabel]]:
    claims = {
        "C01": Claim(
            claim_id="C01",
            policy_id="POL-A",
            claim_text="Cataract surgery. Hospital billed Rs 50,000. Policy active 2 years.",
        ),
    }
    labels = {
        "C01": GoldenLabel(
            claim_id="C01",
            decision=Decision.PARTIAL,
            payable_amount=30000,
            governing_clauses=[],
            expected_tool_calls=["lookup_policy"],
            reasoning="cataract sub-limit",
            reviewed=True,
        )
    }
    return claims, labels


def _agent_factory(brittle_amount_on_reformat: bool):
    counter = {"n": 0}

    def agent(text: str, policy_id: str, claim_id: str) -> AgentState:
        counter["n"] += 1
        amount = 30000
        if brittle_amount_on_reformat and ("INR" in text or re.search(r"Rs\.?\s*\d{5,}", text)):
            amount = 30001  # decision holds, amount drifts under reformatting
        result = Adjudication(
            decision=Decision.PARTIAL,
            payable_amount=amount,
            governing_clauses=[],
            reasoning="stub",
        )
        return AgentState(result=result, trace_id=f"t{counter['n']}")

    return agent


def test_run_l4_reports_decision_and_amount_separately() -> None:
    claims, labels = _claims_and_labels()
    tracer = _RecordingTracer()
    result = run_l4(claims, labels, _agent_factory(brittle_amount_on_reformat=True), tracer=tracer)

    assert result.per_axis["typo"].decision_stability == 1.0
    assert result.per_axis["typo"].amount_stability == 1.0
    assert result.per_axis["number_reformat"].decision_stability == 1.0
    assert result.per_axis["number_reformat"].amount_stability == 0.0
    assert result.mean_decision_stability == 1.0
    assert result.mean_amount_stability < 1.0


def test_run_l4_bonds_two_scores_per_axis_to_traces() -> None:
    claims, labels = _claims_and_labels()
    tracer = _RecordingTracer()
    run_l4(claims, labels, _agent_factory(brittle_amount_on_reformat=False), tracer=tracer)

    assert len(tracer.writes) == len(AXES)
    for _trace_id, scores in tracer.writes:
        names = {s.name for s in scores}
        assert any(n.endswith("_decision") for n in names)
        assert any(n.endswith("_amount") for n in names)


def test_run_l4_skips_untraced_runs() -> None:
    claims, labels = _claims_and_labels()
    tracer = _RecordingTracer()

    def untraced_agent(text: str, policy_id: str, claim_id: str) -> AgentState:
        result = Adjudication(
            decision=Decision.PARTIAL, payable_amount=30000, governing_clauses=[], reasoning="stub"
        )
        return AgentState(result=result, trace_id=None)

    run_l4(claims, labels, untraced_agent, tracer=tracer)
    assert tracer.writes == []
