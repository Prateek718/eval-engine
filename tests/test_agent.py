"""Tests for the adjudication agent. Hermetic: a scripted stub model drives the
graph (no Gemini), and a stub tool-caller returns canned results (no Qdrant).
Verifies the bounded loop, trajectory tracking, and finalization.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from eval_engine.agent.graph import AgentState, build_graph, initial_state
from eval_engine.agent.policies import load_schedule
from eval_engine.agent.schema import Adjudication, Decision


class ScriptedModel:
    """Emits a predetermined sequence of tool-calling turns, then a final
    structured Adjudication. Lets tests drive the graph deterministically.
    """

    def __init__(self, tool_turns: list[list[dict[str, Any]]], final: Adjudication) -> None:
        self._turns = tool_turns
        self._final = final
        self._i = 0

    def invoke_with_tools(self, messages: list[AnyMessage]) -> AIMessage:
        if self._i < len(self._turns):
            calls = self._turns[self._i]
            self._i += 1
            tool_calls = [
                {"name": c["name"], "args": c["args"], "id": f"call_{j}"}
                for j, c in enumerate(calls)
            ]
            return AIMessage(content="", tool_calls=tool_calls)
        return AIMessage(content="done")  # no tool calls -> route to finalize

    def invoke_structured(self, messages: list[AnyMessage]) -> Adjudication:
        return self._final


def _stub_tool_caller(name: str, args: dict[str, Any]) -> str:
    if name == "retrieve_regulations":
        return json.dumps([{"chunk_id": "2020-S1-CHI-P10-C33", "clause_text": "PED def"}])
    if name == "lookup_policy":
        return json.dumps({"policy_id": args["policy_id"], "sum_insured": 300000})
    return json.dumps({"error": "unknown"})


_FINAL = Adjudication(
    decision=Decision.PARTIAL,
    payable_amount=30000,
    governing_clauses=["2020-S1-CHI-P10-C33"],
    reasoning="capped by sub-limit",
)


def test_agent_runs_tool_loop_then_finalizes() -> None:
    """Two tool turns (retrieve, then lookup), then a structured decision."""
    model = ScriptedModel(
        tool_turns=[
            [{"name": "retrieve_regulations", "args": {"query": "PED"}}],
            [{"name": "lookup_policy", "args": {"policy_id": "POL-A"}}],
        ],
        final=_FINAL,
    )
    app = build_graph(model, _stub_tool_caller, max_tool_calls=2)
    final = AgentState(**app.invoke(initial_state("cataract claim", "POL-A")))

    assert final.result is not None
    assert final.result.decision is Decision.PARTIAL
    assert final.result.payable_amount == 30000
    assert final.tool_calls_made == 2
    assert "2020-S1-CHI-P10-C33" in final.retrieved_chunk_ids


def test_bounded_loop_caps_tool_calls() -> None:
    """A model that always wants tools must still be forced to finalize at the
    cap — this is the loop bound that prevents runaway agents.
    """
    greedy_turns = [[{"name": "retrieve_regulations", "args": {"query": "x"}}]] * 10
    model = ScriptedModel(tool_turns=greedy_turns, final=_FINAL)
    app = build_graph(model, _stub_tool_caller, max_tool_calls=2)
    final = AgentState(**app.invoke(initial_state("claim", "POL-A")))

    assert final.tool_calls_made == 2  # capped, did not run all 10
    assert final.result is not None


def test_agent_can_finalize_without_tools() -> None:
    """A model that calls no tools routes straight to finalize."""
    model = ScriptedModel(tool_turns=[], final=_FINAL)
    app = build_graph(model, _stub_tool_caller, max_tool_calls=2)
    final = AgentState(**app.invoke(initial_state("claim", "POL-C")))

    assert final.tool_calls_made == 0
    assert final.result is not None


def test_schedule_loads_three_policies() -> None:
    schedule = load_schedule()
    assert {p.policy_id for p in schedule.policies} == {"POL-A", "POL-B", "POL-C"}
