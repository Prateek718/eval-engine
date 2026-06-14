"""Minimal claim-adjudication agent, built as an explicit StateGraph.

Deliberately minimal in capability (two tools, a bounded loop), but explicit in
structure so the trajectory is first-class state the L2 eval layer can read
directly rather than reconstruct from traces.

Graph shape:

    call_model --(tool calls?)--> call_tools --> call_model   (loop, bounded)
        |
        (no tool calls / cap reached)
        v
      finalize  --> END   (one structured-output call -> Adjudication)

The model/tool loop and the final structured-output call are separated because
structured output and tool-calling should not share one call; the explicit
graph makes that separation natural.
"""

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from eval_engine.agent.schema import Adjudication

_SYSTEM_PROMPT = """You are a health-insurance claim adjudicator for Indian policies.
Decide whether a claim is payable, denied, or partially payable, and the amount.

You have two tools:
- retrieve_regulations(query): search IRDAI regulations for governing clauses.
- lookup_policy(policy_id): get the specific policy's terms.

A correct decision generally requires BOTH: the regulation that governs the
situation AND the specific policy's terms (a specific-disease waiting period
applies only if the policy names that disease; amounts depend on sub-limits and
sum insured). Call tools as needed, then state your decision.
Keep tool use minimal."""


@dataclass
class AgentState:
    """Typed state threaded through the graph. The trajectory fields exist for
    the L2 eval layer: they make tool usage inspectable without parsing traces.
    """

    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    tool_calls_made: int = 0
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    result: Adjudication | None = None
    trace_id: str | None = None  # the Langfuse trace this run logged under, if traced


class ToolCaller(Protocol):
    """Executes a single named tool call and returns its string result."""

    def __call__(self, name: str, args: dict[str, Any]) -> str: ...


class ChatModel(Protocol):
    """The LLM seam. bind_tools for the loop; structured for the final answer.
    Both real (ChatGoogleGenerativeAI) and stub satisfy this structurally.
    """

    def invoke_with_tools(self, messages: list[AnyMessage]) -> AIMessage: ...

    def invoke_structured(self, messages: list[AnyMessage]) -> Adjudication: ...


def build_graph(
    model: ChatModel,
    tool_caller: ToolCaller,
    max_tool_calls: int = 2,
) -> Any:
    """Compile the adjudication graph. max_tool_calls bounds the loop (your
    '1-2 tool calls' cap); reaching it forces finalization.
    """

    def call_model(state: AgentState) -> dict[str, Any]:
        ai = model.invoke_with_tools(state.messages)
        return {"messages": [ai]}

    def call_tools(state: AgentState) -> dict[str, Any]:
        last = state.messages[-1]
        assert isinstance(last, AIMessage)
        tool_messages: list[AnyMessage] = []
        new_chunk_ids: list[str] = []
        for call in last.tool_calls:
            result = tool_caller(call["name"], call["args"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            if call["name"] == "retrieve_regulations":
                new_chunk_ids.extend(_chunk_ids_from(result))
        return {
            "messages": tool_messages,
            "tool_calls_made": state.tool_calls_made + len(last.tool_calls),
            "retrieved_chunk_ids": state.retrieved_chunk_ids + new_chunk_ids,
        }

    def finalize(state: AgentState) -> dict[str, Any]:
        result = model.invoke_structured(state.messages)
        return {"result": result}

    def route_after_model(state: AgentState) -> str:
        last = state.messages[-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        if wants_tools and state.tool_calls_made < max_tool_calls:
            return "call_tools"
        return "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", call_tools)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model", route_after_model, {"call_tools": "call_tools", "finalize": "finalize"}
    )
    graph.add_edge("call_tools", "call_model")
    graph.add_edge("finalize", END)
    return graph.compile()


def _chunk_ids_from(tool_result: str) -> list[str]:
    """Extract chunk_ids from a retrieve_regulations JSON result, defensively."""
    try:
        rows = json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    return [r["chunk_id"] for r in rows if isinstance(r, dict) and "chunk_id" in r]


def initial_state(claim: str, policy_id: str) -> AgentState:
    """Seed the graph with the system prompt and the claim."""
    user = f"Policy: {policy_id}\nClaim: {claim}"
    return AgentState(messages=[SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user)])
