"""Real-side wiring: the Gemini chat model adapter and the agent runner.

Like the embedder, the LangChain/Gemini SDK is imported lazily so the module
(and the stub-based tests) load without credentials or the SDK present.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage
from qdrant_client import QdrantClient

from eval_engine.agent.graph import AgentState, build_graph, initial_state
from eval_engine.agent.policies import PolicySchedule, load_schedule
from eval_engine.agent.schema import Adjudication
from eval_engine.agent.tools import lookup_policy, retrieve_regulations
from eval_engine.config import Settings
from eval_engine.indexing.embedder import Embedder, GeminiEmbedder
from eval_engine.indexing.indexer import make_client

# Tool schemas advertised to the model. Kept minimal and explicit.
_TOOL_SCHEMAS = [
    {
        "name": "retrieve_regulations",
        "description": "Search IRDAI regulations for clauses relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "lookup_policy",
        "description": "Get the terms of a specific policy by its policy_id (e.g. POL-A).",
        "parameters": {
            "type": "object",
            "properties": {"policy_id": {"type": "string"}},
            "required": ["policy_id"],
        },
    },
]


class GeminiChatModel:
    """Adapts ChatGoogleGenerativeAI to the graph's ChatModel protocol."""

    def __init__(self, api_key: str, model: str) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        base = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
        self._with_tools = base.bind_tools(_TOOL_SCHEMAS)
        self._structured = base.with_structured_output(Adjudication, method="json_schema")

    def invoke_with_tools(self, messages: list[AnyMessage]) -> AIMessage:
        result = self._with_tools.invoke(messages)
        assert isinstance(result, AIMessage)
        return result

    def invoke_structured(self, messages: list[AnyMessage]) -> Adjudication:
        instruction = "Based on the conversation, output the final adjudication as structured data."
        result = self._structured.invoke([*messages, _human(instruction)])
        assert isinstance(result, Adjudication)
        return result


def _human(text: str) -> AnyMessage:
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=text)


def make_tool_caller(
    embedder: Embedder,
    client: QdrantClient,
    collection: str,
    schedule: PolicySchedule,
) -> Any:
    """Build the dispatcher the graph uses to execute a named tool call."""

    def call(name: str, args: dict[str, Any]) -> str:
        if name == "retrieve_regulations":
            return retrieve_regulations(args["query"], embedder, client, collection)
        if name == "lookup_policy":
            return lookup_policy(args["policy_id"], schedule)
        return json.dumps({"error": f"unknown tool: {name}"})

    return call


def adjudicate(claim: str, policy_id: str, settings: Settings) -> AgentState:
    """Run the agent on one claim with real Gemini + Qdrant. Returns the final
    state (decision in .result, trajectory in .tool_calls_made / .retrieved_chunk_ids).
    """
    embedder: Embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    client = make_client(settings)
    schedule = load_schedule()
    model = GeminiChatModel(api_key=settings.gemini_api_key, model="gemini-2.5-flash")
    tool_caller = make_tool_caller(embedder, client, settings.qdrant_collection, schedule)

    app = build_graph(model, tool_caller)
    final = app.invoke(initial_state(claim, policy_id))
    return AgentState(**final)
