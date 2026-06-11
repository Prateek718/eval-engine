"""The agent's two tools.

The two-tool design is the agentic spine: deciding a claim needs both the
regulation (is this exclusion permitted? what is the waiting-period rule?) and
the specific policy's terms (does THIS policy name the condition? what is its
sub-limit?). A single retrieval cannot answer both, so the agent must call
tools iteratively — that dependency is what makes it an agent, not a pipeline.

Tools are built from injected dependencies (embedder, Qdrant client, schedule)
rather than globals, so tests can supply stubs.
"""

import json

from qdrant_client import QdrantClient

from eval_engine.agent.policies import PolicySchedule
from eval_engine.indexing.embedder import Embedder


def retrieve_regulations(
    query: str,
    embedder: Embedder,
    client: QdrantClient,
    collection: str,
    top_k: int = 4,
) -> str:
    """Search the IRDAI corpus for clauses relevant to a query. Returns a JSON
    list of {chunk_id, clause_text} so the agent can both reason over and cite
    the regulatory text.
    """
    qv = embedder.embed_query(query)
    hits = client.query_points(collection, query=qv, limit=top_k).points
    results = [
        {"chunk_id": h.payload["chunk_id"], "clause_text": h.payload["clause_text"]}
        for h in hits
        if h.payload is not None
    ]
    return json.dumps(results)


def lookup_policy(policy_id: str, schedule: PolicySchedule) -> str:
    """Return the terms of a specific policy as JSON: sum insured, PED waiting
    period, named specific-disease waiting periods, and sub-limits. Returns an
    error object if the policy id is unknown.
    """
    policy = schedule.get(policy_id)
    if policy is None:
        return json.dumps({"error": f"unknown policy_id: {policy_id}"})
    return policy.model_dump_json()
