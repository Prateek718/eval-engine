"""Run L1 output-quality scoring over the golden set with the live agent.

Runs the agent on each claim, maps retrieved chunk ids to clause texts, scores
the three Ragas metrics against the reviewed golden labels, and prints aggregates.

Run locally with a real GEMINI_API_KEY:

    uv run python scripts/run_l1.py
"""

import asyncio

from eval_engine.agent.policies import load_schedule
from eval_engine.agent.runner import build_agent
from eval_engine.agent.tools import lookup_policy
from eval_engine.config import get_settings
from eval_engine.eval import AgentOutput, RagasScorer, load_golden_set, run_l1
from eval_engine.eval.golden import claims_by_id
from eval_engine.ingestion import parse_corpus
from eval_engine.observability.tracing import build_tracer


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set (put it in .env).")

    tracer = build_tracer(
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
    )

    labels = {label.claim_id: label for label in load_golden_set()}
    claims = claims_by_id()
    chunk_text = {c.chunk_id: c.clause_text for c in parse_corpus()}

    run_agent = build_agent(settings, tracer=tracer)  # one rig (Qdrant client, model, graph) reused
    schedule = load_schedule()

    outputs: list[AgentOutput] = []
    failures: list[tuple[str, str]] = []
    for claim_id, claim in claims.items():
        try:
            state = run_agent(claim.claim_text, claim.policy_id, claim_id)
        except Exception as exc:  # transient API errors (timeouts, 5xx) skip the claim
            failures.append((claim_id, type(exc).__name__))
            print(f"{claim_id}: agent run failed ({type(exc).__name__}: {exc}), skipping")
            continue
        if state.result is None:
            failures.append((claim_id, "no result"))
            print(f"{claim_id}: agent produced no result, skipping")
            continue
        outputs.append(
            AgentOutput(
                claim_id=claim_id,
                decision=state.result.decision.value,
                payable_amount=state.result.payable_amount,
                reasoning=state.result.reasoning,
                retrieved_clause_texts=[
                    chunk_text[cid] for cid in state.retrieved_chunk_ids if cid in chunk_text
                ],
                policy_terms_text=lookup_policy(claim.policy_id, schedule),
                trace_id=state.trace_id,
            )
        )
        print(f"scored agent run for {claim_id}: {state.result.decision.value}")

    scorer = RagasScorer(api_key=settings.gemini_api_key)
    claim_texts = {cid: c.claim_text for cid, c in claims.items()}
    result = asyncio.run(run_l1(outputs, labels, claim_texts, scorer, tracer=tracer))

    print("\n=== L1 output quality (means over golden set) ===")
    print(f"faithfulness:      {result.mean_faithfulness:.3f}")
    print(f"answer_relevancy:  {result.mean_answer_relevancy:.3f}")
    print(f"context_precision: {result.mean_context_precision:.3f}")
    print(f"\nscored {len(outputs)} of {len(claims)} claims", end="")
    if failures:
        print(f"; {len(failures)} failed: {failures}")
    else:
        print(".")
    tracer.flush()


if __name__ == "__main__":
    main()
