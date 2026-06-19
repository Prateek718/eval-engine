"""Run the eval layers over the golden set with the live agent.

One agent pass per claim feeds every layer, so all scores describe the same
(non-deterministic) execution and bond to one trace:

- L1 output quality (Ragas): faithfulness, answer relevancy, context precision.
- L2 trajectory: tool precision/recall and step overage against the golden
  ``expected_tool_calls``.

The whole session is wrapped in one MLflow run, logging the aggregate means as
metrics and the model id, claim count and git sha as params, so scores can be
compared across runs over time.

Run locally with a real GEMINI_API_KEY:

    uv run python scripts/run_eval.py
"""

import asyncio
import subprocess

from eval_engine.agent.policies import load_schedule
from eval_engine.agent.runner import build_agent
from eval_engine.agent.tools import lookup_policy
from eval_engine.config import get_settings
from eval_engine.eval import AgentOutput, RagasScorer, load_golden_set, run_l1
from eval_engine.eval.golden import claims_by_id
from eval_engine.eval.l2_trajectory import TrajectoryInput
from eval_engine.eval.l3_regression import Thresholds
from eval_engine.eval.runner import run_l2, run_l3
from eval_engine.ingestion import parse_corpus
from eval_engine.observability.metrics import build_metrics_publisher
from eval_engine.observability.tracing import build_tracer
from eval_engine.observability.tracking import build_run_tracker


def _git_sha() -> str:
    """Short commit sha of the current code, or 'unknown' outside a repo.

    Ties each tracked run to the code version that produced it; degrades
    gracefully so a missing git never breaks an eval.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set (put it in .env).")

    tracer = build_tracer(
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
    )
    run_tracker = build_run_tracker(settings.mlflow_tracking_uri, settings.mlflow_experiment)
    metrics_publisher = build_metrics_publisher(settings.pushgateway_url)

    labels = {label.claim_id: label for label in load_golden_set()}
    claims = claims_by_id()
    chunk_text = {c.chunk_id: c.clause_text for c in parse_corpus()}

    run_agent = build_agent(settings, tracer=tracer)  # one rig (Qdrant client, model, graph) reused
    schedule = load_schedule()

    with run_tracker.session(
        run_name=f"eval-{len(claims)}-claims",
        params={
            "agent_model": settings.agent_model,
            "n_claims": len(claims),
            "git_sha": _git_sha(),
        },
    ):
        outputs: list[AgentOutput] = []
        trajectories: list[TrajectoryInput] = []
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
            # Both inputs are built from the same state and the same trace_id, so L1
            # and L2 grade one execution and their scores land on one trace.
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
            trajectories.append(
                TrajectoryInput(
                    claim_id=claim_id,
                    tools_used=state.tools_used,
                    tool_calls_made=state.tool_calls_made,
                    expected_tool_calls=labels[claim_id].expected_tool_calls,
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

        l2 = run_l2(trajectories, tracer=tracer)
        print("\n=== L2 trajectory (means over golden set) ===")
        print(f"tool_precision: {l2.mean_tool_precision:.3f}")
        print(f"tool_recall:    {l2.mean_tool_recall:.3f}")
        print(f"step_overage:   {l2.mean_step_overage:.3f}")

        # Decision accuracy: the headline business metric -- fraction of scored
        # claims whose decision matches the golden label. Computed over the
        # claims that produced a result (failures are excluded from the rate, not
        # counted as wrong, so a transient API error doesn't depress accuracy).
        correct = sum(1 for o in outputs if o.decision == labels[o.claim_id].decision.value)
        decision_accuracy = correct / len(outputs) if outputs else 0.0
        print("\n=== Decision accuracy ===")
        print(f"decision_accuracy: {decision_accuracy:.3f} ({correct}/{len(outputs)})")

        metrics = {
            "faithfulness": result.mean_faithfulness,
            "answer_relevancy": result.mean_answer_relevancy,
            "context_precision": result.mean_context_precision,
            "tool_precision": l2.mean_tool_precision,
            "tool_recall": l2.mean_tool_recall,
            "step_overage": l2.mean_step_overage,
            "decision_accuracy": decision_accuracy,
        }

        # L3: build this run's sheet, persist it, and diff against the frozen
        # baseline. decisions are unpacked from the run's outputs so L3 needs no
        # AgentOutput import. No baseline yet (first run) -> drift is None.
        decisions = {o.claim_id: (o.decision, o.payable_amount) for o in outputs}
        thresholds = Thresholds(
            score=settings.drift_threshold_score,
            step_overage=settings.drift_threshold_step_overage,
            amount_fraction=settings.drift_threshold_amount_fraction,
            denial_rate=settings.drift_threshold_denial_rate,
        )
        drift = run_l3(
            decisions,
            result,
            l2,
            settings.current_sheet_path,
            settings.baseline_sheet_path,
            thresholds,
        )
        print("\n=== L3 regression (vs frozen baseline) ===")
        if drift is None:
            print("no baseline yet; recorded this run's sheet as a candidate.")
        else:
            print(f"drifted signals: {drift.n_drifted} {drift.flags}")
            for col, d in drift.column_deltas.items():
                metrics[f"regression_delta_{col}"] = d
            metrics["regression_denial_rate_delta"] = drift.denial_rate_delta
            metrics["regression_n_drifted"] = float(drift.n_drifted)

        run_tracker.log_metrics(metrics)

        # Publish the same session metrics to the Pushgateway, plus L3 drift
        # flags as per-signal gauges (no-op when no gateway is configured).
        flags = {name: True for name in drift.flags} if drift else {}
        metrics_publisher.publish(metrics, flags)

        print(f"\nscored {len(outputs)} of {len(claims)} claims", end="")
        if failures:
            print(f"; {len(failures)} failed: {failures}")
        else:
            print(".")

    tracer.flush()


if __name__ == "__main__":
    main()
