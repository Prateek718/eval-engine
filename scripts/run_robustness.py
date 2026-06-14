"""Run the L4 robustness suite: decision stability under input perturbation.

A separate, deliberately heavier pass than scripts/run_eval.py. Each golden
claim is perturbed along every axis and re-run through the agent, so this makes
~(claims x axes) agent calls -- its own entry point, not a flag, because the
cost profile is far higher than the L1/L2 eval.

Run locally with a real GEMINI_API_KEY:

    uv run python scripts/run_robustness.py
"""

from eval_engine.agent.runner import build_agent
from eval_engine.config import get_settings
from eval_engine.eval.golden import claims_by_id, load_golden_set
from eval_engine.eval.runner import run_l4
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

    run_agent = build_agent(settings, tracer=tracer)  # one rig, reused across perturbations
    result = run_l4(claims, labels, run_agent, tracer=tracer)

    print("\n=== L4 robustness (stability under perturbation, per axis) ===")
    print(f"{'axis':<20}{'decision':>10}{'amount':>10}{'n':>6}")
    for axis, a in result.per_axis.items():
        print(f"{axis:<20}{a.decision_stability:>10.2f}{a.amount_stability:>10.2f}{a.n:>6}")
    print(f"\noverall decision_stability: {result.mean_decision_stability:.3f}")
    print(f"overall amount_stability:   {result.mean_amount_stability:.3f}")

    tracer.flush()


if __name__ == "__main__":
    main()
