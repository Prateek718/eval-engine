"""Draft golden-set labels for review.

Sends each claim, its policy terms, and the full corpus to a model in a single
call, requesting a structured label (decision, amount, governing clause ids,
expected tool calls) with citations drawn from the supplied clause ids.

The drafter uses gemini-3.5-flash and places the entire corpus (~8k tokens) in
context rather than retrieving, so it does not depend on the agent's retrieval
pipeline. Output is written to data/golden_set.json with reviewed=false; labels
require manual verification before use.

Run locally with a real GEMINI_API_KEY:

    uv run python scripts/draft_golden_set.py
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from eval_engine.agent.policies import load_schedule
from eval_engine.agent.schema import Decision
from eval_engine.config import get_settings
from eval_engine.ingestion import parse_corpus

_CLAIMS_PATH = Path("data/claims.json")
_OUTPUT_PATH = Path("data/golden_set.json")
_DRAFTER_MODEL = "gemini-3.5-flash"


class DraftLabel(BaseModel):
    """A drafted golden label. Mirrors the agent's Adjudication plus the
    expected trajectory, so labels are directly comparable to agent output.
    """

    claim_id: str
    decision: Decision
    payable_amount: int
    governing_clauses: list[str]  # chunk_ids from the corpus
    expected_tool_calls: list[str]  # e.g. ["retrieve_regulations", "lookup_policy"]
    reasoning: str
    reviewed: bool = False  # set true by hand after verification


def _corpus_block() -> str:
    """Every clause as 'chunk_id: clause_text', for the drafter to cite from."""
    lines = [f"{c.chunk_id}: {c.clause_text}" for c in parse_corpus()]
    return "\n".join(lines)


def _prompt(corpus: str, policy_json: str, claim_text: str, policy_id: str) -> str:
    return f"""You are drafting a reference answer key for a health-insurance claim
adjudicator. Decide the correct adjudication for the claim below, citing the
exact regulatory clauses (by chunk_id) that govern it.

REGULATIONS (cite governing_clauses by these chunk_ids):
{corpus}

POLICY {policy_id}:
{policy_json}

CLAIM:
{claim_text}

Rules:
- decision is exactly one of: payable, denied, partial.
  - payable: the full billed amount is paid.
  - partial: a smaller amount than billed is paid because a sub-limit or the sum
    insured caps it. Any reduction below the bill makes the decision partial.
  - denied: nothing is paid (payable_amount 0).
- A specific-disease waiting period applies ONLY if the policy names that disease.
- payable_amount is capped by any applicable sub-limit, then by sum insured; 0 if denied.
- If the policy_id is not a real policy in the schedule, the claim cannot be
  adjudicated: decision denied, payable_amount 0.
- governing_clauses: cite a regulation chunk_id ONLY when the decision turns on
  that clause's substance. Use these and only these grounds:
    * a denial caused by a regulatory exclusion or rule (e.g. congenital anomaly,
      not-medically-necessary, OPD-not-hospitalisation, established fraud);
    * a PED decision (the pre-existing-disease definition clause);
    * an accident carve-out from a waiting period.
  Do NOT cite a clause merely because it mentions or defines the topic. In
  particular:
    * a decision capped by a policy SUB-LIMIT or SUM INSURED is a policy term,
      not a regulation -> governing_clauses MUST be empty;
    * a decision driven by a NAMED or UNNAMED specific waiting period is a policy
      term -> empty;
    * a plain payable claim with no restriction needs NO citation -> empty;
    * do NOT cite definitional clauses (Medical Expenses, Surgery, Hospitalization,
      Illness) just to establish that something is covered;
    * do NOT cite claim-settlement/CRC procedure clauses.
  An empty governing_clauses list is correct and expected for any policy-term or
  no-restriction decision. Most payable and most sub-limit/SI partial decisions
  will have an empty list.
- expected_tool_calls: which of retrieve_regulations, lookup_policy a correct agent would need.

Output the adjudication as structured data."""


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set (put it in .env).")

    from langchain_google_genai import ChatGoogleGenerativeAI

    claims = json.loads(_CLAIMS_PATH.read_text(encoding="utf-8"))["claims"]
    schedule = load_schedule()
    corpus = _corpus_block()

    model = ChatGoogleGenerativeAI(
        model=_DRAFTER_MODEL, google_api_key=settings.gemini_api_key, temperature=0
    )
    structured = model.with_structured_output(DraftLabel, method="json_schema")

    drafts: list[dict[str, Any]] = []
    for claim in claims:
        policy = schedule.get(claim["policy_id"])
        policy_json = policy.model_dump_json() if policy else "{}"
        prompt = _prompt(corpus, policy_json, claim["claim_text"], claim["policy_id"])
        label = structured.invoke(prompt)
        assert isinstance(label, DraftLabel)
        label.claim_id = claim["claim_id"]  # trust our id, not the model's
        drafts.append(label.model_dump(mode="json"))
        print(f"drafted {claim['claim_id']}: {label.decision} / {label.payable_amount}")

    _OUTPUT_PATH.write_text(
        json.dumps({"schema_version": "1", "labels": drafts}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(drafts)} draft labels to {_OUTPUT_PATH}. ALL require hand review.")


if __name__ == "__main__":
    main()
