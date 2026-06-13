"""Golden set: the reviewed reference labels the eval layers grade against.

Mirrors the drafter's output schema. Loaded once and treated as immutable.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from eval_engine.agent.schema import Decision

_GOLDEN_PATH = Path("data/golden_set.json")
_CLAIMS_PATH = Path("data/claims.json")


class GoldenLabel(BaseModel):
    claim_id: str
    decision: Decision
    payable_amount: int
    governing_clauses: list[str]
    expected_tool_calls: list[str]
    reasoning: str
    reviewed: bool


class Claim(BaseModel):
    claim_id: str
    policy_id: str
    claim_text: str
    mechanism: str = ""


@lru_cache(maxsize=1)
def load_golden_set(path: str | None = None) -> list[GoldenLabel]:
    p = Path(path) if path else _GOLDEN_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [GoldenLabel(**row) for row in raw["labels"]]


@lru_cache(maxsize=1)
def load_claims(path: str | None = None) -> list[Claim]:
    p = Path(path) if path else _CLAIMS_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Claim(**row) for row in raw["claims"]]


def claims_by_id(path: str | None = None) -> dict[str, Claim]:
    return {c.claim_id: c for c in load_claims(path)}
