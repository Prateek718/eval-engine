"""The agent's output contract: a structured claim-adjudication decision.

Each field feeds a different evaluation layer:
- decision        -> the categorical call; L1 checks it is grounded, the golden
                     set checks it is correct.
- payable_amount  -> objective, exactly gradable; wrong reasoning surfaces as a
                     wrong number.
- governing_clauses -> the chunk ids the decision rests on; the bridge to L2
                     trajectory eval (cited vs retrieved vs expected).
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Decision(StrEnum):
    PAYABLE = "payable"
    DENIED = "denied"
    PARTIAL = "partial"


class Adjudication(BaseModel):
    """The agent's final answer for a claim."""

    decision: Decision = Field(
        description="Whether the claim is payable, denied, or partially payable"
    )
    payable_amount: int = Field(description="Amount payable in INR; 0 if denied")
    governing_clauses: list[str] = Field(
        default_factory=list,
        description="chunk_ids of the regulatory clauses the decision rests on",
    )
    reasoning: str = Field(
        description="Brief justification tying the decision to the clauses and policy terms"
    )
