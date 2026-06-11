"""Synthetic policy schedule: the policy terms the agent reasons over.

Regulations alone cannot decide a claim amount — that needs policy terms (sum
insured, sub-limits, waiting periods). This schedule supplies them. It is
synthetic and frozen (committed to the repo) so golden-set labels can reference
exact terms by policy id.

Authoring this as JSON deliberately skips the policy-extraction step a real
system would need (PDF schedule -> structured fields). Extraction is a separate
problem from evaluation and adds no eval signal, so it is out of scope here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_POLICIES_PATH = Path("data/policies.json")


class SpecificWaitingPeriod(BaseModel):
    condition: str
    waiting_months: int


class SubLimit(BaseModel):
    procedure: str
    cap_per_eye: int | None = None
    cap: int | None = None


class Policy(BaseModel):
    policy_id: str
    product_name: str
    sum_insured: int
    ped_waiting_months: int
    room_rent_cap_per_day: int | None = None
    specific_waiting_periods: list[SpecificWaitingPeriod] = []
    sub_limits: list[SubLimit] = []

    def specific_wait_for(self, condition: str) -> int | None:
        """Waiting months for a named condition, or None if the policy does not
        name it. 'Not named' is decision-relevant: a specific-disease wait only
        applies if the policy lists that disease.
        """
        for swp in self.specific_waiting_periods:
            if swp.condition.lower() == condition.lower():
                return swp.waiting_months
        return None

    def sub_limit_for(self, procedure: str) -> SubLimit | None:
        for sl in self.sub_limits:
            if sl.procedure.lower() == procedure.lower():
                return sl
        return None


class PolicySchedule(BaseModel):
    schema_version: str
    currency: str
    policies: list[Policy]

    def get(self, policy_id: str) -> Policy | None:
        for p in self.policies:
            if p.policy_id == policy_id:
                return p
        return None


@lru_cache(maxsize=1)
def load_schedule(path: str | None = None) -> PolicySchedule:
    """Load and validate the frozen schedule. Cached: the artifact is immutable
    at runtime, so it is parsed once.
    """
    p = Path(path) if path else _POLICIES_PATH
    return PolicySchedule.model_validate_json(p.read_text(encoding="utf-8"))
