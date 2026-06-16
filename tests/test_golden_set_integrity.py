"""Golden-set integrity gate. Pure, deterministic, offline: asserts the
committed reference labels obey the invariants the eval system assumes.

These are not quality checks on the agent -- they guard the oracle itself. A
bad edit to the golden set (an invalid decision, a denied claim with a payout,
a clause cited without the regulation lookup that found it, an unreviewed
label) would silently poison every layer that grades against it. CI runs this
under the existing pytest step, so such an edit fails the build before merge.

The invariants hold by design: the golden set is a fixed, human-reviewed oracle,
not an evolving production dataset, so these are true regressions if broken, not
legitimate data drift.
"""

from eval_engine.eval.golden import load_golden_set

_VALID_DECISIONS = {"payable", "partial", "denied"}
_VALID_TOOLS = {"lookup_policy", "retrieve_regulations"}


def test_every_label_is_reviewed() -> None:
    # The methodological-integrity claim: every committed label was
    # human-adjudicated. An unreviewed label in the set breaks the oracle's
    # non-circularity guarantee.
    labels = load_golden_set()
    unreviewed = [label.claim_id for label in labels if not label.reviewed]
    assert unreviewed == [], f"unreviewed labels: {unreviewed}"


def test_claim_ids_are_unique() -> None:
    labels = load_golden_set()
    ids = [label.claim_id for label in labels]
    assert len(ids) == len(set(ids)), "duplicate claim_ids in golden set"


def test_decisions_are_valid() -> None:
    labels = load_golden_set()
    bad = [label.claim_id for label in labels if label.decision.value not in _VALID_DECISIONS]
    assert bad == [], f"invalid decision values: {bad}"


def test_amounts_are_non_negative() -> None:
    labels = load_golden_set()
    negative = [label.claim_id for label in labels if label.payable_amount < 0]
    assert negative == [], f"negative payable_amount: {negative}"


def test_denied_claims_pay_nothing() -> None:
    # A denied claim with a non-zero payout is a contradiction in the oracle.
    labels = load_golden_set()
    contradictions = [
        label.claim_id
        for label in labels
        if label.decision.value == "denied" and label.payable_amount != 0
    ]
    assert contradictions == [], f"denied claims with non-zero amount: {contradictions}"


def test_expected_tool_calls_are_valid_and_non_empty() -> None:
    labels = load_golden_set()
    bad = [
        label.claim_id
        for label in labels
        if not label.expected_tool_calls
        or any(tool not in _VALID_TOOLS for tool in label.expected_tool_calls)
    ]
    assert bad == [], f"empty or invalid expected_tool_calls: {bad}"


def test_clauses_match_regulation_lookups() -> None:
    # Coherence invariant: a label cites governing clauses if and only if its
    # expected trajectory includes the regulation lookup that would surface
    # them. Catches an edit that adds a clause without the tool, or vice versa.
    labels = load_golden_set()
    mismatched = [
        label.claim_id
        for label in labels
        if bool(label.governing_clauses) != ("retrieve_regulations" in label.expected_tool_calls)
    ]
    assert mismatched == [], f"clause/retrieve_regulations mismatch: {mismatched}"
