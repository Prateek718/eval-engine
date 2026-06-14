"""L4: robustness scoring -- decision stability under input perturbation.

The oracle is each golden label: a perturbed claim must still produce the
label's decision AND payable_amount. Perturbations are deterministic (seeded)
and fact-preserving BY CONSTRUCTION -- each one is defined to never alter a
decision-critical token (amount, policy id, duration). That structural
guarantee is what makes the golden label a trustworthy oracle for the
perturbed input, with no generation step to distrust.

Four axes, each a pure ``str -> str`` keyed by a per-claim seed:
- typo: char-level noise on ordinary words; fact tokens are skipped.
- whitespace_case: spacing/casing noise; policy ids are skipped from casing.
- irrelevant_suffix: appends one decision-neutral sentence.
- number_reformat: re-renders rupee amounts in an equivalent form, asserting
  the parsed value is unchanged. This is the axis that genuinely stresses the
  agent's amount parsing under the exact-amount bar.

LLM-paraphrase perturbation is deliberately deferred: it needs constrained
generation, token-validation and a frozen, human-reviewed artifact to keep the
oracle trustworthy. Deterministic axes get fact-preservation for free.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

# A token is "fact-critical" -- never corrupted -- if it is a rupee amount, a
# policy id, or a bare number/duration. Matched generously on purpose: when in
# doubt, preserve.
_NUMBER = re.compile(r"\d")
_POLICY_ID = re.compile(r"\bPOL-\w+\b", re.IGNORECASE)
_DURATION_WORDS = {"month", "months", "year", "years", "day", "days"}

_SUFFIXES = [
    "The claim was submitted through the online portal.",
    "All supporting documents were attached.",
    "The policyholder was informed of the next steps.",
    "This claim was logged by the intake team.",
]


def _is_fact_token(word: str) -> bool:
    if _NUMBER.search(word):
        return True
    if _POLICY_ID.search(word):
        return True
    return word.strip(".,").lower() in _DURATION_WORDS


def _corrupt_one_char(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    i = rng.randrange(len(word))
    op = rng.choice(("dup", "drop", "swap"))
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    if op == "drop":
        return word[:i] + word[i + 1 :]
    # swap with neighbour
    if i + 1 < len(word):
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word


def perturb_typo(text: str, rng: random.Random) -> str:
    out = []
    for w in text.split():
        if not _is_fact_token(w) and len(w) > 4 and rng.random() < 0.4:
            out.append(_corrupt_one_char(w, rng))
        else:
            out.append(w)
    return " ".join(out)


def perturb_whitespace_case(text: str, rng: random.Random) -> str:
    out = []
    for w in text.split():
        if _is_fact_token(w):
            out.append(w)  # never recase a fact token (policy id casing may matter)
            continue
        if rng.random() < 0.3:
            out.append(w.upper() if rng.random() < 0.5 else w.lower())
        else:
            out.append(w)
    sep = "  " if rng.random() < 0.5 else " "
    return sep.join(out)


def perturb_irrelevant_suffix(text: str, rng: random.Random) -> str:
    return text + " " + rng.choice(_SUFFIXES)


def _parse_rupees(token: str) -> int | None:
    digits = re.sub(r"[^\d]", "", token)
    return int(digits) if digits else None


def perturb_number_reformat(text: str, rng: random.Random) -> str:
    """Re-render rupee amounts in an equivalent format. Asserts the numeric
    value is unchanged -- a failed assert is a perturbation bug, not a finding.
    """
    # match "Rs 50,000", "Rs 50000", "Rs. 1,20,000"
    pattern = re.compile(r"Rs\.?\s*([\d,]+)")

    def repl(m: re.Match[str]) -> str:
        original = m.group(0)
        value = _parse_rupees(m.group(1))
        if value is None:
            return original
        forms = [f"INR {value}", f"Rs {value:,}", f"Rs.{value}", f"INR {value:,}/-"]
        new = rng.choice(forms)
        # the guarantee: the re-rendered amount must parse to the same value
        assert _parse_rupees(new) == value, f"number_reformat altered value: {original}->{new}"
        return new

    return pattern.sub(repl, text)


AXES: dict[str, Callable[[str, random.Random], str]] = {
    "typo": perturb_typo,
    "whitespace_case": perturb_whitespace_case,
    "irrelevant_suffix": perturb_irrelevant_suffix,
    "number_reformat": perturb_number_reformat,
}


def perturb(text: str, axis: str, seed: int) -> str:
    """Apply one named axis deterministically. Same (text, axis, seed) -> same
    output, every run -- reproducibility without a frozen artifact.
    """
    return AXES[axis](text, random.Random(seed))
