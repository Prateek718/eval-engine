"""Conservative text normalization for citation fidelity.

pdfplumber already extracts clean text, so this does the minimum: it joins
mid-sentence line wraps while preserving meaningful structure (list-item
boundaries). It deliberately does NOT do aggressive find/replace, because the
golden set cites this text verbatim and altering a number or term would
silently corrupt adjudication labels.
"""

import re

# A line that begins a new list item / clause, e.g. "a)", "1.", "ii)", "(3)"
_LIST_ITEM_START = re.compile(r"^\s*(\(?[a-zA-Z0-9]{1,4}[).]|\u2022)")


def normalize_clause_text(raw: str) -> str:
    """Join line-wrapped prose while keeping list-item line breaks intact.

    Rule: a newline is a real break only if the next line starts a new list
    item (a), b), 1., etc.) or is blank. Otherwise it is line-wrap and the
    lines are joined with a single space.
    """
    lines = raw.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if out and not _LIST_ITEM_START.match(line):
            # continuation of the previous line (line-wrap) -> join with space
            out[-1] = f"{out[-1]} {stripped}"
        else:
            out.append(stripped)
    # collapse any double spaces introduced by joining
    return "\n".join(re.sub(r" {2,}", " ", line) for line in out).strip()
