"""Tests for corpus parsing. These run fully offline (no Gemini, no Qdrant),
so they execute in CI against the committed source PDFs.
"""

from collections import Counter

import pytest

from eval_engine.ingestion import parse_corpus
from eval_engine.ingestion.models import Chunk, Topic
from eval_engine.ingestion.normalize import normalize_clause_text


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return parse_corpus()


def test_corpus_parses_nonempty(chunks: list[Chunk]) -> None:
    assert len(chunks) > 50


def test_all_chunk_ids_unique(chunks: list[Chunk]) -> None:
    """Regression guard: chunk_ids are cited by the golden set, so any
    collision makes a citation ambiguous. A page component was added to the
    id specifically to prevent cross-page clause-number collisions.
    """
    ids = [c.chunk_id for c in chunks]
    dupes = [cid for cid, n in Counter(ids).items() if n > 1]
    assert not dupes, f"duplicate chunk ids: {dupes}"


def test_clause_text_excludes_context_header(chunks: list[Chunk]) -> None:
    """clause_text must be verbatim regulatory text only — never the prepended
    context header (which would pollute faithfulness scoring).
    """
    for c in chunks:
        assert not c.clause_text.startswith("["), c.chunk_id


def test_ped_definition_present_and_addressable(chunks: list[Chunk]) -> None:
    ped = [c for c in chunks if c.chunk_id.endswith("C33") and c.metadata.doc_year == 2020]
    assert len(ped) == 1
    c = ped[0]
    assert "Pre-existing" in c.clause_text or "Pre-Existing" in c.clause_text
    assert c.metadata.topic is Topic.DEFINITION
    assert c.metadata.page == 10


def test_permanent_exclusion_table_rows(chunks: list[Chunk]) -> None:
    table_chunks = [c for c in chunks if c.metadata.is_table]
    assert len(table_chunks) >= 10
    sarcoidosis = [c for c in table_chunks if "Sarcoidosis" in c.clause_text]
    assert sarcoidosis, "expected the Sarcoidosis row in the permanent-exclusions table"


def test_both_documents_represented(chunks: list[Chunk]) -> None:
    docs = Counter(c.metadata.source_doc for c in chunks)
    assert "irdai_master_circular_health_2024" in docs
    assert "irdai_standardization_health_2020" in docs


def test_normalize_joins_wraps_preserves_list_items() -> None:
    raw = (
        "means any disease:\n"
        "a) diagnosed within 48 months prior to the effective\n"
        "date of the policy or\n"
        "b) for which advice was received."
    )
    out = normalize_clause_text(raw)
    # line-wrap joined: "effective date" reunited
    assert "effective date" in out
    # list structure preserved: a) and b) on their own lines
    lines = out.split("\n")
    assert any(line.startswith("a)") for line in lines)
    assert any(line.startswith("b)") for line in lines)
    # numbers untouched
    assert "48 months" in out
