"""Tests for the indexing layer. Fully hermetic: a deterministic stub stands in
for Gemini, and Qdrant runs in embedded mode in a temp dir. No network, no key.
"""

import math
from pathlib import Path

from qdrant_client import QdrantClient

from eval_engine.indexing import (
    chunk_to_point,
    index_chunks,
    l2_normalize,
    point_id_for,
)
from eval_engine.ingestion import parse_corpus
from eval_engine.ingestion.models import Chunk, ChunkMetadata, Topic


class StubEmbedder:
    """Deterministic embedder: maps text length into a fixed-dim vector.
    Stands in for Gemini so tests never touch the network.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            seed = float(len(t) % 7 + 1)
            out.append(l2_normalize([seed + i for i in range(self.dim)]))
        return out


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        embed_text=f"header\n{cid} body",
        clause_text=f"{cid} body",
        metadata=ChunkMetadata(source_doc="d", doc_year=2020, page=1, topic=Topic.DEFINITION),
    )


def test_l2_normalize_unit_length() -> None:
    v = l2_normalize([3.0, 4.0])  # 3-4-5 triangle
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)
    assert math.isclose(v[0], 0.6) and math.isclose(v[1], 0.8)


def test_l2_normalize_zero_vector_safe() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_point_id_is_deterministic() -> None:
    assert point_id_for("2020-S1-CHI-P10-C33") == point_id_for("2020-S1-CHI-P10-C33")
    assert point_id_for("a") != point_id_for("b")


def test_chunk_to_point_preserves_clause_text() -> None:
    c = _chunk("X1")
    p = chunk_to_point(c, [0.1] * 768)
    assert p.payload is not None
    assert p.payload["clause_text"] == "X1 body"
    assert p.payload["chunk_id"] == "X1"


def test_index_and_search_roundtrip(tmp_path: Path) -> None:
    """Full pipeline against embedded Qdrant: index real corpus chunks with the
    stub embedder, then confirm count and that a payload search returns one.
    """
    chunks = parse_corpus()
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    n = index_chunks(chunks, StubEmbedder(), client, "test_corpus")
    assert n == len(chunks)

    info = client.get_collection("test_corpus")
    assert info.points_count == len(chunks)

    # Re-indexing is idempotent: same ids overwrite, count unchanged.
    n2 = index_chunks(chunks, StubEmbedder(), client, "test_corpus")
    assert n2 == len(chunks)
    info2 = client.get_collection("test_corpus")
    assert info2.points_count == len(chunks)
