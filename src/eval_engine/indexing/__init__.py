"""Indexing: embed parsed chunks and store them in Qdrant."""

from eval_engine.indexing.embedder import Embedder, GeminiEmbedder, l2_normalize
from eval_engine.indexing.indexer import (
    chunk_to_point,
    ensure_collection,
    index_chunks,
    make_client,
    point_id_for,
)

__all__ = [
    "Embedder",
    "GeminiEmbedder",
    "chunk_to_point",
    "ensure_collection",
    "index_chunks",
    "l2_normalize",
    "make_client",
    "point_id_for",
]
