"""Corpus parsing: PDF source documents into contextual, addressable chunks."""

from eval_engine.ingestion.corpus import CORPUS
from eval_engine.ingestion.models import Chunk, ChunkMetadata, Topic
from eval_engine.ingestion.parser import DocSpec, PageRange, parse_document

__all__ = [
    "CORPUS",
    "Chunk",
    "ChunkMetadata",
    "DocSpec",
    "PageRange",
    "Topic",
    "parse_corpus",
]


def parse_corpus(specs: list[DocSpec] | None = None) -> list[Chunk]:
    """Parse every document in the corpus into a flat list of chunks."""
    specs = specs if specs is not None else CORPUS
    chunks: list[Chunk] = []
    for spec in specs:
        chunks.extend(parse_document(spec))
    return chunks
