"""Indexing: chunks -> embedded vectors -> Qdrant.

The Qdrant backend is selected from settings: a server URL if provided, else
embedded local-file mode. The rest of the code is identical either way, which
is what lets dev (embedded) and deployment (server) share one path.

Qdrant point ids must be ints or UUIDs, but our chunk_id is a legible string
(e.g. 2020-S1-CHI-P10-C33). We keep the legible id in the payload and derive a
deterministic UUID5 from it for the point id, so re-indexing the same chunk
overwrites rather than duplicates.
"""

import uuid

from qdrant_client import QdrantClient, models

from eval_engine.config import Settings
from eval_engine.indexing.embedder import Embedder
from eval_engine.ingestion.models import Chunk

# Fixed namespace so chunk_id -> point id is stable across runs.
_NAMESPACE = uuid.UUID("b9d0e2a4-3c5f-4e8a-9b1d-2f6a7c8e0d11")


def point_id_for(chunk_id: str) -> str:
    """Deterministic UUID5 from the legible chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def chunk_to_point(chunk: Chunk, vector: list[float]) -> models.PointStruct:
    """Map a chunk + its vector to a Qdrant point. The full chunk (including
    the verbatim clause_text the golden set cites) is stored in the payload.
    """
    return models.PointStruct(
        id=point_id_for(chunk.chunk_id),
        vector=vector,
        payload={
            "chunk_id": chunk.chunk_id,
            "clause_text": chunk.clause_text,
            "embed_text": chunk.embed_text,
            **chunk.metadata.model_dump(mode="json"),
        },
    )


def make_client(settings: Settings) -> QdrantClient:
    """Server when a URL is configured, else embedded local-file."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Create the collection if absent, sized to the embedding dim with cosine
    distance. Recreated dim must match the embedder, or search returns garbage.
    """
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


def index_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    client: QdrantClient,
    collection: str,
) -> int:
    """Embed and upsert all chunks. Returns the number indexed."""
    ensure_collection(client, collection, embedder.dim)
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    points = [chunk_to_point(c, v) for c, v in zip(chunks, vectors, strict=True)]
    client.upsert(collection_name=collection, points=points)
    return len(points)
