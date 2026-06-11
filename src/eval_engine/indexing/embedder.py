"""Embedding: text -> normalized vectors via Gemini.

Two correctness points specific to ``gemini-embedding-001`` at 768 dims:

- ``task_type`` is asymmetric. Documents are embedded with RETRIEVAL_DOCUMENT
  here; queries are embedded with RETRIEVAL_QUERY at search time (PR-4). Using
  the right type on each side aligns the subspaces and improves retrieval.
- Only the full 3072-dim output is pre-normalized by the API. Any MRL-truncated
  dimension (we use 768) must be L2-normalized by us before cosine similarity,
  or distances are distorted. That normalization is done here, not assumed.

The embedder is defined against a small Protocol so tests can substitute a
deterministic stub and never touch the network.
"""

import math
from typing import Protocol


def l2_normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length. Required for truncated-dim Gemini
    embeddings before cosine similarity (the API only pre-normalizes 3072-dim).
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class Embedder(Protocol):
    """The seam: anything that turns texts into vectors of a fixed dim."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class GeminiEmbedder:
    """Real embedder. Calls Gemini and L2-normalizes each returned vector."""

    # Gemini's per-request content cap for this model.
    _BATCH_SIZE = 100

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        # Imported lazily so the module (and the stub-based tests) load without
        # the SDK or any credentials present.
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            batch: list[str] = texts[start : start + self._BATCH_SIZE]
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,  # type: ignore[arg-type]  # SDK union doesn't list[str] cleanly
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dim,
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            embeddings = result.embeddings or []
            for e in embeddings:
                if e.values is None:
                    raise RuntimeError("Gemini returned an embedding with no values")
                vectors.append(l2_normalize(list(e.values)))
        return vectors
