"""Index the corpus into Qdrant. Run locally with a real GEMINI_API_KEY.

    uv run python scripts/index.py

This is a one-shot operational script, not part of the tested code path: it
performs real network calls (Gemini) and real writes (Qdrant), so it is never
exercised in CI.
"""

from eval_engine.config import get_settings
from eval_engine.indexing import GeminiEmbedder, index_chunks, make_client
from eval_engine.ingestion import parse_corpus


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set (put it in .env).")

    chunks = parse_corpus()
    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    client = make_client(settings)
    count = index_chunks(chunks, embedder, client, settings.qdrant_collection)
    print(f"Indexed {count} chunks into '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()
