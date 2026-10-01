"""
backend/rag/retrieval/dense_retriever.py
=========================================
FAISS-based dense vector retrieval, one cached index per municipality.

On each query, embeds the query string and returns the top-K most similar
chunks by inner-product (= cosine for unit-normalised vectors).
"""

from __future__ import annotations

from typing import Any

from backend.config import Settings, get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

# Per-municipality cache: {municipality: (faiss.Index, metadata_list)}
_cache: dict[str, tuple[Any, list[dict[str, Any]]]] = {}


def _ensure_loaded(municipality: str, settings: Settings) -> None:
    municipality = municipality.upper()
    if municipality in _cache:
        return
    from backend.rag.ingestion.indexer import load_index

    index, metadata = load_index(municipality, settings=settings)
    _cache[municipality] = (index, metadata)


def reload(municipality: str) -> None:
    """Force reload of a municipality's FAISS index from disk (call after new ingestion)."""
    _cache.pop(municipality.upper(), None)


def dense_search(
    query: str,
    municipality: str,
    top_k: int | None = None,
    city_filter: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Search a municipality's FAISS index for top-K chunks most similar to
    `query`.

    Returns a list of ChunkDicts with an added "score" field (cosine similarity).
    """
    settings = settings or get_settings()
    top_k = top_k or settings.rag_top_k_dense
    municipality = municipality.upper()

    _ensure_loaded(municipality, settings)
    index, metadata = _cache.get(municipality, (None, []))
    if index is None or index.ntotal == 0:
        logger.warning("FAISS index for %s is empty or not loaded.", municipality)
        return []

    from backend.rag.ingestion.embedder import embed_query

    query_vec = embed_query(query)

    search_k = top_k * 4 if city_filter else top_k
    search_k = min(search_k, index.ntotal)

    scores, indices = index.search(query_vec, search_k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])
        chunk["score"] = float(score)
        chunk["retrieval_method"] = "dense"

        if city_filter:
            c = chunk.get("city", "UNKNOWN")
            if c not in (city_filter, "NATIONAL"):
                continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
