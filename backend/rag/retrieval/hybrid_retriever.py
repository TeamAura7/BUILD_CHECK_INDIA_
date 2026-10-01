"""
backend/rag/retrieval/hybrid_retriever.py
==========================================
Reciprocal Rank Fusion (RRF) hybrid retriever over one municipality's
regulation corpus.

Merges dense (FAISS) and BM25 results using RRF scoring:

    rrf_score(d) = sum_r  1 / (k + rank_r(d))

The fused result is deduplicated by chunk_id and re-ranked by RRF score.
This is the single entry point RASE (rule drafting) and any future
"why does this rule apply" explanation feature should call.
"""

from __future__ import annotations

from typing import Any

from backend.config import Settings, get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)


def hybrid_search(
    query: str,
    municipality: str,
    top_k: int | None = None,
    city_filter: str | None = None,
    dense_k: int | None = None,
    bm25_k: int | None = None,
    rrf_k: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Perform hybrid retrieval by fusing dense and BM25 results via RRF,
    scoped to one municipality's regulation corpus.

    Returns fused, deduplicated ChunkDicts with an "rrf_score" field,
    sorted descending by RRF score.
    """
    settings = settings or get_settings()
    from backend.rag.retrieval.bm25_retriever import bm25_search
    from backend.rag.retrieval.dense_retriever import dense_search

    top_k = top_k or settings.rag_top_k_final
    dense_k = dense_k or settings.rag_top_k_dense
    bm25_k = bm25_k or settings.rag_top_k_bm25
    rrf_k = rrf_k or settings.rag_rrf_k
    municipality = municipality.upper()

    dense_results = dense_search(
        query, municipality, top_k=dense_k, city_filter=city_filter, settings=settings
    )
    bm25_results = bm25_search(
        query, municipality, top_k=bm25_k, city_filter=city_filter, settings=settings
    )

    logger.debug(
        "Dense: %d BM25: %d results before fusion (municipality=%s)",
        len(dense_results),
        len(bm25_results),
        municipality,
    )

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    def _accumulate(results: list[dict[str, Any]]) -> None:
        for rank, chunk in enumerate(results):
            cid = chunk.get("chunk_id", "")
            if not cid:
                cid = str(hash(chunk.get("text", "")))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    _accumulate(dense_results)
    _accumulate(bm25_results)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    final: list[dict[str, Any]] = []
    for cid in sorted_ids[:top_k]:
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = round(rrf_scores[cid], 6)
        chunk["retrieval_method"] = "hybrid"
        final.append(chunk)

    logger.info(
        "Hybrid search (%s) returned %d chunks (query=%r)",
        municipality,
        len(final),
        query[:60],
    )
    return final


def reload_retrievers(municipality: str) -> None:
    """Reload both underlying retrievers for a municipality after new ingestion."""
    from backend.rag.retrieval.bm25_retriever import reload as bm25_reload
    from backend.rag.retrieval.dense_retriever import reload as dense_reload

    dense_reload(municipality)
    bm25_reload(municipality)
