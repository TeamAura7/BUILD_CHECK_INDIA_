"""
backend/rag/retrieval/bm25_retriever.py
========================================
BM25 keyword retrieval over a municipality's regulation corpus, using
rank-bm25.

Each municipality gets its own in-memory BM25 index, built lazily from
that municipality's metadata.jsonl and cached per-process.
"""

from __future__ import annotations

import re
from typing import Any

from backend.config import Settings, get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

# Per-municipality cache: {municipality: (BM25Okapi, corpus_list)}
_cache: dict[str, tuple[Any, list[dict[str, Any]]]] = {}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
    return text.split()


def _ensure_loaded(municipality: str, settings: Settings) -> None:
    municipality = municipality.upper()
    if municipality in _cache:
        return

    from backend.rag.ingestion.indexer import _load_metadata
    from rank_bm25 import BM25Okapi

    corpus = _load_metadata(municipality, settings)
    if not corpus:
        logger.warning("BM25: no metadata for %s; index not built yet.", municipality)
        _cache[municipality] = (None, [])
        return

    tokenized = [_tokenize(c.get("text", "")) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    _cache[municipality] = (bm25, corpus)
    logger.info("BM25 index built for %s over %d documents.", municipality, len(corpus))


def reload(municipality: str) -> None:
    """Force rebuild of a municipality's BM25 index (call after new ingestion)."""
    _cache.pop(municipality.upper(), None)


def bm25_search(
    query: str,
    municipality: str,
    top_k: int | None = None,
    city_filter: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Keyword search using BM25, scoped to one municipality's corpus.

    Returns a list of ChunkDicts with an added "score" field.
    """
    settings = settings or get_settings()
    top_k = top_k or settings.rag_top_k_bm25
    municipality = municipality.upper()

    _ensure_loaded(municipality, settings)
    bm25, corpus = _cache.get(municipality, (None, []))
    if bm25 is None or not corpus:
        return []

    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for idx, score in ranked:
        if score <= 0:
            break
        chunk = dict(corpus[idx])
        chunk["score"] = float(score)
        chunk["retrieval_method"] = "bm25"

        if city_filter:
            c = chunk.get("city", "UNKNOWN")
            if c not in (city_filter, "NATIONAL"):
                continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
