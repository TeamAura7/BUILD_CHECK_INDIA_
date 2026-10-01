"""
backend/rag/ingestion/pipeline.py
==================================
Orchestrates the full regulation-ingestion flow for one municipality:

    PDFs/.txt in data/regulations/<MUNICIPALITY>/
        -> parse_regulation_directory   (pdf_parser.py)
        -> chunk_pages                  (chunker.py)
        -> build_and_save / add_chunks  (indexer.py, embeds via embedder.py)

Idempotent: files already indexed (by content hash) are skipped unless
`force=True`.
"""

from __future__ import annotations

from backend.config import Settings, get_settings
from backend.rag.ingestion.chunker import chunk_pages
from backend.rag.ingestion.indexer import add_chunks, already_indexed, build_and_save, get_index_stats
from backend.rag.ingestion.pdf_parser import parse_regulation_directory
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)


def ingest_municipality(
    municipality: str,
    force: bool = False,
    settings: Settings | None = None,
) -> dict[str, int]:
    """
    Ingest every regulation document under
    settings.regulations_dir_for(municipality) into that municipality's
    FAISS + BM25-backing metadata store.

    Args:
        municipality: e.g. "BBMP".
        force:        if True, re-parse and rebuild the whole index even
                       for files already indexed by content hash.

    Returns:
        {"pages": N, "chunks": N, "vectors": N} summary.
    """
    settings = settings or get_settings()
    municipality = municipality.upper()

    pages = parse_regulation_directory(municipality, settings=settings)
    if not pages:
        logger.warning("No regulation pages found for %s; nothing to ingest.", municipality)
        return {"pages": 0, "chunks": 0, "vectors": 0}

    if not force:
        # Filter out pages whose source file is already indexed unchanged.
        pages = [
            p
            for p in pages
            if not already_indexed(municipality, p["file_path"], settings=settings)
        ]
        if not pages:
            logger.info("All regulation files for %s already indexed; nothing new.", municipality)
            stats = get_index_stats(municipality, settings=settings)
            return {"pages": 0, "chunks": 0, "vectors": stats["vectors"]}

    chunks = chunk_pages(pages, settings=settings)
    if not chunks:
        logger.warning("Chunking produced no chunks for %s.", municipality)
        return {"pages": len(pages), "chunks": 0, "vectors": 0}

    stats_before = get_index_stats(municipality, settings=settings)
    if force or stats_before["vectors"] == 0:
        build_and_save(municipality, chunks, settings=settings)
    else:
        add_chunks(municipality, chunks, settings=settings)

    # Reload retrievers so subsequent hybrid_search calls in this process
    # see the freshly ingested chunks immediately.
    from backend.rag.retrieval.hybrid_retriever import reload_retrievers

    reload_retrievers(municipality)

    stats_after = get_index_stats(municipality, settings=settings)
    logger.info(
        "Ingested %s: %d pages -> %d chunks -> %d total vectors",
        municipality,
        len(pages),
        len(chunks),
        stats_after["vectors"],
    )
    return {"pages": len(pages), "chunks": len(chunks), "vectors": stats_after["vectors"]}
