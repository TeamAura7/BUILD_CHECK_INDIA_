"""
backend/rag/ingestion/indexer.py
=================================
FAISS index builder, persister, and loader — one index per municipality
(BBMP, MCGM, ...) so byelaws never mix across cities and swapping/adding a
municipality never requires code changes.

Index type: IndexFlatIP (exact inner-product = cosine for unit vectors)

Persisted per municipality, under settings.vector_store_dir_for(municipality):
    faiss.index      — binary FAISS index
    metadata.jsonl    — one JSON object per chunk (all fields)
    file_hashes.json  — {filename: sha256} for idempotency

Public API:
    build_and_save(municipality, chunks)  -> None
    add_chunks(municipality, chunks)      -> None  (incremental update)
    load_index(municipality)              -> (faiss.Index, list[dict])
    already_indexed(municipality, path)   -> bool
    get_index_stats(municipality)         -> dict
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)


def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def _load_hashes(municipality: str, settings: Settings) -> dict[str, str]:
    path = settings.file_hashes_path(municipality)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_hashes(municipality: str, hashes: dict[str, str], settings: Settings) -> None:
    path = settings.file_hashes_path(municipality)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(hashes, f, indent=2)


def already_indexed(
    municipality: str, pdf_path: str | Path, settings: Settings | None = None
) -> bool:
    settings = settings or get_settings()
    pdf_path = Path(pdf_path)
    hashes = _load_hashes(municipality, settings)
    return hashes.get(pdf_path.name) == _sha256(str(pdf_path))


def _mark_indexed(municipality: str, pdf_path: str | Path, settings: Settings) -> None:
    pdf_path = Path(pdf_path)
    hashes = _load_hashes(municipality, settings)
    hashes[pdf_path.name] = _sha256(str(pdf_path))
    _save_hashes(municipality, hashes, settings)


# ─── FAISS Helpers ───────────────────────────────────────────────────────────


def _load_faiss():
    try:
        import faiss

        return faiss
    except ImportError:
        raise ImportError("faiss-cpu is not installed. Run: pip install faiss-cpu")


def _load_metadata(municipality: str, settings: Settings) -> list[dict[str, Any]]:
    path = settings.metadata_jsonl_path(municipality)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_metadata_append(
    municipality: str, chunks: list[dict[str, Any]], settings: Settings
) -> None:
    path = settings.metadata_jsonl_path(municipality)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ─── Public API ──────────────────────────────────────────────────────────────


def load_index(municipality: str, settings: Settings | None = None):
    """Load persisted FAISS index + metadata for a municipality.

    Returns (faiss.Index, list[dict]) or (None, []) if not yet built.
    """
    settings = settings or get_settings()
    faiss = _load_faiss()

    index_path = settings.faiss_index_path(municipality)
    if not index_path.exists():
        logger.info("No FAISS index found for %s at %s", municipality, index_path)
        return None, []

    index = faiss.read_index(str(index_path))
    metadata = _load_metadata(municipality, settings)
    logger.info(
        "Loaded FAISS index for %s: %d vectors, %d metadata records",
        municipality,
        index.ntotal,
        len(metadata),
    )
    return index, metadata


def build_and_save(
    municipality: str,
    chunks: list[dict[str, Any]],
    settings: Settings | None = None,
) -> None:
    """Build a fresh FAISS index for `municipality` from chunks. REPLACES
    any existing index for that municipality only — other municipalities'
    indices are untouched."""
    settings = settings or get_settings()
    from backend.rag.ingestion.embedder import embed_texts

    faiss = _load_faiss()

    if not chunks:
        logger.warning("No chunks provided to build_and_save for %s.", municipality)
        return

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks for %s...", len(texts), municipality)
    embeddings = embed_texts(texts, show_progress=True)

    index = faiss.IndexFlatIP(settings.embedding_dim)
    index.add(embeddings)

    index_path = settings.faiss_index_path(municipality)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved for %s: %d vectors -> %s", municipality, index.ntotal, index_path)

    metadata_path = settings.metadata_jsonl_path(municipality)
    metadata_path.unlink(missing_ok=True)
    _save_metadata_append(municipality, chunks, settings)
    logger.info("Metadata saved for %s: %d records -> %s", municipality, len(chunks), metadata_path)

    seen_files: set[str] = set()
    for c in chunks:
        fp = c.get("file_path", "")
        if fp and fp not in seen_files:
            _mark_indexed(municipality, fp, settings)
            seen_files.add(fp)


def add_chunks(
    municipality: str,
    chunks: list[dict[str, Any]],
    settings: Settings | None = None,
) -> None:
    """Incrementally add new chunks to an existing municipality index (or
    create one if none exists)."""
    settings = settings or get_settings()
    from backend.rag.ingestion.embedder import embed_texts

    faiss = _load_faiss()

    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d new chunks for %s...", len(texts), municipality)
    embeddings = embed_texts(texts, show_progress=True)

    index_path = settings.faiss_index_path(municipality)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = faiss.IndexFlatIP(settings.embedding_dim)
        index_path.parent.mkdir(parents=True, exist_ok=True)

    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    _save_metadata_append(municipality, chunks, settings)
    logger.info("Added %d vectors for %s. Total: %d", len(chunks), municipality, index.ntotal)

    seen_files: set[str] = set()
    for c in chunks:
        fp = c.get("file_path", "")
        if fp and fp not in seen_files:
            _mark_indexed(municipality, fp, settings)
            seen_files.add(fp)


def get_index_stats(municipality: str, settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    faiss = _load_faiss()
    index_path = settings.faiss_index_path(municipality)
    if not index_path.exists():
        return {"vectors": 0, "chunks": 0}
    index = faiss.read_index(str(index_path))
    metadata = _load_metadata(municipality, settings)
    return {"vectors": index.ntotal, "chunks": len(metadata)}
