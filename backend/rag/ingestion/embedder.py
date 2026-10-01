"""
backend/rag/ingestion/embedder.py
==================================
HuggingFace sentence-transformer embedder for the regulation corpus.

Default model: sentence-transformers/all-MiniLM-L6-v2 (settings.embedding_model_name)
  - 384-dimensional embeddings (settings.embedding_dim)
  - Vectors are L2-normalised for cosine similarity via FAISS IndexFlatIP

Public API:
    embed_texts(texts)   -> np.ndarray  shape (N, dim) float32
    embed_query(query)   -> np.ndarray  shape (1, dim) float32
    get_model()          -> SentenceTransformer (singleton, lazy)
    set_model(fn)         -> inject a stub encoder (tests / offline smoke runs)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from backend.config import get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

_model = None  # singleton
_stub_encode: Optional[Callable[[list[str]], np.ndarray]] = None


def set_stub_encoder(fn: Callable[[list[str]], np.ndarray] | None) -> None:
    """
    Inject a deterministic stand-in encoder (e.g. a hash-based embedding)
    so tests and offline smoke-runs don't require downloading the real
    sentence-transformers model. Pass None to restore the real model.
    """
    global _stub_encode
    _stub_encode = fn


def get_model():
    """Load (or return cached) SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        _model = SentenceTransformer(settings.embedding_model_name)
        logger.info("Model loaded.")
    return _model


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of strings.

    Returns np.ndarray of shape (len(texts), embedding_dim), dtype float32,
    L2-normalised (unit vectors).
    """
    settings = get_settings()
    if not texts:
        return np.empty((0, settings.embedding_dim), dtype=np.float32)

    if _stub_encode is not None:
        vecs = _stub_encode(texts).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, embedding_dim)."""
    return embed_texts([query])
