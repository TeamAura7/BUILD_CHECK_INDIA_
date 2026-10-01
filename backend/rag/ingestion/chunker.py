"""
backend/rag/ingestion/chunker.py
=================================
Clause-aware text splitter for Indian building codes.

NBC 2016 clause format:   "7.3.1  Minimum Plot Area..."
DCR section format:        "Section 23  Height Restrictions..."
                           "Regulation 12.  Setbacks..."
                           "Chapter IV  Use Regulations..."

Strategy:
1. Detect clause/section boundary lines using compiled regex patterns.
2. Split the page stream on those boundaries, carrying the heading into
   each chunk (so chunks are self-contained and citable).
3. If a clause chunk exceeds max_tokens, sub-split on paragraph
   boundaries with token-based overlap.

Output ChunkDict:
    {
        "chunk_id":    str,   # "{municipality}::{source_file}::p{page_num}::{clause_ref}::{idx}"
        "text":        str,   # chunk text (includes heading)
        "clause_ref":  str,   # parsed clause number / heading
        "source_file": str,
        "file_path":   str,
        "doc_type":    str,
        "city":        str,
        "municipality": str,
        "page_num":    int,   # page where the clause starts
    }
"""

from __future__ import annotations

import re
from typing import Any

from backend.config import Settings, get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

# ─── Clause Boundary Patterns ───────────────────────────────────────────────

_NBC_CLAUSE = re.compile(
    r"^(?P<ref>\d{1,2}(?:\.\d{1,3}){1,4})\s{1,4}(?P<heading>[A-Z][^\n]{3,})",
    re.MULTILINE,
)

_DCR_SECTION = re.compile(
    r"^(?P<ref>(?:Section|Regulation|Clause|Rule|Chapter|Article|Part)\s+[\dIVXivx]+[A-Za-z]?\.?)\s*(?P<heading>[^\n]*)",
    re.MULTILINE | re.IGNORECASE,
)

_NUMBERED_HEADING = re.compile(
    r"^(?P<ref>\d{1,3}\.)\s+(?P<heading>[A-Z][^\n]{5,})",
    re.MULTILINE,
)

_ALL_PATTERNS = [_NBC_CLAUSE, _DCR_SECTION, _NUMBERED_HEADING]


def _find_clause_boundaries(text: str) -> list[dict[str, Any]]:
    hits: dict[int, dict] = {}
    for pattern in _ALL_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if pos not in hits:  # first match wins (NBC > DCR > numbered)
                hits[pos] = {
                    "start": pos,
                    "ref": m.group("ref").strip(),
                    "heading": m.group("heading").strip(),
                }
    return sorted(hits.values(), key=lambda x: x["start"])


def _approx_tokens(text: str) -> int:
    """Approximate token count (1 token ~= 4 chars for English/Indic mix)."""
    return len(text) // 4


def _sub_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return [text.strip()]

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))
            overlap_paras: list[str] = []
            overlap_acc = 0
            for p in reversed(current_paras):
                if overlap_acc + _approx_tokens(p) <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_acc += _approx_tokens(p)
                else:
                    break
            current_paras = overlap_paras + [para]
            current_tokens = sum(_approx_tokens(p) for p in current_paras)
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks if chunks else [text.strip()]


# ─── Public API ──────────────────────────────────────────────────────────────


def chunk_pages(
    pages: list[dict[str, Any]],
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a list of PageDicts (from `pdf_parser`) into ChunkDicts.

    Works on a per-page basis: each page's text is split on clause
    boundaries. Chunks inherit the page's metadata (including municipality).
    """
    settings = settings or get_settings()
    max_tokens = max_tokens or settings.rag_max_chunk_tokens
    overlap_tokens = overlap_tokens or settings.rag_chunk_overlap_tokens

    all_chunks: list[dict[str, Any]] = []

    for page in pages:
        text = page["text"]
        meta = {k: v for k, v in page.items() if k != "text"}

        boundaries = _find_clause_boundaries(text)

        if not boundaries:
            chunks_text = _sub_split(text, max_tokens, overlap_tokens)
            for idx, ct in enumerate(chunks_text):
                if ct.strip():
                    all_chunks.append(_make_chunk(ct, "UNCATEGORISED", idx, meta))
            continue

        for i, bnd in enumerate(boundaries):
            start = bnd["start"]
            end = boundaries[i + 1]["start"] if i + 1 < len(boundaries) else len(text)
            clause_text = text[start:end].strip()
            ref = f"{bnd['ref']} {bnd['heading']}".strip()

            if not clause_text:
                continue

            if _approx_tokens(clause_text) <= max_tokens:
                all_chunks.append(_make_chunk(clause_text, ref, 0, meta))
            else:
                sub_chunks = _sub_split(clause_text, max_tokens, overlap_tokens)
                for idx, sc in enumerate(sub_chunks):
                    if sc.strip():
                        all_chunks.append(_make_chunk(sc, ref, idx, meta))

    logger.info(
        "Chunking produced %d chunks from %d pages", len(all_chunks), len(pages)
    )
    return all_chunks


def _make_chunk(
    text: str,
    clause_ref: str,
    sub_idx: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    source = meta.get("source_file", "unknown")
    municipality = meta.get("municipality", "UNKNOWN")
    page_num = meta.get("page_num", 0)
    # `page_num` must be part of the id: two different pages of the same
    # source document both lacking a detectable clause boundary (e.g. a
    # preface page and an appendix page) previously both produced
    # "{municipality}::{source}::UNCATEGORISED::0" -- an identical chunk_id
    # for genuinely different text. `hybrid_retriever.hybrid_search`
    # deduplicates fused results by this id, so the collision silently
    # dropped one page's chunk from a fused result set and could attribute
    # a citation to the wrong page. Confirmed reproducible with any two-page
    # document where both pages lack clause markers.
    chunk_id = f"{municipality}::{source}::p{page_num}::{clause_ref}::{sub_idx}"
    return {
        "chunk_id": chunk_id,
        "text": text,
        "clause_ref": clause_ref,
        "source_file": source,
        "file_path": meta.get("file_path", ""),
        "doc_type": meta.get("doc_type", "UNKNOWN"),
        "city": meta.get("city", "UNKNOWN"),
        "municipality": municipality,
        "page_num": meta.get("page_num", 0),
    }
