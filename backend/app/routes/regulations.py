"""
HTTP API layer for the regulation-corpus RAG pipeline.

ADDITIVE ONLY, same pattern as backend/app/routes/analyze.py: this file
does not reimplement ingestion or retrieval logic. It imports and calls
the exact same underlying functions used by the CLI entry point
(backend/tools/run_ingest_regulations.py):

    save uploaded PDFs -> ingest_municipality() -> parse_regulation_directory
        -> chunk_pages -> build_and_save / add_chunks (FAISS + BM25 index)

This gives the frontend a way to:
  1. upload PDFs of a municipality's building byelaws/NBC text and have
     them chunked + embedded into that municipality's FAISS/BM25 index
     (background job, pollable, mirrors the /api/analyze/* job pattern), and
  2. inspect exactly what is currently indexed for a municipality
     (source files, chunk counts, per-chunk text) so "where did this
     rule come from" is answerable directly from the vector store rather
     than from the placeholder data/runtime_rules/<muni>/rules.json.

Nothing here writes to rules.json. Compliance verdicts are always drafted
live from whatever is in the FAISS/BM25 index at query time (see
backend/rase/extractor.py + backend/app/routes/analyze.py).
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import get_settings
from backend.rag.ingestion.pipeline import ingest_municipality
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/regulations", tags=["regulations"])

# --- in-memory job store (mirrors analyze.py; separate namespace) --------
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "status": "pending",  # pending | running | done | error
            "created_at": datetime.now(timezone.utc).isoformat(),
            "log": [],
            "result": None,
            "error": None,
        }
    return job_id


def _log(job_id: str, message: str) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["log"].append(
                {"t": datetime.now(timezone.utc).isoformat(), "message": message}
            )
    logger.info(message)


def _set_status(job_id: str, status: str) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = status


def _run_ingest(job_id: str, municipality: str, force: bool) -> None:
    try:
        _set_status(job_id, "running")
        _log(job_id, f"[1/3] Parsing regulation PDFs for {municipality}...")
        stats = ingest_municipality(municipality, force=force)
        _log(
            job_id,
            f"[2/3] Chunked {stats['pages']} page(s) into {stats['chunks']} new chunk(s).",
        )
        _log(job_id, "[3/3] Embedded and wrote to FAISS + BM25 index.")
        _log(
            job_id,
            f"Done. {municipality}: {stats['vectors']} total vector(s) now indexed.",
        )
        with _JOBS_LOCK:
            _JOBS[job_id]["result"] = {"municipality": municipality, **stats}
        _set_status(job_id, "done")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _log(job_id, f"FATAL ERROR: {exc}")
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["error"] = str(exc)
                _JOBS[job_id]["traceback"] = tb
        _set_status(job_id, "error")


@router.post("/upload")
async def upload_regulations(
    municipality: str = Form(...),
    files: list[UploadFile] = File(...),
    force: bool = Form(False),
) -> dict[str, Any]:
    """
    Save one or more regulation PDFs under
    data/regulations/<MUNICIPALITY>/ and kick off ingestion
    (parse -> clause-aware chunk -> embed -> FAISS/BM25 index) as a
    background job. Poll /api/regulations/jobs/{job_id} for progress.
    """
    settings = get_settings()
    municipality = (municipality or "").strip().upper()
    if not municipality:
        raise HTTPException(status_code=400, detail="municipality is required.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    reg_dir = settings.regulations_dir_for(municipality)
    reg_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    saved: list[str] = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(
                status_code=400,
                detail=f"Only .pdf/.txt files are supported (got {f.filename!r}).",
            )
        contents = await f.read()
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"{f.filename} exceeds {settings.max_upload_size_mb}MB."
            )
        dest = reg_dir / Path(f.filename).name
        dest.write_bytes(contents)
        saved.append(dest.name)

    job_id = _new_job()
    _log(job_id, f"Saved {len(saved)} file(s) to {reg_dir}: {', '.join(saved)}")
    thread = threading.Thread(
        target=_run_ingest, args=(job_id, municipality, force), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "municipality": municipality, "saved_files": saved}


@router.get("/jobs/{job_id}")
def get_ingest_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(job)


def _load_metadata_rows(municipality: str) -> list[dict[str, Any]]:
    settings = get_settings()
    path = settings.metadata_jsonl_path(municipality.upper())
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@router.get("")
def list_regulation_libraries() -> dict[str, Any]:
    """All municipalities that have either raw regulation files or an
    index built, with quick counts, so the frontend can render a picker
    without needing a municipality selected first."""
    settings = get_settings()
    munis: set[str] = set()
    for base in (settings.regulations_dir, settings.vector_store_dir):
        p = Path(base)
        if p.exists():
            munis.update(d.name for d in p.iterdir() if d.is_dir())

    out = []
    for m in sorted(munis):
        rows = _load_metadata_rows(m)
        reg_dir = settings.regulations_dir_for(m)
        raw_files = sorted(f.name for f in reg_dir.glob("*")) if reg_dir.exists() else []
        out.append(
            {
                "municipality": m,
                "source_files": sorted({r.get("source_file", "") for r in rows if r.get("source_file")}),
                "chunks": len(rows),
                "raw_files": raw_files,
            }
        )
    return {"municipalities": out}


@router.get("/{municipality}")
def get_regulation_library(municipality: str) -> dict[str, Any]:
    """
    What is currently indexed for one municipality: per-document chunk
    counts plus enough per-chunk metadata (clause_ref, page_num) for the
    frontend to show provenance without a retrieval query.
    """
    settings = get_settings()
    municipality = municipality.upper()
    rows = _load_metadata_rows(municipality)

    reg_dir = settings.regulations_dir_for(municipality)
    raw_files = sorted(f.name for f in reg_dir.glob("*")) if reg_dir.exists() else []

    by_source: dict[str, dict[str, Any]] = {}
    for r in rows:
        src = r.get("source_file", "unknown")
        entry = by_source.setdefault(
            src,
            {
                "source_file": src,
                "doc_type": r.get("doc_type", "UNKNOWN"),
                "city": r.get("city", "UNKNOWN"),
                "chunks": 0,
                "pages": set(),
            },
        )
        entry["chunks"] += 1
        entry["pages"].add(r.get("page_num", 0))

    documents = []
    for entry in by_source.values():
        entry["pages"] = len(entry["pages"])
        documents.append(entry)
    documents.sort(key=lambda d: d["source_file"])

    return {
        "municipality": municipality,
        "documents": documents,
        "raw_files": raw_files,
        "total_chunks": len(rows),
    }


class RetrievalTestRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


@router.post("/{municipality}/search")
def test_retrieval(municipality: str, payload: RetrievalTestRequest) -> dict[str, Any]:
    """
    Run the exact same FAISS + BM25 + RRF hybrid_search() used by the
    compliance pipeline, standalone, so the frontend can show which
    chunks a query would retrieve without running a full plan check.
    """
    from backend.rag.retrieval.hybrid_retriever import hybrid_search

    settings = get_settings()
    municipality = municipality.upper()
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required.")
    chunks = hybrid_search(
        payload.query, municipality, top_k=payload.top_k or settings.rag_top_k_final, settings=settings
    )
    return {"municipality": municipality, "query": payload.query, "chunks": chunks}
