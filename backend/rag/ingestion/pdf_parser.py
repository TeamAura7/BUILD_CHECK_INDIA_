"""
backend/rag/ingestion/pdf_parser.py
====================================
PDF text extraction for regulation documents (NBC / DCR byelaws), using
PyMuPDF (primary) + pdfplumber (table fallback, optional).

Adapted from the standalone ingestion/pdf_parser.py contract, wired to
`backend.config.Settings` instead of a bespoke `config` module, so
doc-type/city inference keywords are configurable (not hard-coded) and
consistent with the rest of the app.

Each parsed document yields a list of PageDict objects:
    {
        "text":        str,   # extracted text for this page
        "page_num":    int,   # 1-indexed
        "source_file": str,   # filename
        "file_path":   str,   # absolute path
        "doc_type":    str,   # "NBC" | "DCR" | "UNKNOWN"
        "city":        str,   # e.g. "Bengaluru" | "NATIONAL" | "UNKNOWN"
        "municipality": str,  # the municipality this ingestion run is for
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid loading these at module-import time
_fitz = None
_pdfplumber = None


def _get_fitz():
    global _fitz
    if _fitz is None:
        import fitz  # PyMuPDF

        _fitz = fitz
    return _fitz


def _get_pdfplumber():
    global _pdfplumber
    if _pdfplumber is None:
        import pdfplumber

        _pdfplumber = pdfplumber
    return _pdfplumber


# ─── Metadata Inference ────────────────────────────────────────────────────


def _infer_doc_type(filename: str, settings: Settings) -> str:
    """Infer NBC vs DCR from filename keywords (configured, not hard-coded)."""
    lower = filename.lower()
    for kw, doc_type in settings.rag_doc_type_keywords.items():
        if kw in lower:
            return doc_type
    return "UNKNOWN"


def _infer_city(filename: str, settings: Settings) -> str:
    """Infer city name from filename. Returns 'NATIONAL' for NBC docs."""
    lower = filename.lower()
    if "nbc" in lower or "national_building" in lower:
        return "NATIONAL"
    for kw, city in settings.rag_city_keywords.items():
        if kw in lower:
            return city
    return "UNKNOWN"


# ─── PyMuPDF Extraction ─────────────────────────────────────────────────────


def _extract_with_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Primary extractor. Returns one dict per page with raw text."""
    fitz = _get_fitz()
    pages: list[dict[str, Any]] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("PyMuPDF failed to open %s: %s", pdf_path.name, exc)
        return pages

    for page_index in range(len(doc)):
        page = doc[page_index]
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        pages.append({"text": text, "page_num": page_index + 1})

    doc.close()
    return pages


def _extract_tables_with_pdfplumber(
    pdf_path: Path, target_pages: list[int]
) -> dict[int, str]:
    """Secondary extractor for table-heavy pages (BBMP setback/coverage/FAR
    thresholds very often live in lookup tables, not prose)."""
    try:
        pdfplumber = _get_pdfplumber()
    except ImportError:
        logger.warning("pdfplumber not installed; skipping table supplement.")
        return {}

    table_texts: dict[int, str] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pg in pdf.pages:
                page_num = pg.page_number
                if page_num not in target_pages:
                    continue
                tables = pg.extract_tables()
                if not tables:
                    continue
                rows_text: list[str] = []
                for table in tables:
                    for row in table:
                        if row:
                            cleaned = " | ".join(
                                cell.strip() if cell else "" for cell in row
                            )
                            rows_text.append(cleaned)
                if rows_text:
                    table_texts[page_num] = "\n".join(rows_text)
    except Exception as exc:
        logger.warning(
            "pdfplumber table extraction failed for %s: %s", pdf_path.name, exc
        )
    return table_texts


# ─── Public API ──────────────────────────────────────────────────────────────


def parse_pdf(
    pdf_path: Path | str,
    municipality: str,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Parse a single regulation PDF and return a list of PageDicts, scoped to
    `municipality` (e.g. "BBMP").

    Strategy:
    1. Extract all text via PyMuPDF.
    2. Detect pages with low text yield (likely table-heavy — thresholds
       are frequently tabular, not prose).
    3. Supplement those pages with pdfplumber table extraction.
    4. Attach metadata (doc_type, city, source_file, municipality).
    """
    settings = settings or get_settings()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Regulation PDF not found: {pdf_path}")

    doc_type = _infer_doc_type(pdf_path.stem, settings)
    city = _infer_city(pdf_path.stem, settings)

    logger.info(
        "Parsing %s [municipality=%s, doc_type=%s, city=%s]",
        pdf_path.name,
        municipality,
        doc_type,
        city,
    )

    raw_pages = _extract_with_pymupdf(pdf_path)

    avg_len = sum(len(p["text"]) for p in raw_pages) / max(len(raw_pages), 1)
    sparse_pages = [
        p["page_num"] for p in raw_pages if len(p["text"]) < 0.3 * avg_len
    ]

    if sparse_pages:
        table_supplement = _extract_tables_with_pdfplumber(pdf_path, sparse_pages)
        for page in raw_pages:
            if page["page_num"] in table_supplement:
                extra = table_supplement[page["page_num"]]
                page["text"] = (page["text"] + "\n\n[TABLE]\n" + extra).strip()

    result: list[dict[str, Any]] = []
    for page in raw_pages:
        text = page["text"].strip()
        if not text:
            continue
        result.append(
            {
                "text": text,
                "page_num": page["page_num"],
                "source_file": pdf_path.name,
                "file_path": str(pdf_path.resolve()),
                "doc_type": doc_type,
                "city": city,
                "municipality": municipality.upper(),
            }
        )

    logger.info("  -> %d usable pages extracted from %s", len(result), pdf_path.name)

    if not result:
        # Some regulation PDFs in this corpus are scanned/image-only, so
        # neither PyMuPDF's text layer nor pdfplumber's table extraction
        # finds anything -- but a pre-OCR'd sidecar text file for exactly
        # this document may already exist (found in this codebase:
        # "Revised setback gazette copy.pdf" -> 0 pages, sitting right next
        # to "Revised setback gazette copy_ocr.txt" with real content that
        # was never actually wired into ingestion). Use it if present,
        # rather than silently excluding a real regulatory document -- in
        # this case a 2025 government notification amending setbacks --
        # from the retrieval corpus entirely.
        ocr_sidecar = pdf_path.with_name(pdf_path.stem + "_ocr.txt")
        if ocr_sidecar.exists():
            ocr_text = ocr_sidecar.read_text(encoding="utf-8", errors="ignore")
            page_chunks = re.split(r"=====\s*PAGE\s+(\d+)\s*=====", ocr_text)
            # re.split with a capturing group yields [pre, num1, text1, num2, text2, ...]
            for i in range(1, len(page_chunks), 2):
                page_num = int(page_chunks[i])
                text = page_chunks[i + 1].strip()
                if not text:
                    continue
                result.append(
                    {
                        "text": text,
                        "page_num": page_num,
                        "source_file": pdf_path.name,
                        "file_path": str(pdf_path.resolve()),
                        "doc_type": doc_type,
                        "city": city,
                        "municipality": municipality.upper(),
                    }
                )
            if result:
                logger.info(
                    "  -> recovered %d page(s) for %s from OCR sidecar %s "
                    "(native extraction found none)",
                    len(result), pdf_path.name, ocr_sidecar.name,
                )

    return result


def parse_regulation_directory(
    municipality: str,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Parse every PDF (and .txt, for synthetic/smoke-test regulation text)
    under `settings.regulations_dir_for(municipality)`.
    """
    settings = settings or get_settings()
    reg_dir = settings.regulations_dir_for(municipality)
    all_pages: list[dict[str, Any]] = []

    if not reg_dir.exists():
        logger.warning("Regulations directory does not exist: %s", reg_dir)
        return all_pages

    pdf_files = sorted(reg_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        try:
            all_pages.extend(parse_pdf(pdf_path, municipality, settings=settings))
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pdf_path.name, exc)

    # Plain-text regulation files are supported too (e.g. a synthetic
    # smoke-test corpus, or byelaws already transcribed to .txt) so the
    # pipeline is exercisable without a real scanned PDF.
    txt_files = sorted(reg_dir.glob("*.txt"))
    for txt_path in txt_files:
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.error("Failed to read %s: %s", txt_path.name, exc)
            continue
        if not text:
            continue
        doc_type = _infer_doc_type(txt_path.stem, settings)
        city = _infer_city(txt_path.stem, settings)
        all_pages.append(
            {
                "text": text,
                "page_num": 1,
                "source_file": txt_path.name,
                "file_path": str(txt_path.resolve()),
                "doc_type": doc_type,
                "city": city,
                "municipality": municipality.upper(),
            }
        )

    logger.info(
        "Total pages parsed for %s: %d from %d files",
        municipality,
        len(all_pages),
        len(pdf_files) + len(txt_files),
    )
    return all_pages
