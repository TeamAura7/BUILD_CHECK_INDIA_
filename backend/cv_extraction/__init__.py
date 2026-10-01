"""
Phase 2 — hybrid PDF/vector/text/OCR/OpenCV architectural-plan extraction.

Public entry point: `PDFHybridExtractor` (implements `GeometryExtractor`
from `interfaces.py`). Everything else in this package is internal
plumbing (see `raw_types.py` for the pre-contract intermediate
representation).
"""

from __future__ import annotations

from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.cv_extraction.registry import ExtractorRegistry, default_registry

__all__ = ["PDFHybridExtractor", "ExtractorRegistry", "default_registry"]
