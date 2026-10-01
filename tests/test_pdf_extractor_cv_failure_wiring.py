"""
Regression tests for a confirmed production bug: `PDFHybridExtractor.extract`
used to leave `ExtractionResult.independent_cv` as bare `None` whenever
`extract_independent_cv` raised (or the whole extraction timed out), which is
indistinguishable from "independent CV validation was never attempted" to
`backend.spatial_reasoning.final_fusion.apply_final_agreement_to_plan` --
that function used to short-circuit on `cv is None` and return the
pre-fusion plan completely unreconciled. `independent_cv` must instead be an
explicit `IndependentCVResult(status="FAILED", error=...)` so the failure is
visible and the finalization layer still runs (see
`tests/test_final_fusion.py`'s matching tests for the fusion-side half of
this fix).
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytest

from backend.cv_extraction.pdf_extractor import PDFHybridExtractor


def _build_minimal_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(60, 60, 550, 700), color=(0, 0, 0), width=1.0)
    page.insert_text((100, 40), "12.5 m", fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def test_independent_cv_failure_is_recorded_explicitly_not_left_none(tmp_path, monkeypatch):
    pdf_path = _build_minimal_pdf(tmp_path / "plan.pdf")

    def _boom(document_path, document_id):
        raise RuntimeError("synthetic independent-CV failure for this test")

    monkeypatch.setattr(
        "backend.cv_extraction.site_plan.extract_independent_cv", _boom
    )

    extractor = PDFHybridExtractor(enable_vision=False)
    result = extractor.extract(pdf_path, "doc-1")

    assert result.independent_cv is not None, (
        "independent_cv must never be left as bare None on a real CV "
        "failure -- that is indistinguishable from 'never attempted' and "
        "causes the final-fusion layer to bypass reconciliation entirely."
    )
    assert result.independent_cv.status == "FAILED"
    assert "synthetic independent-CV failure" in (result.independent_cv.error or "")
    assert any("independent CV validation path failed" in w for w in result.warnings)
