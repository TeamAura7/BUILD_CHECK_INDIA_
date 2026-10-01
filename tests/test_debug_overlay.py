from __future__ import annotations

from backend.cv_extraction.debug_overlay import render_debug_overlays
from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from tests.fixtures import pdf_builders


def test_render_debug_overlays_writes_one_png_per_page(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    extractor = PDFHybridExtractor()
    _result, bundle = extractor.extract_with_debug(pdf_path, document_id="doc-debug")

    out_dir = tmp_path / "debug_out"
    written = render_debug_overlays(pdf_path, bundle, out_dir)

    assert len(written) == 1
    assert written[0].exists()
    assert written[0].stat().st_size > 0


def test_render_debug_overlays_multi_page(tmp_path):
    pdf_path = pdf_builders.build_multi_page_size_pdf(tmp_path / "multi.pdf")
    extractor = PDFHybridExtractor()
    _result, bundle = extractor.extract_with_debug(pdf_path, document_id="doc-multi")

    out_dir = tmp_path / "debug_out"
    written = render_debug_overlays(pdf_path, bundle, out_dir)

    assert len(written) == 2
    for p in written:
        assert p.exists()
