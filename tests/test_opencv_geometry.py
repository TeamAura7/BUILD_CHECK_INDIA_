from __future__ import annotations

from backend.cv_extraction import ocr_fallback, opencv_geometry, pdf_native
from backend.cv_extraction.raw_types import SourceKind
from tests.fixtures import pdf_builders


def test_detect_lines_on_vector_plan(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    image = ocr_fallback.rasterize_page(page, dpi=150)
    doc.close()

    lines = opencv_geometry.detect_lines(image, page_number=0, dpi=150)
    assert len(lines) > 0
    for line in lines:
        assert line.source == SourceKind.OPENCV_RASTER
        assert line.line.length > 0


def test_detect_contours_finds_rectangles(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    image = ocr_fallback.rasterize_page(page, dpi=150)
    doc.close()

    polygons, rectangles = opencv_geometry.detect_contours(image, page_number=0, dpi=150)
    assert len(polygons) > 0
    assert all(p.source == SourceKind.OPENCV_RASTER for p in polygons)
    # At least one of the rectangular outlines (plot/building/road strip) should be picked up.
    assert len(rectangles) >= 1


def test_geometry_evidence_for_page_returns_all_kinds(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    image = ocr_fallback.rasterize_page(page, dpi=150)
    doc.close()

    evidence = opencv_geometry.geometry_evidence_for_page(image, page_number=0, dpi=150)
    assert set(evidence.keys()) == {"lines", "polygons", "rectangles"}


def test_blank_page_yields_no_lines(tmp_path):
    import fitz

    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=300)
    doc.save(str(pdf_path))
    doc.close()

    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    image = ocr_fallback.rasterize_page(page, dpi=150)
    doc.close()

    lines = opencv_geometry.detect_lines(image, page_number=0, dpi=150)
    assert lines == []
