from __future__ import annotations

from backend.cv_extraction import pdf_native
from tests.fixtures import pdf_builders


def test_vector_pdf_has_native_text(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector_plan.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    text_items = pdf_native.extract_text_items(page, 0)
    doc.close()

    assert len(text_items) > 0
    assert pdf_native.has_sufficient_native_text(text_items)
    assert any("ROAD" in t.text.upper() for t in text_items)


def test_vector_pdf_extracts_lines_and_rectangles(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector_plan.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    lines, rectangles, polygons = pdf_native.extract_vector_geometry(page, 0)
    doc.close()

    # Plot rect + building rect + road-strip rect should all surface as rectangles.
    assert len(rectangles) >= 2
    # The dimension line at the top of the plot should surface as a line.
    assert len(lines) >= 1
    for rect in rectangles:
        assert rect.bounding_box.width > 0
        assert rect.bounding_box.height > 0


def test_text_only_pdf_has_no_vector_geometry(tmp_path):
    pdf_path = pdf_builders.build_text_only_pdf(tmp_path / "text_only.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    text_items = pdf_native.extract_text_items(page, 0)
    lines, rectangles, polygons = pdf_native.extract_vector_geometry(page, 0)
    doc.close()

    assert len(text_items) > 0
    assert lines == []
    assert rectangles == []
    assert polygons == []


def test_missing_text_pdf_has_no_text_but_has_geometry(tmp_path):
    pdf_path = pdf_builders.build_missing_text_pdf(tmp_path / "missing_text.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    text_items = pdf_native.extract_text_items(page, 0)
    lines, rectangles, polygons = pdf_native.extract_vector_geometry(page, 0)
    doc.close()

    assert text_items == []
    assert not pdf_native.has_sufficient_native_text(text_items)
    assert len(lines) + len(rectangles) > 0


def test_scanned_pdf_has_no_native_text(tmp_path):
    pdf_path = pdf_builders.build_scanned_pdf(tmp_path / "scanned.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    text_items = pdf_native.extract_text_items(page, 0)
    doc.close()

    assert text_items == []
    assert not pdf_native.has_sufficient_native_text(text_items)


def test_different_page_sizes_are_preserved(tmp_path):
    pdf_path = pdf_builders.build_multi_page_size_pdf(tmp_path / "multi_size.pdf")
    doc = pdf_native.open_document(pdf_path)

    meta_0 = pdf_native.extract_page_metadata(doc.load_page(0), 0)
    meta_1 = pdf_native.extract_page_metadata(doc.load_page(1), 1)
    doc.close()

    assert (meta_0.width_pts, meta_0.height_pts) != (meta_1.width_pts, meta_1.height_pts)
    assert meta_0.width_pts == 595
    assert meta_1.width_pts == 612
