from __future__ import annotations

from backend.cv_extraction import ocr_fallback, pdf_native
from tests.fixtures import pdf_builders


def test_needs_ocr_when_no_native_text():
    assert ocr_fallback.needs_ocr(has_native_text=False, native_text_char_count=0) is True


def test_does_not_need_ocr_when_native_text_present():
    assert ocr_fallback.needs_ocr(has_native_text=True, native_text_char_count=200) is False


def test_needs_ocr_can_be_forced():
    assert ocr_fallback.needs_ocr(has_native_text=True, native_text_char_count=200, force=True) is True


def test_scanned_pdf_triggers_ocr_and_finds_text(tmp_path):
    pdf_path = pdf_builders.build_scanned_pdf(tmp_path / "scanned.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)

    text_items = pdf_native.extract_text_items(page, 0)
    assert not pdf_native.has_sufficient_native_text(text_items)
    assert ocr_fallback.needs_ocr(False, 0) is True

    image = ocr_fallback.rasterize_page(page, dpi=200)
    ocr_items = ocr_fallback.ocr_page(image, page_number=0, dpi=200)
    doc.close()

    assert len(ocr_items) > 0
    joined = " ".join(item.text for item in ocr_items).upper()
    assert "ROAD" in joined or "PLAN" in joined
    for item in ocr_items:
        assert item.ocr_confidence is not None
        assert 0.0 <= item.ocr_confidence <= 1.0
        assert item.bounding_box.width >= 0


def test_vector_pdf_does_not_need_ocr(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    doc = pdf_native.open_document(pdf_path)
    page = doc.load_page(0)
    text_items = pdf_native.extract_text_items(page, 0)
    doc.close()

    assert ocr_fallback.needs_ocr(
        pdf_native.has_sufficient_native_text(text_items),
        sum(len(t.text) for t in text_items),
    ) is False
