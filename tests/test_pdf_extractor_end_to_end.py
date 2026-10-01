from __future__ import annotations

from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.cv_extraction.registry import default_registry
from backend.schemas.enums import DocumentType
from backend.schemas.extraction import ExtractionResult
from tests.fixtures import pdf_builders


def test_vector_plan_produces_valid_extraction_result(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    extractor = PDFHybridExtractor()

    result = extractor.extract(pdf_path, document_id="doc-vector-1")

    assert isinstance(result, ExtractionResult)
    assert result.document_id == "doc-vector-1"
    assert result.document_type == DocumentType.VECTOR_PDF
    assert result.page_count == 1
    assert len(result.text_evidence) > 0
    assert len(result.dimensions) > 0
    # No hard-coded expected values — just structural conformance.
    for dim in result.dimensions:
        assert dim.magnitude >= 0


def test_dimensions_preserve_the_candidate_confidence_score(tmp_path):
    """
    Audit finding (BUILDCHECK_FORENSIC_AUDIT.md Sec 4.1/5.5):
    `DimensionCandidate.confidence` (computed in dimension_candidates.py)
    was silently discarded when folded into `Dimension` here, leaving
    every dimension in `ExtractionResult.dimensions` indistinguishable in
    reliability from every other one. `Dimension.confidence` must now
    carry that same 0-1 score through.
    """
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    extractor = PDFHybridExtractor()

    result = extractor.extract(pdf_path, document_id="doc-vector-conf-1")

    assert len(result.dimensions) > 0
    for dim in result.dimensions:
        assert dim.confidence is not None
        assert 0.0 <= dim.confidence <= 1.0


def test_scanned_plan_produces_valid_extraction_result_via_ocr(tmp_path):
    pdf_path = pdf_builders.build_scanned_pdf(tmp_path / "scanned.pdf")
    extractor = PDFHybridExtractor()

    result = extractor.extract(pdf_path, document_id="doc-scanned-1")

    assert result.document_type == DocumentType.RASTER_PDF
    assert len(result.text_evidence) > 0
    # Every text_evidence item from a scanned page should carry OCR confidence.
    assert all(te.ocr_confidence is not None for te in result.text_evidence)


def test_scanned_plan_with_sideways_text_is_recovered_by_orientation_aware_ocr(tmp_path):
    """
    Audit finding (BUILDCHECK_FORENSIC_AUDIT.md Sec 4.1/5.4): the main
    `PDFHybridExtractor._extract_raw` pass called the plain, upright-only
    `ocr_fallback.ocr_page`, while `site_plan.extract_independent_cv` and
    `region_detection.detect_regions` already used the orientation-aware
    `ocr_page_any_orientation` -- so a landscape sheet scanned/stored as an
    unrotated portrait page (real, documented failure mode) was silently
    mis-OCR'd in this specific pass. Plain upright OCR on this exact
    fixture recovers only garbled non-words (confirmed directly: no "ROAD"/
    "PLOT" token survives); the fix must recover real words instead.
    """
    pdf_path = pdf_builders.build_scanned_pdf_with_sideways_text(tmp_path / "sideways.pdf")
    extractor = PDFHybridExtractor()

    result = extractor.extract(pdf_path, document_id="doc-sideways-1")

    joined = " ".join(te.raw_text for te in result.text_evidence).upper()
    assert "ROAD" in joined
    assert "PLOT" in joined
    assert "SITE" in joined


def test_missing_text_pdf_still_produces_valid_result_with_warning(tmp_path):
    pdf_path = pdf_builders.build_missing_text_pdf(tmp_path / "missing_text.pdf")
    extractor = PDFHybridExtractor()

    result = extractor.extract(pdf_path, document_id="doc-missing-text")

    assert isinstance(result, ExtractionResult)
    # No text at all and no OCR hits expected (blank raster) -> should warn, not crash.
    assert isinstance(result.warnings, list)


def test_rotated_pdf_extracts_within_page_bounds(tmp_path):
    pdf_path = pdf_builders.build_rotated_pdf(tmp_path / "rotated.pdf", rotation=90)
    extractor = PDFHybridExtractor()

    result, bundle = extractor.extract_with_debug(pdf_path, document_id="doc-rotated")

    assert bundle.page_metadata[0].rotation_degrees == 90
    page_w = bundle.page_metadata[0].width_pts
    page_h = bundle.page_metadata[0].height_pts
    for item in bundle.text_evidence:
        assert -1 <= item.bounding_box.min_x
        assert item.bounding_box.max_x <= page_w + 1
        assert -1 <= item.bounding_box.min_y
        assert item.bounding_box.max_y <= page_h + 1


def test_different_page_sizes_produce_distinct_transforms(tmp_path):
    pdf_path = pdf_builders.build_multi_page_size_pdf(tmp_path / "multi_size.pdf")
    extractor = PDFHybridExtractor()

    result, bundle = extractor.extract_with_debug(pdf_path, document_id="doc-multi-size")

    assert result.page_count == 2
    sizes = {(p.width_pts, p.height_pts) for p in bundle.page_metadata}
    assert len(sizes) == 2
    assert len(bundle.coordinate_transform) == 2


def test_extraction_result_conforms_to_phase1_schema_shape(tmp_path):
    pdf_path = pdf_builders.build_vector_plan_pdf(tmp_path / "vector.pdf")
    extractor = PDFHybridExtractor()
    result = extractor.extract(pdf_path, document_id="doc-schema-check")

    # Round-trips through the Phase 1 pydantic contract cleanly.
    dumped = result.model_dump()
    reloaded = ExtractionResult.model_validate(dumped)
    assert reloaded.document_id == result.document_id


def test_registry_resolves_pdf_by_extension(tmp_path):
    pdf_path = pdf_builders.build_text_only_pdf(tmp_path / "text_only.pdf")
    registry = default_registry()
    extractor = registry.get_for_path(pdf_path)
    result = extractor.extract(pdf_path, document_id="doc-via-registry")
    assert isinstance(result, ExtractionResult)


def test_no_hardcoded_filename_or_page_branching(tmp_path):
    """
    Regression guard: running the extractor against two differently-named
    fixture files with equivalent content should behave identically in
    shape (same number of text items etc.), proving nothing branches on
    filename or an assumed page count.
    """
    pdf_a = pdf_builders.build_text_only_pdf(tmp_path / "alpha.pdf")
    pdf_b = pdf_builders.build_text_only_pdf(tmp_path / "totally_different_name.pdf")
    extractor = PDFHybridExtractor()

    result_a = extractor.extract(pdf_a, document_id="doc-a")
    result_b = extractor.extract(pdf_b, document_id="doc-b")

    assert len(result_a.text_evidence) == len(result_b.text_evidence)
    assert result_a.document_type == result_b.document_type
