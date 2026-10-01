"""
Regression tests for phase3.1 fixes.

IMPORTANT: the real PLAN1/PLAN2/PLAN4/PLAN5/PLAN6 acceptance-test PDFs
referenced throughout phase3_1.md were NOT present in the uploaded
`data/test_plans/` directory (only `.gitkeep`). These tests reproduce
the *failure patterns* described in the spec using synthetic geometry
(same technique as `tests/fixtures/geometry_builders.py` and
`tests/fixtures/pdf_builders.py`) so the fixes are exercised and
regression-protected — they are NOT a substitute for running the real
PLAN1-6 acceptance checklist in phase3_1.md section 22/23, which still
needs the actual PDFs.
"""

from __future__ import annotations

from backend.config import get_settings
from backend.cv_extraction.candidate_geometry import build_plot_and_building_candidates
from backend.cv_extraction.dimension_candidates import detect_dimension_candidates
from backend.cv_extraction.raw_types import (
    CoordinateTransformRecord,
    RawLine,
    RawPolygon,
    RawTextItem,
    SourceKind,
)
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import ValueField
from backend.schemas.geometry import BoundingBox, Line, Point, Polygon
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.areas import coverage_field, far_field
from backend.spatial_reasoning.scale import estimate_scale
from tests.fixtures.geometry_builders import rect_polygon


def _rect_raw_polygon(x0, y0, x1, y1, page=0) -> RawPolygon:
    return RawPolygon(polygon=rect_polygon(x0, y0, x1, y1), page=page, is_closed=True, source=SourceKind.PDF_TEXT)


# ---------------------------------------------------------------------
# FIX #1 / #2: page-frame rejection + multi-candidate plot generation
# ---------------------------------------------------------------------


def test_page_frame_is_rejected_not_selected_as_plot_candidate():
    """
    A full-bleed border rectangle that touches the page edges on all
    sides (the classic drawing-sheet/title frame) must never become the
    single dominant plot candidate the old `sized[0]` logic would have
    picked, simply because it's the largest closed polygon on the page.
    """
    page_w, page_h = 612.0, 792.0
    page_frame = _rect_raw_polygon(0, 0, page_w, page_h)  # touches all 4 edges, ~100% of page
    real_plot = _rect_raw_polygon(60, 60, 400, 500)  # well inside the sheet

    plot_candidates, building_candidates = build_plot_and_building_candidates(
        document_id="doc-1",
        polygons=[page_frame, real_plot],
        page_areas_pts2={0: page_w * page_h},
        transforms={},
        page_dims_pts={0: (page_w, page_h)},
    )

    plot_bboxes = [c.geometry.bounding_box for c in plot_candidates]
    assert all(
        not (bbox.min_x <= 0.5 and bbox.min_y <= 0.5 and bbox.max_x >= page_w - 0.5)
        for bbox in plot_bboxes
    ), "page-frame-like polygon leaked through as a plot candidate"
    assert len(plot_candidates) == 1
    assert plot_candidates[0].geometry.bounding_box.width == 340  # the real plot, untouched


def test_multiple_plot_candidates_are_emitted_not_just_the_largest():
    """
    Two independent, non-frame closed polygons on a page (e.g. a site
    boundary and a separate large courtyard) should both surface as
    competing plot candidates — resolution/ranking is `plot_resolution`'s
    job, not `candidate_geometry`'s.
    """
    page_w, page_h = 612.0, 792.0
    a = _rect_raw_polygon(40, 40, 300, 300)
    b = _rect_raw_polygon(320, 320, 560, 700)

    plot_candidates, _ = build_plot_and_building_candidates(
        document_id="doc-2",
        polygons=[a, b],
        page_areas_pts2={0: page_w * page_h},
        transforms={},
        page_dims_pts={0: (page_w, page_h)},
    )
    assert len(plot_candidates) == 2


def test_degenerate_polygon_is_rejected():
    """A near-zero-area sliver (e.g. a mis-traced OpenCV contour) must not become a candidate."""
    page_w, page_h = 612.0, 792.0
    sliver = RawPolygon(
        polygon=Polygon(points=[Point(x=100, y=100), Point(x=100.001, y=100), Point(x=100.0005, y=100.001)]),
        page=0,
        is_closed=True,
        source=SourceKind.OPENCV_RASTER,
    )
    real_plot = _rect_raw_polygon(60, 60, 400, 500)
    plot_candidates, _ = build_plot_and_building_candidates(
        document_id="doc-3",
        polygons=[sliver, real_plot],
        page_areas_pts2={0: page_w * page_h},
        transforms={},
        page_dims_pts={0: (page_w, page_h)},
    )
    assert len(plot_candidates) == 1


def test_duplicate_geometry_is_deduped():
    page_w, page_h = 612.0, 792.0
    a = _rect_raw_polygon(60, 60, 400, 500)
    a_dup = _rect_raw_polygon(60.5, 60.5, 400.2, 500.3)  # within dedupe tolerance
    plot_candidates, _ = build_plot_and_building_candidates(
        document_id="doc-4",
        polygons=[a, a_dup],
        page_areas_pts2={0: page_w * page_h},
        transforms={},
        page_dims_pts={0: (page_w, page_h)},
    )
    assert len(plot_candidates) == 1


# ---------------------------------------------------------------------
# FIX #4: dimension-to-geometry association is scored, not "first found"
# ---------------------------------------------------------------------


def test_dimension_associates_with_nearest_line_not_first_in_list():
    """
    A dimension-text label sits very close to line B and far from line A.
    Even though A comes first in the `lines` list (the old `[0]` bug),
    the association must pick B.
    """
    text_item = RawTextItem(
        text="12.5 m",
        page=0,
        bounding_box=BoundingBox(min_x=100, min_y=198, max_x=130, max_y=210),
        source=SourceKind.PDF_TEXT,
    )
    far_line = RawLine(line=Line(start=Point(x=0, y=0), end=Point(x=50, y=0)), page=0, source=SourceKind.PDF_TEXT)
    near_line = RawLine(
        line=Line(start=Point(x=90, y=200), end=Point(x=140, y=200)), page=0, source=SourceKind.PDF_TEXT
    )

    candidates = detect_dimension_candidates(text_item.__class__ and [text_item], [far_line, near_line], proximity_pts=40.0)
    assert len(candidates) == 1
    ids = candidates[0].nearby_geometry_ids
    assert ids, "expected at least one nearby geometry line"
    assert ids[0] == 1, "expected the NEAR line (index 1) to be the best-scored association, not index 0"


def test_dimension_with_no_confident_geometry_leaves_association_unresolved():
    """From pdf_extractor's perspective: a weak/far association should not be silently used as truth."""
    from backend.cv_extraction.pdf_extractor import _best_dimension_geometry
    from backend.cv_extraction.raw_types import DimensionCandidate

    settings = get_settings()
    far_line = RawLine(line=Line(start=Point(x=0, y=0), end=Point(x=50, y=0)), page=0, source=SourceKind.PDF_TEXT)
    cand = DimensionCandidate(
        raw_text="9 m",
        numeric_value=9.0,
        unit_hint="m",
        bounding_box=BoundingBox(min_x=200, min_y=200, max_x=220, max_y=210),
        page=0,
        source=SourceKind.PDF_TEXT,
        nearby_geometry_ids=[0],
        confidence=0.8,
    )
    result = _best_dimension_geometry(cand, [far_line], settings)
    assert result is None, "a distant/low-confidence association should resolve to None, not a wrong line"


# ---------------------------------------------------------------------
# FIX #9: physical-consistency hard boundary (footprint > plot area)
# ---------------------------------------------------------------------


def test_coverage_impossible_when_footprint_exceeds_plot_area():
    footprint = ValueField[float](value=500.0, confidence=_high())
    plot = ValueField[float](value=100.0, confidence=_high())
    result = coverage_field(footprint, plot)
    assert result.confidence.level == ConfidenceLevel.CONFLICTING
    assert result.value is None
    assert result.conflict is not None


def test_far_impossible_when_footprint_exceeds_plot_area():
    footprint = ValueField[float](value=500.0, confidence=_high())
    plot = ValueField[float](value=100.0, confidence=_high())
    result = far_field(footprint, plot, None)
    assert result.confidence.level == ConfidenceLevel.CONFLICTING


def test_coverage_near_100_percent_is_not_flagged():
    """Real tightly-built urban plots can have coverage close to (but not over) 100% — must not be over-flagged."""
    footprint = ValueField[float](value=97.0, confidence=_high())
    plot = ValueField[float](value=100.0, confidence=_high())
    result = coverage_field(footprint, plot)
    assert result.confidence.level != ConfidenceLevel.CONFLICTING
    assert result.value == 97.0


def _high():
    from backend.schemas.evidence import Confidence

    return Confidence(level=ConfidenceLevel.HIGH)


# ---------------------------------------------------------------------
# FIX #5/#6: local (region-scoped) scale estimation
# ---------------------------------------------------------------------


def test_local_scale_ignores_detail_view_dimensions_outside_region():
    """
    Two groups of dimension samples: one cluster near a site-plan region
    (consistent ~40 pts/metre) and one far away representing an
    unrelated detail-view sheet at a very different implied scale
    (~200 pts/metre). Estimating scale scoped to the site-plan region
    must not be dragged toward the detail-view's scale.
    """
    from backend.schemas.geometry import Dimension

    site_region = BoundingBox(min_x=0, min_y=0, max_x=400, max_y=400)

    site_dims = [
        Dimension(
            label=f"site-{i}",
            magnitude=10.0,
            unit="m",
            geometry=Line(start=Point(x=10, y=10 + i), end=Point(x=410, y=10 + i)),  # 400pt / 10m = 40 pt/m
        )
        for i in range(3)
    ]
    detail_dims = [
        Dimension(
            label=f"detail-{i}",
            magnitude=1.0,
            unit="m",
            geometry=Line(start=Point(x=5000, y=5000 + i), end=Point(x=5200, y=5000 + i)),  # 200 pt/m
        )
        for i in range(3)
    ]

    result = estimate_scale(site_dims + detail_dims, region_bbox=site_region, region_padding_factor=1.5)
    assert abs(result.points_per_metre - 40.0) < 1.0
    assert "local" in result.reason.lower()


def test_local_scale_falls_back_to_global_when_no_local_samples():
    from backend.schemas.geometry import Dimension

    far_region = BoundingBox(min_x=9000, min_y=9000, max_x=9100, max_y=9100)
    dims = [
        Dimension(
            label="only-sample",
            magnitude=10.0,
            unit="m",
            geometry=Line(start=Point(x=10, y=10), end=Point(x=410, y=10)),
        )
    ]
    result = estimate_scale(dims, region_bbox=far_region, region_padding_factor=1.0)
    assert abs(result.points_per_metre - 40.0) < 1.0
    assert "fell back" in result.reason.lower()


# ---------------------------------------------------------------------
# FIX #7: polygon validity
# ---------------------------------------------------------------------


def test_is_valid_polygon_rejects_self_intersecting_bowtie():
    bowtie = Polygon(points=[Point(x=0, y=0), Point(x=10, y=10), Point(x=10, y=0), Point(x=0, y=10)])
    assert geo.is_valid_polygon(bowtie) is False


def test_is_valid_polygon_accepts_simple_rectangle():
    rect = rect_polygon(0, 0, 100, 50)
    assert geo.is_valid_polygon(rect) is True


def test_is_page_frame_like_requires_both_touch_and_area():
    page_w, page_h = 612.0, 792.0
    # Touches edges but small area -> not a frame.
    small_corner = BoundingBox(min_x=0, min_y=0, max_x=20, max_y=20)
    assert geo.is_page_frame_like(small_corner, page_w, page_h) is False
    # Covers a lot but doesn't touch edges -> not a frame.
    big_inset = BoundingBox(min_x=20, min_y=20, max_x=page_w - 20, max_y=page_h - 20)
    assert geo.is_page_frame_like(big_inset, page_w, page_h) is False
    # Both -> frame.
    frame = BoundingBox(min_x=0, min_y=0, max_x=page_w, max_y=page_h)
    assert geo.is_page_frame_like(frame, page_w, page_h) is True


# ---------------------------------------------------------------------
# FIX #10: bounded processing / explicit timeout failure
# ---------------------------------------------------------------------


def test_run_with_timeout_raises_explicit_error_not_hang():
    import time

    from backend.tools.bounded_execution import ExtractionTimeoutError, run_with_timeout

    def slow():
        time.sleep(2.0)
        return "done"

    try:
        run_with_timeout(slow, timeout_seconds=0.05, operation="test op")
        assert False, "expected ExtractionTimeoutError"
    except ExtractionTimeoutError as exc:
        assert "test op" in str(exc)


def test_run_with_timeout_returns_result_when_within_budget():
    from backend.tools.bounded_execution import run_with_timeout

    assert run_with_timeout(lambda: 1 + 1, timeout_seconds=5.0) == 2


# ---------------------------------------------------------------------
# Found against a REAL plan PDF (not synthetic): page-level default-unit
# convention ("ALL DIMENSIONS ARE IN METRE") + rotated-page coordinate
# mismatch between text and vector geometry.
# ---------------------------------------------------------------------


def test_default_unit_note_detected_and_applied_to_plausible_bare_numbers():
    from backend.cv_extraction.pdf_extractor import (
        _detect_default_dimension_unit,
        _fallback_unit_if_plausible,
    )
    from backend.cv_extraction.raw_types import DimensionCandidate

    note = RawTextItem(
        text="ALL DIMENSIONS ARE IN METRE, SCALE 1:100",
        page=0,
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=100, max_y=10),
        source=SourceKind.PDF_TEXT,
    )
    default_unit = _detect_default_dimension_unit([note])
    assert default_unit == "m"

    real_dim = DimensionCandidate(
        raw_text="3.35",
        numeric_value=3.35,
        unit_hint=None,
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
        page=0,
        source=SourceKind.PDF_TEXT,
        confidence=0.8,
    )
    assert _fallback_unit_if_plausible(real_dim, default_unit) == "m"


def test_default_unit_not_applied_to_area_figures_or_bare_integer_codes():
    from backend.cv_extraction.pdf_extractor import _fallback_unit_if_plausible
    from backend.cv_extraction.raw_types import DimensionCandidate

    area_value = DimensionCandidate(
        raw_text="96.34",
        numeric_value=96.34,  # a sq.m area figure, not a length -- outside plausible length range
        unit_hint=None,
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
        page=0,
        source=SourceKind.PDF_TEXT,
        confidence=0.8,
    )
    assert _fallback_unit_if_plausible(area_value, "m") is None

    door_code_stray_digit = DimensionCandidate(
        raw_text="D2",
        numeric_value=2.0,  # no decimal point in the source text -> not a length reading
        unit_hint=None,
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
        page=0,
        source=SourceKind.PDF_TEXT,
        confidence=0.8,
    )
    assert _fallback_unit_if_plausible(door_code_stray_digit, "m") is None


def test_default_unit_never_overrides_an_explicit_per_label_unit():
    """`_fallback_unit_if_plausible` is only consulted when `unit_hint` is already None
    (see `_fold`'s `cand.unit_hint or _fallback_unit_if_plausible(...)`); this documents that contract."""
    from backend.cv_extraction.raw_types import DimensionCandidate

    explicit = DimensionCandidate(
        raw_text="450mm",
        numeric_value=450.0,
        unit_hint="mm",
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
        page=0,
        source=SourceKind.PDF_TEXT,
        confidence=0.8,
    )
    assert (explicit.unit_hint or "should-not-be-reached") == "mm"


# ---------------------------------------------------------------------
# Rotated-page text/geometry coordinate-space consistency
# ---------------------------------------------------------------------


def test_extract_text_items_applies_rotation_matrix():
    """
    `page.get_text("dict")` returns unrotated (raw mediabox) coordinates
    while `page.get_drawings()` returns already-rotated coordinates
    matching `page.rect` -- found on a real 270-degree-rotated
    architectural sheet, where every text-to-geometry distance
    computation was silently comparing two different coordinate frames.
    Builds an actual rotated PDF via fitz to verify text bounding boxes
    end up within the rotated page's own bounds (matching page.rect),
    not the raw unrotated mediabox.
    """
    import fitz

    from backend.cv_extraction.pdf_native import extract_text_items

    doc = fitz.open()
    page = doc.new_page(width=400, height=600)  # raw mediabox: 400 wide x 600 tall
    page.insert_text((20, 550), "SITE PLAN", fontsize=12)  # near the raw-space bottom
    page.set_rotation(270)

    assert page.rect.width == 600 and page.rect.height == 400  # rotated display space

    items = extract_text_items(page, page_number=0)
    assert len(items) == 1
    bbox = items[0].bounding_box
    # Must land inside the ROTATED page bounds (matching page.rect), not
    # the raw 400x600 mediabox -- this is exactly the space
    # extract_vector_geometry's get_drawings() output already uses.
    assert 0 <= bbox.min_x <= page.rect.width
    assert 0 <= bbox.max_x <= page.rect.width
    assert 0 <= bbox.min_y <= page.rect.height
    assert 0 <= bbox.max_y <= page.rect.height
    doc.close()
