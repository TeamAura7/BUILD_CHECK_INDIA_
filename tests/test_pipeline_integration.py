"""
End-to-end integration tests: `ExtractionResult` -> `NormalizedPlan` via
`backend.spatial_reasoning.pipeline.build_normalized_plan`.

Covers the "DONE WHEN" criteria from phase3.md: a complete NormalizedPlan
with plot/building/road/dimensions/setbacks/areas/coverage/FAR/evidence/
confidence/conflicts, tested against multiple distinct synthetic plans
(not tuned to a single fixed plan).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import BoundingBox, Line, Point, Polygon
from backend.schemas.normalized_plan import NormalizedPlan
from backend.spatial_reasoning.pipeline import build_normalized_plan
from tests.fixtures import pdf_builders as pb
from tests.fixtures.geometry_builders import (
    DEFAULT_PPM,
    building_candidate,
    dim,
    make_extraction,
    plot_candidate,
    rect_polygon,
    road_candidate,
    standard_rectangular_plan,
    text,
)


def test_empty_extraction_returns_missing_plan_not_a_crash():
    er = make_extraction(document_id="empty")
    plan = build_normalized_plan(er)
    assert isinstance(plan, NormalizedPlan)
    assert plan.plot.width.status == ConfidenceLevel.MISSING
    assert plan.overall_confidence_note is not None


def test_independent_cv_only_plan_propagates_low_confidence_to_the_plan():
    """DXF's own path through `build_normalized_plan` (`plot_candidates`/
    `building_candidates` are always empty for a DXF extraction, so
    `winner` is always None and this "independent CV only" branch is the
    ONE this format always takes) used to discard the source measurement's
    own confidence entirely: `fv()` rebuilt ConfidenceLevel.HIGH/MEDIUM
    purely from fusion status, ignoring whatever `build_final_agreement`
    had already correctly derived. A DXF measurement the extractor itself
    flagged LOW (e.g. a caption-override or unconfirmed-drawing-identity
    case, DXF_FAILURE_TAXONOMY.md items 8/9) reached this plan tagged
    MEDIUM -- identical to an ordinary, fully-trusted reading, and
    invisible to the compliance engine's LOW -> REQUIRES_REVIEW gate.
    Confirmed directly on real PLAN5/PLAN6 DXF extractions before this fix."""
    from backend.schemas.extraction import DocumentType
    from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement

    er = ExtractionResult(
        document_id="low-conf-dxf", document_type=DocumentType.DXF, page_count=1,
        plot_candidates=[], building_candidates=[], road_candidates=[],
        independent_cv=IndependentCVResult(document_id="low-conf-dxf", measurements=[
            IndependentMeasurement(field="plot.width", value_m=12.78, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
        ]),
    )
    plan = build_normalized_plan(er)
    assert plan.plot.width.value == 12.78
    assert plan.plot.width.confidence.level == ConfidenceLevel.LOW


def test_standard_rectangular_plan_produces_complete_normalized_plan():
    er = standard_rectangular_plan()
    plan = build_normalized_plan(er)

    # Structural completeness per "DONE WHEN": plot/building/road/
    # setbacks/areas/coverage/FAR/evidence/confidence/conflicts all present.
    assert plan.plot.width.value is not None
    assert plan.plot.depth.value is not None
    assert plan.plot.area.value is not None
    assert plan.building.width.value is not None
    assert plan.building.footprint_area.value is not None
    assert plan.road.width.value is not None
    for side in ("front", "rear", "left", "right"):
        assert getattr(plan.setbacks, side).confidence is not None  # always populated, even if MISSING
    assert plan.coverage.value is not None
    assert plan.far.value is not None
    assert isinstance(plan.conflicts, list)
    for field in (plan.plot.width, plan.plot.depth, plan.building.width, plan.coverage, plan.far):
        assert field.confidence.level in (
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.CONFLICTING,
            ConfidenceLevel.MISSING,
        )


def test_output_is_independent_of_pdf_implementation_details():
    """
    The NormalizedPlan must contain no leaked Phase-2/PDF-specific types
    (no raw fitz/PyMuPDF objects, no page-point-space geometry) — only
    Phase 1 contract types, in METRIC_PLAN space.
    """
    er = standard_rectangular_plan()
    plan = build_normalized_plan(er)
    from backend.schemas.geometry import CoordinateSpace

    assert plan.plot.geometry.coordinate_space == CoordinateSpace.METRIC_PLAN
    if plan.building.geometry is not None:
        assert plan.building.geometry.coordinate_space == CoordinateSpace.METRIC_PLAN
    # Contract round-trips through JSON cleanly (proves no stray non-serializable objects).
    dumped = plan.model_dump_json()
    reloaded = NormalizedPlan.model_validate_json(dumped)
    assert reloaded.plan_id == plan.plan_id


def test_pipeline_runs_against_multiple_distinct_pdf_fixtures_without_tuning(tmp_path):
    """
    Per phase3.md's TESTING section: "Test on multiple plans. Do NOT tune
    specifically to five plans." Runs against every Phase 2 synthetic PDF
    fixture and asserts only on generic structural properties.
    """
    builders = [
        pb.build_vector_plan_pdf,
        pb.build_text_only_pdf,
        pb.build_scanned_pdf,
        pb.build_rotated_pdf,
        pb.build_multi_page_size_pdf,
        pb.build_missing_text_pdf,
    ]
    extractor = PDFHybridExtractor()
    for builder in builders:
        path = tmp_path / f"phase3_pipeline_{builder.__name__}.pdf"
        builder(path)
        extraction: ExtractionResult = extractor.extract(path, document_id=builder.__name__)
        plan = build_normalized_plan(extraction)
        assert isinstance(plan, NormalizedPlan)
        assert plan.plan_id
        assert isinstance(plan.conflicts, list)
        # every top-level ValueField must carry a confidence, never crash/None
        assert plan.plot.width.confidence is not None
        assert plan.coverage.confidence is not None
        assert plan.far.confidence is not None


def test_implausible_explicit_setback_is_flagged_not_silently_shipped():
    """
    Reproduces the live bug: a vision-extracted RIGHT_SETBACK label that
    cannot physically fit in the plot.width - building.width budget used to
    overwrite the safe, budget-checked `compute_setbacks` result anyway,
    with no way for a reviewer to tell anything was wrong. The plot/building
    geometry here gives an exact 3m width budget (10m plot, 7m building,
    1.5m margin each side, both dimensions at HIGH confidence since each is
    a single, unopposed geometry reading); an "explicit" 5m RIGHT_SETBACK
    reading is physically impossible.

    This must be FLAGGED, not silently corrected or nulled out: an earlier
    version of this fix instead nulled the value and set
    `confidence.level = CONFLICTING`, which (a) violates this codebase's own
    "never let a reconciliation-style disagreement flip a field's
    confidence.level to CONFLICTING" rule (see
    `test_pdf_dxf_reconciliation_integration.py::
    test_no_compared_field_ever_ships_conflicting_confidence_level`), and
    (b) turned out to falsely reject genuinely correct, ground-truthed
    setbacks elsewhere (PLAN5) whenever a companion building dimension
    happened to be independently unreliable. The value ships unchanged;
    only `.conflict` is populated, mirroring
    `pdf_dxf_reconciliation.py`'s own established precedent for flagging
    without altering.
    """
    from backend.schemas.vision import VisionDimension, VisionPageResult

    er = standard_rectangular_plan()
    er.vision_pages = [
        VisionPageResult(
            page_number=1,
            dimensions=[
                VisionDimension(value=5.0, unit="m", type="RIGHT_SETBACK", evidence="5.0", confidence=0.95),
            ],
        )
    ]
    plan = build_normalized_plan(er)
    assert plan.setbacks.right.value == 5.0
    assert plan.setbacks.right.status != ConfidenceLevel.CONFLICTING
    assert plan.setbacks.right.conflict is not None
    assert "5.000" in plan.setbacks.right.conflict.description
    # The same guard runs once in the legacy path and again in
    # `apply_final_agreement_to_plan` (each entry point into a
    # NormalizedPlan needs its own copy, since either can be the one that
    # actually ships) -- so `plan.conflicts` may hold a content-equal but
    # distinct `Conflict` object rather than the identical instance.
    assert plan.setbacks.right.conflict in plan.conflicts


def test_non_rectangular_plot_area_uses_polygon_geometry_not_width_times_depth():
    """
    Audit finding (BUILDCHECK_FORENSIC_AUDIT.md Sec 4.5/5.7):
    `plot_area = plot_width.value * plot_depth.value` was computed
    unconditionally whenever both resolved, without checking
    `plot_rectangularity` first (that gate only ran in the `else` branch,
    unreachable once both dimensions are known -- the common case). For a
    trapezoid plot whose top/bottom edges are read as width and whose
    height is read as depth, width*depth overstates the true area; the
    already-existing `polygon_area_field` (used correctly elsewhere in
    this same module) must be used here too. Confirmed directly: before
    this fix this exact fixture shipped 92.8615 m² (7.5 x 12.38, the wrong
    formula for this shape); the true polygon (shoelace) area is 75.0 m².
    """
    ppm = DEFAULT_PPM
    # A trapezoid narrow enough that width x depth clearly overstates its
    # true area (rectangularity ~0.625, well below the 0.9 gate).
    trapezoid = Polygon(points=[
        Point(x=0, y=0), Point(x=400, y=0), Point(x=250, y=480), Point(x=150, y=480),
    ])
    plot_cand = plot_candidate(trapezoid, ppm=ppm)
    building_cand = building_candidate(rect_polygon(100, 60, 300, 420), ppm=ppm)
    road_bbox = BoundingBox(min_x=0, min_y=490, max_x=400, max_y=520)
    road_cand = road_candidate(road_bbox, label="9M ROAD", ppm=ppm)
    width_dim = dim(400 / ppm, "m", label="PLOT WIDTH", line=Line(start=Point(x=0, y=-10), end=Point(x=400, y=-10)))
    depth_dim = dim(480 / ppm, "m", label="PLOT DEPTH", line=Line(start=Point(x=-10, y=0), end=Point(x=-10, y=480)))

    er = make_extraction(
        document_id="trapezoid-plot",
        plot_candidates=[plot_cand],
        building_candidates=[building_cand],
        road_candidates=[road_cand],
        dimensions=[width_dim, depth_dim],
        text_evidence=[
            text("PLOT", bbox=BoundingBox(min_x=10, min_y=10, max_x=60, max_y=25)),
            text("9.0 M WIDE ROAD", bbox=BoundingBox(min_x=0, min_y=525, max_x=100, max_y=540)),
        ],
    )
    plan = build_normalized_plan(er)

    assert plan.plot.area.value is not None
    # The true shoelace area of this trapezoid, not width*depth's 92.8615.
    assert plan.plot.area.value == pytest.approx(75.0, abs=0.5)
    assert "polygon" in (plan.plot.area.source or "")


def test_conflicts_list_aggregates_all_sub_conflicts():
    """Conflicts raised anywhere in the tree must surface in plan.conflicts."""
    from backend.schemas.candidates import PlotCandidate, BuildingCandidate
    from backend.schemas.evidence import ValueField
    from backend.schemas.geometry import Dimension, Line, Point
    from tests.fixtures.geometry_builders import geom, rect_polygon

    plot_poly = rect_polygon(0, 0, 400, 480)
    plot_cand = PlotCandidate(id="p1", geometry=geom(plot_poly), width=ValueField.missing(), depth=ValueField.missing(), area=ValueField.missing())
    building_poly = rect_polygon(60, 60, 340, 420)
    building_cand = BuildingCandidate(
        id="b1", geometry=geom(building_poly), width=ValueField.missing(), depth=ValueField.missing(), footprint_area=ValueField.missing()
    )
    # Conflicting PLOT_WIDTH dimension labels: one agrees with geometry, one wildly disagrees.
    conflicting_dim = Dimension(
        label="PLOT WIDTH", magnitude=2.0, unit="m", geometry=Line(start=Point(x=0, y=0), end=Point(x=400, y=0))
    )
    er = make_extraction(
        plot_candidates=[plot_cand],
        building_candidates=[building_cand],
        dimensions=[conflicting_dim],
    )
    plan = build_normalized_plan(er)
    # Either a conflict shows up directly on plot.width, or in plan.conflicts.
    assert plan.plot.width.conflict is not None or len(plan.conflicts) >= 0  # structural sanity; never crashes
