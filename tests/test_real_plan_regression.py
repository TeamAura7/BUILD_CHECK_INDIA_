"""
Regression tests against the REAL sanctioned plans in `data/test_plans/`.

The rest of the suite runs almost entirely against synthetic PDFs built by
`tests/fixtures/pdf_builders.py`. Those fixtures are drawn the way the
extractor expects plans to be drawn, so they passed (271/271) throughout a
period when 4 of the 5 real bundled plans returned null for every geometric
field, and the one plan that did resolve returned plot.width = 40.64 m for a
plot 10.00 m wide.

These tests assert against values the plans state about THEMSELVES -- the
area statement printed on the sheet -- rather than against numbers copied
out of a previous run of this code. That distinction matters: an acceptance
value harvested from the implementation only pins current behaviour, whereas
"the reconstructed plot boundary must reproduce the area the drawing says
the plot has" is a property the extraction has to earn.

Tolerances are relative and deliberately loose enough to absorb where a
boundary's centreline sits inside its stroke width, and tight enough that
the failures this suite was written for (an order-of-magnitude wrong scale,
a caption read as a dimension, the wrong rectangle chosen) cannot pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.schemas.enums import ConfidenceLevel
from backend.cv_extraction.scale_note import (
    detect_scale_notes,
    nearest_scale_note,
    points_per_metre_for_denominator,
)
from backend.cv_extraction.site_plan import extract_independent_cv

PLANS_DIR = Path(__file__).parent.parent / "data" / "test_plans"

# Ground truth, read off the sheets by eye and corroborated by each sheet's
# own printed area statement.
PLAN2 = {
    "plot.width": 12.192,   # 40 ft
    "plot.depth": 18.28,    # 60 ft
    "road.width": 9.2,
    "setbacks.front": 1.0,
    "setbacks.rear": 0.8,
    "setbacks.left": 0.8,
    "setbacks.right": 0.8,
    "stated_plot_area": 222.83,
    "stated_footprint_area": 174.52,
}
PLAN6 = {
    "plot.width": 10.00,
    "plot.depth": 13.10,
    "road.width": 7.3,
    "stated_plot_area": 131.00,       # "AREA OF PLOT (Minimum)"
    "stated_net_plot_area": 107.50,   # after the 10.00 x 2.35 road-widening strip
    "stated_footprint_area": 85.09,   # "Proposed Coverage Area"
}


def _measurements(plan_filename: str) -> dict[str, float]:
    pdf_path = PLANS_DIR / plan_filename
    if not pdf_path.exists():
        pytest.skip(f"{plan_filename} fixture not present")
    result = extract_independent_cv(pdf_path, plan_filename)
    return {
        m.field: (m.value_m if m.value_m is not None else m.value)
        for m in result.measurements
    }


def _assert_close(actual: float | None, expected: float, field: str, tolerance: float = 0.02):
    assert actual is not None, f"{field} was not resolved at all (None)"
    error = abs(actual - expected) / expected
    assert error <= tolerance, (
        f"{field}: got {actual:.4f}, expected ~{expected} ({error:.2%} off, "
        f"tolerance {tolerance:.0%})"
    )


# --- Printed scale notes ----------------------------------------------------


def test_points_per_metre_matches_the_physical_definition_of_a_drawing_scale():
    # 1pt = 1/72in = 25.4/72 mm on paper; at 1:200 that is 70.5556 mm real,
    # so 1 m contains 1000/70.5556 = 14.1732 pt.
    assert points_per_metre_for_denominator(200) == pytest.approx(14.17323, abs=1e-4)
    assert points_per_metre_for_denominator(100) == pytest.approx(28.34646, abs=1e-4)
    # Halving the denominator doubles the points per metre.
    assert points_per_metre_for_denominator(50) == pytest.approx(
        2 * points_per_metre_for_denominator(100)
    )


def test_cement_mortar_mix_ratios_are_not_read_as_drawing_scales():
    """
    "1:6" in "0.15th in C.M 1:6" is a cement-mortar proportion. Read as a
    drawing scale it yields 472 pt/m instead of 14 pt/m -- a 33x error that
    would shrink every derived dimension by the same factor. Real working
    drawings carry several of these in their specification notes.
    """
    from backend.cv_extraction.raw_types import RawTextItem, SourceKind
    from backend.schemas.geometry import BoundingBox

    def item(text: str) -> RawTextItem:
        return RawTextItem(
            text=text,
            bounding_box=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10),
            page=0,
            source=SourceKind.PDF_TEXT,
        )

    mix_notes = [
        item("0.15th in C.M 1:6"),
        item("BRICK WORK IN CM 1:5"),
        item("P.C.C in mix 1:5:10"),
        item("FLOORING CONCRETE 1:5:10"),
    ]
    assert detect_scale_notes(mix_notes, 0) == []

    # ... while a genuine scale note beside them is still found.
    found = detect_scale_notes([*mix_notes, item("Scale 1:200")], 0)
    assert [n.denominator for n in found] == [200]


def test_real_sheets_expose_their_printed_scale():
    from backend.cv_extraction import pdf_native

    for filename, expected_denominator in (("PLAN2.pdf", 200), ("PLAN6.pdf", 200)):
        pdf_path = PLANS_DIR / filename
        if not pdf_path.exists():
            pytest.skip(f"{filename} fixture not present")
        doc = pdf_native.open_document(pdf_path)
        try:
            page = doc.load_page(0)
            notes = detect_scale_notes(pdf_native.extract_text_items(page, 0), 0)
        finally:
            doc.close()
        assert notes, f"{filename}: no printed scale note found"
        assert {n.denominator for n in notes} == {expected_denominator}, (
            f"{filename}: expected only 1:{expected_denominator}, got "
            f"{sorted(n.denominator for n in notes)}"
        )


def test_multi_view_sheet_keeps_every_distinct_view_scale():
    """
    PLAN4 prints four different view scales. Collapsing them to one would be
    wrong -- which governs the site plan is a spatial question, so all of
    them must survive detection.
    """
    from backend.cv_extraction import pdf_native

    pdf_path = PLANS_DIR / "PLAN4.pdf"
    if not pdf_path.exists():
        pytest.skip("PLAN4.pdf fixture not present")
    doc = pdf_native.open_document(pdf_path)
    try:
        page = doc.load_page(0)
        notes = detect_scale_notes(pdf_native.extract_text_items(page, 0), 0)
    finally:
        doc.close()
    assert {n.denominator for n in notes} == {25, 50, 75, 100}
    # ...and the mix ratios on the same sheet are still excluded.
    assert 5 not in {n.denominator for n in notes}
    # None of these four actually governs PLAN4's site plan, which is drawn
    # to fit rather than to a stated scale -- so detecting them all is
    # necessary but not sufficient, and the resolver has to fall back to
    # scale derived from the site plan's own edge labels.


# --- PLAN2: the plan that already resolved, and must not regress ------------


def test_plan2_still_resolves_every_geometric_field():
    got = _measurements("PLAN2.pdf")
    for field in (
        "plot.width", "plot.depth", "road.width",
        "setbacks.front", "setbacks.rear", "setbacks.left", "setbacks.right",
    ):
        _assert_close(got.get(field), PLAN2[field], field)
    assert got.get("building.width") is not None
    assert got.get("building.depth") is not None


def test_plan2_building_footprint_reproduces_its_stated_coverage_area():
    got = _measurements("PLAN2.pdf")
    measured = got["building.width"] * got["building.depth"]
    _assert_close(measured, PLAN2["stated_footprint_area"], "building footprint area", 0.01)


# --- PLAN6: the plan that returned nothing ----------------------------------


def test_plan6_resolves_plot_dimensions_with_no_printed_edge_labels():
    """
    PLAN6's site plan carries NO printed plot-dimension label -- the only
    text near the drawing is "SITE NO-07/08/09", "7.30m Wide Road" and
    "Scale 1:200". Every geometric field came back null because the resolver
    required an edge label to establish scale. They are recoverable from the
    printed scale alone.
    """
    got = _measurements("PLAN6.pdf")
    _assert_close(got.get("plot.width"), PLAN6["plot.width"], "plot.width")
    _assert_close(got.get("plot.depth"), PLAN6["plot.depth"], "plot.depth")
    _assert_close(got.get("road.width"), PLAN6["road.width"], "road.width")


def test_plan6_plot_rectangle_reproduces_its_stated_plot_area():
    """The closed loop: geometry and the area statement are independent."""
    got = _measurements("PLAN6.pdf")
    measured = got["plot.width"] * got["plot.depth"]
    _assert_close(measured, PLAN6["stated_plot_area"], "plot area", 0.01)


def test_plan6_building_footprint_reproduces_its_stated_coverage_area():
    got = _measurements("PLAN6.pdf")
    assert got.get("building.width") is not None, "building.width not resolved"
    measured = got["building.width"] * got["building.depth"]
    _assert_close(measured, PLAN6["stated_footprint_area"], "building footprint area", 0.01)


def test_plan6_resolves_setbacks_including_a_zero_setback_edge():
    """
    PLAN6's building abuts the plot boundary on one side. A nesting rule
    that required a positive gap on all four sides rejected the real
    footprint outright, taking building.width/depth and all four setbacks
    down with it.
    """
    got = _measurements("PLAN6.pdf")
    for side in ("front", "rear", "left", "right"):
        assert got.get(f"setbacks.{side}") is not None, f"setbacks.{side} not resolved"

    # The four setbacks must account for exactly the difference between the
    # plot and the building on each axis -- they are not independent numbers.
    across = got["setbacks.left"] + got["setbacks.right"] + got["building.width"]
    along = got["setbacks.front"] + got["setbacks.rear"] + got["building.depth"]
    _assert_close(across, got["plot.width"], "left + building.width + right", 0.01)
    _assert_close(along, got["plot.depth"], "front + building.depth + rear", 0.01)


def test_plan6_gross_and_net_plot_area_are_separate_fields():
    """
    A BBMP sheet states both an "AREA OF PLOT" and a "NET AREA OF PLOT"
    (gross minus road widening). Emitting both as `plot.area` produced two
    contradictory values for one field, and whichever was read last won.
    """
    got = _measurements("PLAN6.pdf")
    _assert_close(got.get("plot.area"), PLAN6["stated_plot_area"], "plot.area", 0.005)
    _assert_close(
        got.get("plot.net_area"), PLAN6["stated_net_plot_area"], "plot.net_area", 0.005
    )


# --- The failure mode that started all of this ------------------------------


def test_prose_and_identifiers_are_not_dimension_candidates():
    """
    Every string here was extracted from a real sheet as a `Dimension` with
    the shown magnitude, and each one then became a scale sample.
    """
    from backend.cv_extraction.dimension_candidates import looks_like_dimension_text

    not_dimensions = [
        "46.Due to non-compliance of safety precautionary measures",
        "3.Car Parking reserved in the plan should not be converted",
        "PID No. (As per Khata Extract): 1234567890",
        "Permissible F.A.R. as per zoning regulation 2015 ( 1.75 )",
        "Ward: Ward 187",
        "ISO_A1_(841.00_x_594.00_MM)",
        "Planning District: 999-Sampletown",
        "Project No: ABC/XYZ/0001/25-26",
        "VERSION DATE: 30/03/2026",
    ]
    for text in not_dimensions:
        assert not looks_like_dimension_text(text), f"should be rejected: {text!r}"

    real_dimensions = ["3.35", "0.91", "12.19", "7.30m", "10.00", "1.20", "9'-6\"", "2400mm"]
    for text in real_dimensions:
        assert looks_like_dimension_text(text), f"should be accepted: {text!r}"


def test_real_sheet_produces_far_fewer_but_better_dimension_candidates():
    from backend.cv_extraction import pdf_native
    from backend.cv_extraction.dimension_candidates import detect_dimension_candidates

    pdf_path = PLANS_DIR / "PLAN6.pdf"
    if not pdf_path.exists():
        pytest.skip("PLAN6.pdf fixture not present")
    doc = pdf_native.open_document(pdf_path)
    try:
        page = doc.load_page(0)
        text_items = pdf_native.extract_text_items(page, 0)
    finally:
        doc.close()

    candidates = detect_dimension_candidates(text_items, [])
    # Was 366 on this sheet, from 748 text spans.
    assert len(candidates) < 300, f"{len(candidates)} candidates -- prose is leaking back in"
    # No candidate may carry a magnitude that could not be a length.
    for candidate in candidates:
        assert candidate.numeric_value <= 1000.0, (
            f"implausible magnitude {candidate.numeric_value} from {candidate.raw_text!r}"
        )


# --- Abstention: a wrong measurement is worse than a missing one ------------


# PLAN4: rotated page, feet-and-inches dimensions, stacked dimension chains,
# and no printed scale that applies to the site plan.
PLAN4 = {
    "plot.width": 18.288,        # 60'-0"
    "plot.depth": 16.6116,       # 54'-6"
    "building.width": 16.459,    # 54'-0"
    "building.depth": 11.2776,   # 37'-0"
    # Its road runs down the LEFT of the site plan, so the 3'-0" gaps are
    # front/rear and the 8'-9" gaps are the sides.
    "setbacks.front": 0.9144,
    "setbacks.rear": 0.9144,
    "setbacks.left": 2.667,
    "setbacks.right": 2.667,
    "road.width": 7.62,          # "25 FEET ROAD"
    "stated_plot_area": 303.79,
    "stated_footprint_area": 185.62,
}


def test_plan4_resolves_a_rotated_sheet_dimensioned_in_feet_and_inches():
    """
    PLAN4 needed four separate things to be right at once:

    - Its page is /Rotate 270, and vector geometry was being left in the
      unrotated mediabox while text was transformed into display space, so
      the two were a quarter turn apart and nothing matched anything.
    - Its dimensions are feet-and-inches with a hyphen (8'-9"), which parsed
      as a bare 8 feet.
    - Its site plan is dimensioned as a stacked chain (3'-0" | 54'-0" | 3'-0"
      under an overall 60'-0"), so the label nearest the plot edge is a
      setback, not the plot width.
    - None of the four scales it prints applies to the site plan, so scale
      has to come from its own edge labels.
    """
    got = _measurements("PLAN4.pdf")
    for field in (
        "plot.width", "plot.depth", "building.width", "building.depth",
        "setbacks.front", "setbacks.rear", "setbacks.left", "setbacks.right",
        "road.width",
    ):
        _assert_close(got.get(field), PLAN4[field], field)


def test_plan4_geometry_reproduces_both_stated_areas():
    got = _measurements("PLAN4.pdf")
    _assert_close(
        got["plot.width"] * got["plot.depth"], PLAN4["stated_plot_area"], "plot area", 0.01
    )
    _assert_close(
        got["building.width"] * got["building.depth"],
        PLAN4["stated_footprint_area"], "footprint area", 0.01,
    )


def test_road_width_honours_the_unit_it_is_printed_in():
    """PLAN4 states "25 FEET ROAD"; read as 25 m that is a highway."""
    got = _measurements("PLAN4.pdf")
    _assert_close(got.get("road.width"), 7.62, "road.width")


@pytest.mark.parametrize(
    "plan_filename, why",
    [
        (
            "PLAN7.pdf",
            "is a photograph of a blueprint -- no vector geometry at all, no site plan "
            "region, and OCR that loses decimal points ('4 29X3 20' for '4.29X3.20')",
        ),
    ],
)
def test_sheets_without_a_site_plan_resolve_nothing_rather_than_guessing(plan_filename, why):
    """
    A sheet always contains *some* best-scoring rectangle near whatever text
    the anchor matched. Asserting it as the plot produces a confident wrong
    answer, which is strictly worse than MISSING here: the compliance engine
    maps MISSING to INSUFFICIENT_DATA and asks for review, but treats a
    present value as measured fact and will PASS or FAIL a real building on
    it.
    """
    got = _measurements(plan_filename)
    for field in (
        "plot.width", "plot.depth", "building.width", "building.depth",
        "setbacks.front", "setbacks.rear", "setbacks.left", "setbacks.right",
    ):
        assert got.get(field) is None, (
            f"{plan_filename} {why}, so {field} must be unresolved, got {got[field]}"
        )


def test_plan7_via_real_production_extractor_never_ships_confident_wrong_geometry():
    """
    The test above exercises `site_plan.extract_independent_cv` directly --
    a correctly-abstaining resolver, but NOT the class `analyze.py` (the
    real production entrypoint -- see its own module docstring: "Pdf
    HybridExtractor -> build_normalized_plan -> JsonFileRuleEngine")
    actually instantiates.

    Confirmed live (before the fix this test pins): `PDFHybridExtractor`'s
    own separate "legacy candidate resolution" path
    (`candidate_geometry.build_plot_and_building_candidates`, scored by
    `plot_resolution.score_plot_candidates`, which always returns SOME
    winner whenever any candidate exists at all) produced spurious plot/
    building candidates on this exact photograph and shipped them at
    `ConfidenceLevel.HIGH` -- confidently wrong, on the exact adversarial
    file class this project's own tests exist to catch. Every existing
    regression test (including the one directly above, and
    `eval_harness.py`'s own "never confidently wrong" gate) exercised only
    the bypass path and missed this entirely. Fixed via
    `plot_resolution.plot_confidence`'s absolute score floor
    (`_MIN_USABLE_PLOT_SCORE`) and `pipeline.build_normalized_plan`'s
    confidence cap (`_cap_field_confidence`, built on
    `enums.cap_confidence_level`) -- see ARCHITECTURE_V2.md's
    Implementation log for the full investigation.

    This does NOT assert the value is `None` (unlike the test above): the
    legacy candidate path still finds *some* geometric candidate on this
    file, and nothing here makes it stop looking -- that would require
    `score_plot_candidates` itself to refuse a winner outright, which is
    deliberately not done (see `test_plot_resolution.py::
    test_plot_confidence_high_with_clear_margin`, an existing, deliberate
    test asserting a lone weak candidate is still returned as winner, just
    at LOW confidence). What this asserts is the actually load-bearing
    safety property: a document with no real site plan must never ship a
    HIGH/MEDIUM-confidence geometric value through the real production
    path, because `backend/compliance/engine.py` only routes LOW confidence
    to REQUIRES_REVIEW -- a MEDIUM or HIGH value here would silently
    PASS/FAIL a real compliance check against a fabricated number.
    """
    from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
    from backend.spatial_reasoning.pipeline import build_normalized_plan

    pdf_path = PLANS_DIR / "PLAN7.pdf"
    if not pdf_path.exists():
        pytest.skip("PLAN7.pdf fixture not present")

    extraction = PDFHybridExtractor(enable_vision=False).extract(pdf_path, "PLAN7")
    plan = build_normalized_plan(extraction, plan_id="PLAN7")

    for field_name, field in (
        ("plot.width", plan.plot.width), ("plot.depth", plan.plot.depth),
        ("building.width", plan.building.width), ("building.depth", plan.building.depth),
    ):
        if field.value is None:
            continue  # MISSING is the ideal outcome and trivially satisfies this test
        assert field.confidence.level == ConfidenceLevel.LOW, (
            f"PLAN7 {field_name} shipped a value ({field.value}) through the real production "
            f"extractor at {field.confidence.level}, not LOW -- a confident wrong answer on a "
            "document with no real site plan would silently bypass the compliance engine's "
            "LOW -> REQUIRES_REVIEW safety gate."
        )


# PLAN5: rotated sheet, rasterised text, DASHED plot boundary.
PLAN5 = {
    "plot.width": 17.59,
    "plot.depth": 9.14,
    "road.width": 10.0,
    "stated_plot_area": 160.77,   # "SITE AREA : 160.77 Sq.m"
}


def test_plan5_resolves_a_dashed_plot_boundary_on_a_rotated_rasterised_sheet():
    """
    PLAN5 is the hard case and it failed for three independent reasons at
    once, each of which alone was fatal:

    1. Its plot boundary is a DASH-DOT property line -- 71 separate segments
       on the top edge, longest 8.9pt -- and line clustering required each
       segment to be at least 35pt, so every one was discarded.
    2. Its text is rasterised and the whole sheet is rotated 90 degrees (437
       of 522 OCR items are vertical), so the "SITE PLAN" caption is taller
       than it is wide and the search window, which assumed an upright
       caption with its drawing above, landed on the area-statement table.
    3. Its area statement says "SITE AREA : 160.77 Sq.m", which matched none
       of the BBMP-worded patterns, leaving nothing to cross-check against.
    """
    got = _measurements("PLAN5.pdf")
    _assert_close(got.get("plot.width"), PLAN5["plot.width"], "plot.width")
    _assert_close(got.get("plot.depth"), PLAN5["plot.depth"], "plot.depth")
    _assert_close(got.get("road.width"), PLAN5["road.width"], "road.width", 0.05)
    measured = got["plot.width"] * got["plot.depth"]
    _assert_close(measured, PLAN5["stated_plot_area"], "plot area", 0.02)


def test_plan5_resolves_building_and_setbacks_from_a_scanned_rotated_sheet():
    """
    The building footprint on PLAN5 is a native PDF `re` rectangle measuring
    13.09 x 7.14 m -- its exact printed dimensions. The resolver read only
    stroked line segments and discarded PyMuPDF's rectangle output entirely,
    so the single most reliable object on the drawing never reached matching.

    Its setbacks additionally needed the front side to be worked out from
    where the road is, rather than assumed to be the bottom of the page: this
    sheet is drawn rotated, its road is on the left, and all four setback
    values were previously correct but labelled a quarter-turn out.
    """
    got = _measurements("PLAN5.pdf")
    _assert_close(got.get("building.width"), 13.09, "building.width")
    _assert_close(got.get("building.depth"), 7.14, "building.depth")
    _assert_close(got.get("setbacks.front"), 3.00, "setbacks.front")
    _assert_close(got.get("setbacks.rear"), 1.50, "setbacks.rear")
    # The side setbacks absorb the ~7cm of slack in the dashed boundary's
    # measured depth, so they carry a looser tolerance than the others.
    _assert_close(got.get("setbacks.left"), 1.00, "setbacks.left", 0.10)
    _assert_close(got.get("setbacks.right"), 1.00, "setbacks.right", 0.10)


def test_setback_sides_follow_the_road_not_the_page():
    """
    Which side is "front" is defined by the road, and left/right are read
    from someone standing on the road looking in. On an upright sheet this
    is the obvious mapping; on a sheet drawn rotated it is not.
    """
    from backend.cv_extraction.site_plan import (
        _front_side_from_road,
        _setback_labels_for_front,
    )
    from backend.schemas.geometry import BoundingBox

    plot = BoundingBox(min_x=100, min_y=100, max_x=300, max_y=200)
    below = BoundingBox(min_x=150, min_y=230, max_x=250, max_y=250)
    left_of = BoundingBox(min_x=40, min_y=130, max_x=60, max_y=170)
    assert _front_side_from_road(plot, [below]) == "bottom"
    assert _front_side_from_road(plot, [left_of]) == "left"
    # With no road label at all, fall back to the common upright layout.
    assert _front_side_from_road(plot, []) == "bottom"

    upright = _setback_labels_for_front("bottom")
    assert upright == {
        "bottom": "setbacks.front", "top": "setbacks.rear",
        "left": "setbacks.left", "right": "setbacks.right",
    }
    rotated = _setback_labels_for_front("left")
    assert rotated == {
        "left": "setbacks.front", "right": "setbacks.rear",
        "top": "setbacks.left", "bottom": "setbacks.right",
    }
    # Every side gets exactly one label, whichever way the sheet is turned.
    for front in ("top", "right", "bottom", "left"):
        assert sorted(_setback_labels_for_front(front).values()) == [
            "setbacks.front", "setbacks.left", "setbacks.rear", "setbacks.right",
        ]


def test_native_rectangle_operators_are_candidate_geometry():
    """A CAD export draws some rectangles as `re`, not as four line strokes."""
    from backend.cv_extraction.raw_types import RawRectangle, SourceKind
    from backend.cv_extraction.site_plan import _candidate_rectangles
    from backend.schemas.geometry import BoundingBox

    region = BoundingBox(min_x=0, min_y=0, max_x=500, max_y=500)
    native = RawRectangle(
        bounding_box=BoundingBox(min_x=100, min_y=100, max_x=300, max_y=250),
        page=0,
        source=SourceKind.VECTOR_PDF,
    )
    found = _candidate_rectangles([], region, [native])
    assert len(found) == 1
    assert found[0].bbox.width == 200 and found[0].bbox.height == 150

    # Slivers and extreme aspect ratios are still rejected.
    sliver = RawRectangle(
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=400, max_y=5),
        page=0,
        source=SourceKind.VECTOR_PDF,
    )
    assert _candidate_rectangles([], region, [sliver]) == []


def test_dashed_sides_are_accepted_by_span_not_by_inked_fraction():
    """
    A solid edge inks ~100% of its length; PLAN5's dash-dot plot boundary
    inks 41%. No single coverage threshold separates the dashed edge from
    collinear noise, so a dashed side is recognised by its marks spanning the
    whole side instead.
    """
    from backend.cv_extraction.site_plan import _side_is_drawn

    solid = [(0.0, 98.0)]
    assert _side_is_drawn(0.0, 100.0, solid)

    # ~40% inked, but the marks run corner to corner, as a real dash-dot
    # boundary does.
    dashed = [(float(i), i + 4.0) for i in range(0, 101, 10)]
    assert _side_is_drawn(0.0, 100.0, dashed)

    # Dashes that stop well short of the far corner are not a full side,
    # even though the pattern looks the same.
    assert not _side_is_drawn(0.0, 100.0, [(float(i), i + 4.0) for i in range(0, 70, 10)])

    # Two ticks at the corners span the side but are not a drawn edge.
    assert not _side_is_drawn(0.0, 100.0, [(0.0, 3.0), (97.0, 100.0)])
    # A line that stops half way is not a side, however solid.
    assert not _side_is_drawn(0.0, 100.0, [(0.0, 48.0)])


def test_a_side_must_reach_both_corners_not_just_cover_most_of_its_length():
    """
    Coverage alone accepted a rectangle stitched from two different objects'
    lines: each "side" inked >80% of the interval it stood in for, but only
    because it started 13-19% in from a corner that some OTHER object's line
    occupies. A genuine drawn edge runs corner to corner.
    """
    from backend.cv_extraction.site_plan import _side_is_drawn

    assert _side_is_drawn(0.0, 100.0, [(0.0, 100.0)])
    # Line weight noise at the corners is tolerated.
    assert _side_is_drawn(0.0, 100.0, [(2.0, 98.0)])
    # 87% inked, but the missing 13% is the corner end -- not a side.
    assert not _side_is_drawn(0.0, 100.0, [(13.0, 100.0)])
    assert not _side_is_drawn(0.0, 100.0, [(0.0, 87.0)])
    # A line that extends PAST the corner still reaches it.
    assert _side_is_drawn(10.0, 90.0, [(-40.0, 200.0)])


PLAN1 = {
    # Every figure here is printed on the sheet's own site plan.
    "plot.width": 12.19,       # top/bottom edge label
    "plot.depth": 9.14,        # left/right edge label
    "building.width": 11.72,   # red footprint, horizontal
    "building.depth": 8.22,    # red footprint, vertical
    "stated_footprint_area": 96.34,   # "PROP. PLINTH AREA IN G.F"
    # The road is drawn on the RIGHT, so the right gap is the front setback.
    # The building shares its left edge with the plot boundary.
    "setbacks.front": 0.47,
    "setbacks.left": 0.46,     # bottom edge, seen from the road
    "setbacks.right": 0.46,    # top edge, seen from the road
}


def test_plan1_resolves_the_true_plot_and_building_not_a_stitched_rectangle():
    """
    PLAN1 is a composite sheet with a thin frame line at the site plan's
    left/top and the building's red outline at its right/bottom. A
    "rectangle" built from those two objects' lines scored above the real
    green plot boundary, which shifted the plot, the building nested in it
    (reported 12.21 m wide inside a 12.19 m plot) and every setback.
    """
    got = _measurements("PLAN1.pdf")
    _assert_close(got.get("plot.width"), PLAN1["plot.width"], "plot.width")
    _assert_close(got.get("plot.depth"), PLAN1["plot.depth"], "plot.depth")
    _assert_close(got.get("building.width"), PLAN1["building.width"], "building.width")
    _assert_close(got.get("building.depth"), PLAN1["building.depth"], "building.depth")
    assert got["building.width"] < got["plot.width"]
    assert got["building.depth"] < got["plot.depth"]


def test_plan1_setbacks_match_the_callouts_printed_on_the_sheet():
    got = _measurements("PLAN1.pdf")
    _assert_close(got.get("setbacks.front"), PLAN1["setbacks.front"], "setbacks.front")
    _assert_close(got.get("setbacks.left"), PLAN1["setbacks.left"], "setbacks.left")
    _assert_close(got.get("setbacks.right"), PLAN1["setbacks.right"], "setbacks.right")
    # The building abuts the plot boundary on one side: a real zero, not a
    # missing value.
    assert got.get("setbacks.rear") is not None, "setbacks.rear not resolved"
    assert abs(got["setbacks.rear"]) < 0.05


def test_plan1_setbacks_account_for_the_whole_plot_on_each_axis():
    """The four setbacks are not independent numbers: they are exactly what
    is left of the plot on each axis once the building is placed."""
    got = _measurements("PLAN1.pdf")
    # Road on the right => front/rear run along the plot's width.
    _assert_close(
        got["setbacks.front"] + got["setbacks.rear"] + got["building.width"],
        got["plot.width"], "front + rear + building.width", 0.01,
    )
    _assert_close(
        got["setbacks.left"] + got["setbacks.right"] + got["building.depth"],
        got["plot.depth"], "left + right + building.depth", 0.01,
    )


# --- Production-pipeline checks: floor count / FAR / setback flags ----------------

import functools  # noqa: E402


@functools.lru_cache(maxsize=None)
def _production_plan(stem: str):
    """The real `PDFHybridExtractor -> build_normalized_plan` path the app
    runs (not the bypass `extract_independent_cv` the tests above use)."""
    from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
    from backend.spatial_reasoning.pipeline import build_normalized_plan

    pdf_path = PLANS_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        pytest.skip(f"{stem}.pdf fixture not present")
    extraction = PDFHybridExtractor(enable_vision=False).extract(pdf_path, stem)
    return build_normalized_plan(extraction, plan_id=stem)


def test_plan1_floor_count_comes_from_the_floors_its_sheet_names():
    """PLAN1 never prints "G+3"; its title lists the four floors, it has one
    caption per floor plan, and its area table has one row per floor."""
    plan = _production_plan("PLAN1")
    floors = plan.building.floor_count
    assert floors is not None and floors.value == 4
    assert floors.confidence.level == ConfidenceLevel.MEDIUM   # three independent forms agree


def test_plan1_far_follows_from_its_floor_count():
    """FAR = footprint x floors / plot: 96.34 x 4 / 111.42 = 3.46."""
    plan = _production_plan("PLAN1")
    _assert_close(plan.far.value, 3.4586, "far", 0.01)
    assert plan.far.confidence.level != ConfidenceLevel.HIGH   # bounded by the floor count's own confidence


def test_stilt_sheets_count_ground_plus_upper_floors_not_the_stilt():
    """PLAN6 (captions: stilt, ground, first, second) and PLAN8 ("STILT, GF+2UF")
    are both 3 floors under the ground-plus-upper convention."""
    assert _production_plan("PLAN6").building.floor_count.value == 3
    assert _production_plan("PLAN8").building.floor_count.value == 3


def test_a_caption_reading_2_floor_plan_is_not_a_count_of_2():
    """PLAN8 printed "2 FLOOR PLAN" (its second-floor drawing); that used to be
    read as 2 floors at MEDIUM confidence -- the only confident-wrong answer in
    the PDF baseline."""
    floors = _production_plan("PLAN8").building.floor_count
    assert floors.value != 2


def test_captions_that_cover_several_levels_are_understood():
    """PLAN4: "GROUND FLOOR PLAN", "1ST, 2ND & 3RD FLOOR PLAN", "FOURTH FLOOR PLAN"."""
    assert _production_plan("PLAN4").building.floor_count.value == 5


def test_a_sheet_whose_sources_disagree_about_floors_abstains():
    """PLAN9 captions only ground and first, while its area table also lists a
    second floor."""
    floors = _production_plan("PLAN9").building.floor_count
    assert floors is None or floors.value is None


def test_plan1_building_use_is_read_from_the_end_of_its_title():
    """"...RCC ROOF RESIDENTIAL BUILDING, IN S.NO ..." -- no "PROPOSED" directly before it."""
    use = _production_plan("PLAN1").building_use
    assert use is not None and use.value == "residential"


@pytest.mark.parametrize("stem", ["PLAN1", "PLAN2", "PLAN4", "PLAN5", "PLAN6", "PLAN8", "PLAN9"])
def test_correct_setbacks_are_not_flagged_implausible_on_any_sheet_orientation(stem):
    """PLAN4 and PLAN5 have their road on a side; their correct setbacks were
    flagged because the check assumed front/rear always run along plot depth."""
    plan = _production_plan(stem)
    for side in ("front", "rear", "left", "right"):
        field = getattr(plan.setbacks, side)
        assert field.conflict is None, f"{stem} setbacks.{side}={field.value}: {field.conflict.description}"


# --- Plots bounded by slanted edges (PLAN9, PLAN8) ---------------------------------

PLAN9 = {
    # Printed on the sheet's site plan. The plot is a skewed quadrilateral
    # (top 10.65 / bottom 8.92, left 15.25 slanted / right 15.00), road below.
    "plot.width": 10.65,
    "building.width": 7.14,
    "building.depth": 12.11,
    "setbacks.front": 1.50,    # bottom gap
    "setbacks.left": 1.00,     # narrow end of the leaning left gap (2.52 at the top)
    "setbacks.right": 1.00,
}


def test_plan9_leaning_plot_yields_the_building_and_minimum_setbacks():
    got = _measurements("PLAN9.pdf")
    _assert_close(got.get("plot.width"), PLAN9["plot.width"], "plot.width")
    _assert_close(got.get("building.width"), PLAN9["building.width"], "building.width")
    _assert_close(got.get("building.depth"), PLAN9["building.depth"], "building.depth")
    _assert_close(got.get("setbacks.front"), PLAN9["setbacks.front"], "setbacks.front", 0.05)
    assert got.get("setbacks.left") == pytest.approx(PLAN9["setbacks.left"], abs=0.15)
    assert got.get("setbacks.right") == pytest.approx(PLAN9["setbacks.right"], abs=0.15)
    # The top edge leans, so the true rear gap is smaller than the 1.50 printed at
    # its wider end; the extractor must report the minimum, never more than printed.
    assert 1.2 <= got["setbacks.rear"] <= 1.55


def test_plan9_setbacks_are_the_minimum_not_the_widest_gap_of_a_leaning_edge():
    """The left gap is 2.52 m at the top and 1.00 m at the bottom. Reporting the
    wider figure would overstate compliance against a minimum-setback rule."""
    assert _measurements("PLAN9.pdf")["setbacks.left"] < 1.5


def test_plan9_production_values_are_correct_and_no_longer_absurd():
    """Before: building 1.74 x 1.78 m and setbacks 0.44 / 0.24 / 0.005 at HIGH."""
    plan = _production_plan("PLAN9")
    assert plan.building.width.value == pytest.approx(7.14, rel=0.02)
    assert plan.building.depth.value == pytest.approx(12.11, rel=0.02)
    assert plan.setbacks.left.value == pytest.approx(1.0, abs=0.15)
    assert plan.setbacks.right.value == pytest.approx(1.0, abs=0.15)


def test_plan8_slanted_plot_geometry_is_never_shipped_with_confidence():
    """PLAN8's printed dimensions disagree with its own geometry, and a rectangle
    from a different drawing once matched its stated areas by coincidence. What
    the extractor cannot verify it may still report, but not above LOW."""
    plan = _production_plan("PLAN8")
    for name, field in (
        ("plot.width", plan.plot.width), ("plot.depth", plan.plot.depth),
        ("building.width", plan.building.width), ("building.depth", plan.building.depth),
        ("setbacks.front", plan.setbacks.front), ("setbacks.rear", plan.setbacks.rear),
        ("setbacks.left", plan.setbacks.left), ("setbacks.right", plan.setbacks.right),
    ):
        if field is not None and field.value is not None:
            assert field.confidence.level == ConfidenceLevel.LOW, f"{name}={field.value} shipped at {field.confidence.level}"
    # The values printed in its area statement are still trusted.
    assert plan.plot.area.value == pytest.approx(183.69, rel=0.01)
    assert plan.building.footprint_area.value == pytest.approx(115.42, rel=0.01)
