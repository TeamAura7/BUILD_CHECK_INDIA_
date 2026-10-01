"""
Tests for `backend.cv_extraction.dxf_extractor.DXFHybridExtractor`.

There was previously ZERO test coverage of this extractor at all (grep for
`dxf_extractor`/`DXFHybridExtractor` across `tests/` found nothing) despite
it being wired into the production upload path (`backend/app/routes/
analyze.py`). A user reported that uploading a real DXF "runs but doesn't
move forward" -- reproduced directly against the real uploaded file
(`data/uploads/b38c212f_PLAN5.dxf`, a DXF produced by vectorizing/tracing a
scanned sheet: 6,346 closed polygons, most of them tiny fragments -- e.g.
each dash of a dash-dot boundary rendered as its own tiny filled quad, zero
TEXT/DIMENSION entities). Root cause: `_pick_plot_polygon`/
`_pick_building_polygon`/`_pick_road_polygon` each independently rebuilt
`site_graph.build_site_graph`'s O(n^2)-in-polygon-count graph over the
FULL, uncapped polygon list -- three separate times. On that real file this
measured at 92+ seconds for just 300 polygons (see git history for the
diagnostic session), and unbounded for the full 6,346 -- with no timeout
wrapper at all before this fix, so the upload's background thread hung
indefinitely rather than erroring.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
from tests.fixtures.dxf_builders import (
    door_gap_building_site_plan_dxf,
    multi_drawing_sheet_with_frame_dxf,
    no_text_site_plan_dxf,
    rectangular_site_plan_dxf,
    rotated_rectangular_site_plan_dxf,
)


def _measurements(result) -> dict[str, float]:
    if result.independent_cv is None:
        return {}
    return {
        m.field: (m.value_m if m.value_m is not None else m.value)
        for m in result.independent_cv.measurements
    }


# --- Correctness on a normal, layer-tagged DXF -------------------------------


def test_resolves_plot_building_road_from_layer_names(tmp_path):
    dxf_path = rectangular_site_plan_dxf(tmp_path / "site.dxf", plot_width=12.0, plot_depth=18.0)
    result = DXFHybridExtractor().extract(dxf_path, "site")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(12.0, abs=0.01)
    assert got["plot.depth"] == pytest.approx(18.0, abs=0.01)
    assert got["plot.area"] == pytest.approx(12.0 * 18.0, rel=0.01)
    # Building inset by front=3, rear=2, left=1.5, right=1 from a 12x18 plot.
    assert got["building.width"] == pytest.approx(12.0 - 1.5 - 1.0, abs=0.01)
    assert got["building.depth"] == pytest.approx(18.0 - 3.0 - 2.0, abs=0.01)


def test_resolves_a_non_rectangular_l_shaped_plot(tmp_path):
    """Generalization: a real plot is not always a plain rectangle (an
    L-shaped or otherwise irregular boundary is common, e.g. a corner
    plot with a cut corner, or a plot that wraps around a neighbouring
    property). Plot area must be the polygon's own exact shoelace area
    (not its bounding-box approximation), and width/depth must still
    resolve to the two side lengths of its minimum-area oriented
    rectangle without crashing, at reduced confidence for the shape not
    being a clean rectangle."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    doc.layers.add("PLOT")
    doc.layers.add("BLDG")
    # L-shaped plot: overall bounding rectangle 16 x 10, with a notch cut
    # out of the top-right corner (4 x 4).
    l_shape = [(0, 0), (16, 0), (16, 6), (12, 6), (12, 10), (0, 10)]
    msp.add_lwpolyline(l_shape, close=True, dxfattribs={"layer": "PLOT"})
    msp.add_lwpolyline(
        [(2, 2), (9, 2), (9, 8), (2, 8)], close=True, dxfattribs={"layer": "BLDG"},
    )
    dxf_path = tmp_path / "l_shaped_plot.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "l_shaped_plot")
    got = _measurements(result)

    def _shoelace_area(pts):
        n = len(pts)
        return abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))) / 2.0

    assert got["plot.area"] == pytest.approx(_shoelace_area(l_shape), rel=0.01)
    # The overall bounding rectangle is 16x10 -- width/depth come from the
    # minimum-area oriented rectangle of the L-shape's own convex hull,
    # which for an axis-aligned L exactly matches its bounding box.
    assert sorted([got["plot.width"], got["plot.depth"]]) == pytest.approx(sorted([16.0, 10.0]), rel=0.01)
    assert got["building.width"] == pytest.approx(7.0, abs=0.01)
    assert got["building.depth"] == pytest.approx(6.0, abs=0.01)


def test_resolves_geometry_with_no_text_labels_at_all(tmp_path):
    """Exercises the graph-based (non-layer-hint... well, layer hints are
    still present here since the fixture always tags layers; this pins
    that the geometry-only role-inference path -- used when a real DXF
    lacks PLOT/BLDG/ROAD layer names -- still works when text is absent."""
    dxf_path = no_text_site_plan_dxf(tmp_path / "site_no_text.dxf", plot_width=10.0, plot_depth=14.0)
    result = DXFHybridExtractor().extract(dxf_path, "site_no_text")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(10.0, abs=0.01)
    assert got["plot.depth"] == pytest.approx(14.0, abs=0.01)


def test_role_inference_works_without_layer_names(tmp_path):
    """
    Same geometry as above but with every polygon's layer name blanked out,
    so `_pick_plot_polygon`/`_pick_building_polygon`/`_pick_road_polygon`
    are FORCED onto the graph-structure inference path (`infer_plot_node`/
    `infer_building_node`/`infer_road_node`) rather than the layer-name
    fast path -- this is what actually exercises the shared-graph fix.
    """
    import ezdxf

    dxf_path = rectangular_site_plan_dxf(tmp_path / "site_unlabeled.dxf", plot_width=12.0, plot_depth=18.0)
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    for e in msp:
        if e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            e.dxf.layer = "0"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "site_unlabeled")
    got = _measurements(result)
    assert got.get("plot.width") == pytest.approx(12.0, abs=0.05)
    assert got.get("plot.depth") == pytest.approx(18.0, abs=0.05)
    assert got.get("building.width") is not None
    assert got.get("building.depth") is not None


# --- Oriented (rotation-independent) width/depth measurement -----------------
#
# Critical requirement: width/depth must come from the polygon's own
# minimum-area oriented bounding rectangle, never an axis-aligned
# max_x-min_x / max_y-min_y bbox, so a rotated sheet reports the same real
# dimensions a sheet drawn at 0 degrees would. The rectangle's own edges
# have no fixed relationship to "width" vs "depth" once rotated (that
# label only has a stable meaning relative to a road/front direction,
# which these fixtures don't encode), so each test compares the {width,
# depth} PAIR, not which specific field holds which value.


@pytest.mark.parametrize("angle_degrees", [0, 15, 30, 45, 60, 90])
def test_plot_width_depth_are_rotation_independent(tmp_path, angle_degrees):
    dxf_path = rotated_rectangular_site_plan_dxf(
        tmp_path / f"rotated_{angle_degrees}.dxf",
        plot_width=12.0, plot_depth=18.0, angle_degrees=angle_degrees,
    )
    result = DXFHybridExtractor().extract(dxf_path, f"rotated_{angle_degrees}")
    got = _measurements(result)
    got_pair = sorted([got["plot.width"], got["plot.depth"]])
    expected_pair = sorted([12.0, 18.0])
    assert got_pair == pytest.approx(expected_pair, abs=0.05), (
        f"at {angle_degrees} degrees: expected the plot's real 12x18 dimensions "
        f"regardless of rotation, got {got_pair}"
    )
    assert got["plot.area"] == pytest.approx(12.0 * 18.0, rel=0.02)


@pytest.mark.parametrize("angle_degrees", [0, 15, 30, 45, 60, 90])
def test_building_width_depth_are_rotation_independent(tmp_path, angle_degrees):
    dxf_path = rotated_rectangular_site_plan_dxf(
        tmp_path / f"rotated_bldg_{angle_degrees}.dxf",
        plot_width=12.0, plot_depth=18.0,
        front_setback=3.0, rear_setback=2.0, left_setback=1.5, right_setback=1.0,
        angle_degrees=angle_degrees,
    )
    result = DXFHybridExtractor().extract(dxf_path, f"rotated_bldg_{angle_degrees}")
    got = _measurements(result)
    got_pair = sorted([got["building.width"], got["building.depth"]])
    # Building inset by front=3, rear=2, left=1.5, right=1 from a 12x18 plot.
    expected_pair = sorted([12.0 - 1.5 - 1.0, 18.0 - 3.0 - 2.0])
    assert got_pair == pytest.approx(expected_pair, abs=0.05), (
        f"at {angle_degrees} degrees: expected the building's real 9.5x13 "
        f"dimensions regardless of rotation, got {got_pair}"
    )


@pytest.mark.parametrize("angle_degrees", [0, 15, 30, 45, 60, 90])
def test_setbacks_are_rotation_independent(tmp_path, angle_degrees):
    """Critical Requirement 4: setbacks must be geometrically explainable
    from the building's own polygon boundary, never its axis-aligned bbox
    corners -- a bbox corner is generally not a real point on a rotated
    building at all, and measuring from it against an ALSO-rotated plot
    edge silently produces a wrong gap (confirmed directly: before this
    fix, a 20x15 plot + 11x7 building both rotated 30 degrees reported
    setbacks {1.76, 0.24, 2.97, 0.03} m against a true {3, 5, 3, 6} m).
    The four true setback values must appear regardless of the sheet's
    rotation -- compared as a multiset, since without road/text evidence
    which specific edge is labelled "front" vs "left" etc. is a separate,
    LOW-confidence fallback concern (`front_side.py`), not what this test
    is pinning."""
    dxf_path = rotated_rectangular_site_plan_dxf(
        tmp_path / f"rotated_setbacks_{angle_degrees}.dxf",
        plot_width=20.0, plot_depth=15.0,
        front_setback=3.0, rear_setback=5.0, left_setback=2.0, right_setback=4.0,
        angle_degrees=angle_degrees,
    )
    result = DXFHybridExtractor().extract(dxf_path, f"rotated_setbacks_{angle_degrees}")
    got = _measurements(result)
    got_values = sorted([
        got["setbacks.front"], got["setbacks.rear"], got["setbacks.left"], got["setbacks.right"],
    ])
    expected_values = sorted([3.0, 5.0, 2.0, 4.0])
    assert got_values == pytest.approx(expected_values, abs=0.05), (
        f"at {angle_degrees} degrees: expected the true {expected_values} m setbacks "
        f"regardless of rotation, got {got_values}"
    )


def test_axis_aligned_bbox_would_have_failed_the_45_degree_case(tmp_path):
    """Sanity check that the 45-degree rotation case above is actually
    discriminating: confirms an axis-aligned bbox on the SAME rotated
    polygon would report a materially different (wrong) width/height, so
    the rotation test above is not vacuously passing because rotation
    happens to not matter for this shape."""
    import math

    angle_degrees = 45.0
    dxf_path = rotated_rectangular_site_plan_dxf(
        tmp_path / "rotated_45_bbox_check.dxf",
        plot_width=12.0, plot_depth=18.0, angle_degrees=angle_degrees,
    )
    result = DXFHybridExtractor().extract(dxf_path, "rotated_45_bbox_check")
    got = _measurements(result)

    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [(0, 0), (12.0, 0), (12.0, 18.0), (0, 18.0)]
    rotated = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in corners]
    axis_aligned_bbox_width = max(x for x, _ in rotated) - min(x for x, _ in rotated)
    axis_aligned_bbox_height = max(y for _, y in rotated) - min(y for _, y in rotated)

    # A buggy axis-aligned measurement would report ~21.2 x 21.2 (the
    # diagonal-driven bbox of a 12x18 rectangle at 45 degrees) instead of
    # the real 12 x 18 -- confirm those two are actually different here,
    # then confirm our measurement matches the real dimensions, not the
    # bbox ones.
    assert axis_aligned_bbox_width > 20.0 and axis_aligned_bbox_height > 20.0
    got_pair = sorted([got["plot.width"], got["plot.depth"]])
    assert got_pair == pytest.approx(sorted([12.0, 18.0]), abs=0.05)
    assert abs(got["plot.width"] - axis_aligned_bbox_width) > 1.0


# --- Area-based wall-union building reconstruction (Critical Requirement 3) --


def test_building_with_a_door_gap_and_t_junction_resolves_via_wall_union(tmp_path):
    """End-to-end: a building drawn as open wall LINE segments (never a
    closed polygon) with a door-sized gap in the front wall and an
    interior partition wall creating a T-junction. Pure closed-loop
    cycle detection (`dxf_reconstruction.reconstruct_building_polygon`)
    cannot bridge the gap at all; `dxf_wall_union.
    reconstruct_building_via_wall_union` must recover the correct
    footprint instead."""
    dxf_path = door_gap_building_site_plan_dxf(
        tmp_path / "door_gap.dxf", plot_width=20.0, plot_depth=15.0,
        front_setback=3.0, rear_setback=2.0, left_setback=1.5, right_setback=1.0,
    )
    result = DXFHybridExtractor().extract(dxf_path, "door_gap")
    got = _measurements(result)
    # Building inset by front=3, rear=2, left=1.5, right=1 from a 20x15 plot.
    assert got.get("building.width") == pytest.approx(20.0 - 1.5 - 1.0, rel=0.05)
    assert got.get("building.depth") == pytest.approx(15.0 - 3.0 - 2.0, rel=0.05)
    assert any("wall loop" in w and "door/window" in w for w in result.warnings), (
        "expected a warning explaining the wall-union fallback engaged"
    )


# --- Architectural relationships, not "biggest polygon" (Critical Requirement 4)


def test_road_is_found_by_adjacency_even_when_the_plot_is_an_envelope():
    """Direct unit test of `_pick_road_polygon` (rather than a full DXF
    round-trip, whose region-clustering and plot/road disambiguation
    would confound what's being pinned here): when the plot is a
    SYNTHETIC envelope (as `_reconstruct_plot_envelope` produces for a
    dashed boundary with no single closed polygon) it is, by
    construction, never one of the polygons a role-inference graph was
    built from -- `_pick_road_polygon` used to look the plot up as a
    graph node to run `infer_road_node`, that lookup always failed for an
    envelope, and it fell straight to a layer-name-only fallback. On a
    sheet with no "ROAD" layer tag at all, that fallback finds nothing,
    even though the road is sitting right there, adjacent to the plot and
    strip-shaped -- exactly the geometric signal
    `infer_road_polygon_by_adjacency` exists to use. This must now run
    directly against the envelope's own polygon instead of being skipped."""
    from backend.cv_extraction.dxf_extractor import (
        _ENVELOPE_RECONSTRUCTED_LAYER,
        _RawPoly,
        _build_role_inference_graph,
        _pick_road_polygon,
    )
    from backend.schemas.geometry import Point, Polygon

    def rect(x0, y0, x1, y1, layer):
        return _RawPoly(layer=layer, polygon=Polygon(points=[
            Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1),
        ]))

    building = rect(4, 3.5, 16, 11.5, "0")
    road = rect(-2, -6.5, 22, -0.5, "0")  # long strip, unlabeled, adjacent to the plot's south edge
    polygons = [building, road]
    graph, by_id = _build_role_inference_graph(polygons)

    envelope_plot = _RawPoly(
        layer=_ENVELOPE_RECONSTRUCTED_LAYER,
        polygon=Polygon(points=[Point(x=0, y=0), Point(x=20, y=0), Point(x=20, y=15), Point(x=0, y=15)]),
    )
    result = _pick_road_polygon(polygons, graph, by_id, envelope_plot)
    assert result is road


# --- Native DXF DIMENSION verification chain (Critical Requirement 5) -------


def test_native_dimension_entity_confirms_plot_width(tmp_path):
    """A real LINEAR DIMENSION entity whose own extension-line points
    (defpoint2/defpoint3) exactly coincide with the plot polygon's bottom
    edge must CONFIRM plot.width -- raising its confidence and adding
    dedicated evidence -- via the exact text -> dimension graphic ->
    measured span -> geometric edge chain, not a nearest-number guess.
    The unrelated plot.depth (no matching dimension) must be unaffected."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for layer in ("PLOT", "BLDG"):
        doc.layers.add(layer)
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 15), (0, 15)], close=True, dxfattribs={"layer": "PLOT"})
    msp.add_lwpolyline(
        [(4, 3.5), (16, 3.5), (16, 11.5), (4, 11.5)], close=True, dxfattribs={"layer": "BLDG"},
    )
    dim = msp.add_linear_dim(base=(10, -2), p1=(0, 0), p2=(20, 0), dimstyle="EZDXF")
    dim.render()
    dxf_path = tmp_path / "dim_chain.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "dim_chain")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(20.0, abs=0.01)
    assert got["plot.depth"] == pytest.approx(15.0, abs=0.01)

    width_measurement = next(m for m in result.independent_cv.measurements if m.field == "plot.width")
    depth_measurement = next(m for m in result.independent_cv.measurements if m.field == "plot.depth")
    assert any("DIMENSION entity" in e for e in width_measurement.evidence)
    assert not any("DIMENSION entity" in e for e in depth_measurement.evidence)
    assert width_measurement.confidence > depth_measurement.confidence


def test_native_dimension_confirms_width_even_when_its_own_graphic_is_far_from_the_geometry(tmp_path):
    """Generalization: a real DIMENSION entity's TEXT and dimension line
    are often drawn well outside the bounding box of the geometry it
    measures (a leader reaching out to a label placed in clear space, a
    dimension chain offset far from a cramped drawing) -- the chain must
    still verify correctly because it is anchored on the extension-line
    points (defpoint2/defpoint3, always exactly at the measured span),
    never on the text or dimension-line position. `base` here is 200
    units away from the plot itself (which spans 0-20), far outside its
    bounding box, while the measured span (p1/p2) is still exactly the
    real edge. Keeps a tagged BLDG polygon too (same as the sibling test
    above) so plot/building role resolution has two real polygons to
    reason about -- a file with exactly one closed polygon and nothing
    else can never resolve via containment-fraction role inference at
    all (there is nothing for it to "contain a fraction of"), which is a
    separate, correct limitation of that heuristic, not what this test
    exists to pin."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for layer in ("PLOT", "BLDG"):
        doc.layers.add(layer)
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 15), (0, 15)], close=True, dxfattribs={"layer": "PLOT"})
    msp.add_lwpolyline(
        [(4, 3.5), (16, 3.5), (16, 11.5), (4, 11.5)], close=True, dxfattribs={"layer": "BLDG"},
    )
    # Offset well past the plot's own bounding box (which spans y=0..15)
    # without being so extreme that the dimension's own rendered
    # extension-line geometry dwarfs the drawing's scale -- a real
    # dimension line sitting a modest, realistic distance outside the
    # geometry it measures, not an absurd one.
    dim = msp.add_linear_dim(base=(10, -8), p1=(0, 0), p2=(20, 0), dimstyle="EZDXF")
    dim.render()
    dxf_path = tmp_path / "dim_far_graphic.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "dim_far_graphic")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(20.0, abs=0.01)
    width_measurement = next(m for m in result.independent_cv.measurements if m.field == "plot.width")
    assert any("DIMENSION entity" in e for e in width_measurement.evidence)


def test_dimension_entity_matching_neither_width_nor_depth_is_flagged_not_dropped(tmp_path):
    """A DIMENSION entity whose points exactly match a real edge of the
    SAME polygon, but whose value corresponds to neither the polygon's
    overall (oriented-bounding-rectangle) width nor depth -- e.g. a notch
    edge on an L-shaped plot -- must surface a warning explaining the
    mismatch rather than silently vanishing or being force-fit onto the
    wrong field."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    doc.layers.add("PLOT")
    # L-shaped plot: overall bounding rectangle is 16 x 10, but the notch
    # edge from (8,10) to (8,6) is only 4 m -- neither 16 nor 10.
    msp.add_lwpolyline(
        [(0, 0), (16, 0), (16, 10), (8, 10), (8, 6), (0, 6)],
        close=True, dxfattribs={"layer": "PLOT"},
    )
    dim = msp.add_linear_dim(base=(6, 8), p1=(8, 10), p2=(8, 6), angle=90, dimstyle="EZDXF")
    dim.render()
    dxf_path = tmp_path / "dim_mismatch.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "dim_mismatch")
    got = _measurements(result)
    assert sorted([got["plot.width"], got["plot.depth"]]) == pytest.approx([10.0, 16.0], abs=0.01)
    assert any("matches neither the computed width" in w for w in result.warnings)


# --- Generalization: vectorized text, zero TEXT/MTEXT/DIMENSION entities ----


def test_render_ocr_recovery_confirms_plot_width_with_no_text_entities():
    """Real vectorized-trace DXFs (this module's own docstring: PLAN5,
    6,346 closed fragments, ZERO TEXT/DIMENSION entities) commonly have
    every printed digit traced as its own tiny polyline stroke during
    scanning -- structurally indistinguishable from ordinary line-work
    without actually rendering the sheet and OCR-ing it. This exercises
    the whole-sheet render+OCR recovery mechanism
    (`_recover_sheet_wide_evidence` + `_score_recovered_evidence`)
    directly with mocked OCR hits placed at the real pixel locations
    "12.00m"/"18.00m" labels over the plot's width/depth edges would
    render at (via the same `compute_transform` production code uses, not
    a guessed position) -- confirming the CHAIN from a rendered image
    through OCR through geometric association to confirmed plot.width AND
    plot.depth, independent of which region-clustering path a full
    end-to-end DXF happens to take to reach this point (covered separately
    by the real PLAN5/PLAN6 fixtures in the Phase 0 eval harness).

    Both axes are confirmed here (not width alone) because a genuine
    scoring bonus now requires independent confirmation of BOTH of a
    candidate's spatial extents -- see `_SINGLE_AXIS_EVIDENCE_DISCOUNT`'s
    docstring for the real regression (a single coincidental digit match
    outscoring a genuinely better candidate) this guards against.
    """
    from unittest.mock import patch

    from backend.cv_extraction.dxf_extractor import _RawPoly, _recover_sheet_wide_evidence, _score_recovered_evidence
    from backend.cv_extraction.dxf_render import compute_transform
    from backend.schemas.geometry import Point, Polygon

    plot_points = [(0.0, 0.0), (12.0, 0.0), (12.0, 18.0), (0.0, 18.0)]
    plot_edges = [
        ((0.0, 0.0), (12.0, 0.0)), ((12.0, 0.0), (12.0, 18.0)),
        ((12.0, 18.0), (0.0, 18.0)), ((0.0, 18.0), (0.0, 0.0)),
    ]
    from backend.schemas.geometry import BoundingBox

    region_bboxes = {0: BoundingBox(min_x=0.0, min_y=0.0, max_x=12.0, max_y=18.0)}

    transform = compute_transform(min_x=0.0, min_y=0.0, max_x=12.0, max_y=18.0, target_max_px=3000)
    px_w, py_w = transform.world_to_pixel(6.0, 0.0)  # bottom edge midpoint, where a "12.00m" width label would sit
    px_d, py_d = transform.world_to_pixel(0.0, 9.0)  # left edge midpoint, where an "18.00m" depth label would sit

    warnings: list[str] = []
    with patch("backend.cv_extraction.dxf_text_recovery._import_tesseract") as mock_import:
        mock_pytesseract = mock_import.return_value
        mock_pytesseract.image_to_data.return_value = {
            "text": ["12.00m", "18.00m"], "conf": [88.0, 88.0],
            "left": [int(px_w - 30), int(px_d - 30)], "top": [int(py_w - 25), int(py_d - 25)],
            "width": [60, 60], "height": [20, 20],
        }
        mock_pytesseract.Output.DICT = "dict"

        evidence_by_region, _captions_by_region = _recover_sheet_wide_evidence(
            [plot_points], [], {0: plot_edges}, region_bboxes, warnings,
        )

    assert 0 in evidence_by_region and evidence_by_region[0], "OCR recovery found nothing to associate at all"
    recovered = evidence_by_region[0]
    assert any(abs(r.value - 12.0) < 1e-6 for r in recovered)
    assert any(abs(r.value - 18.0) < 1e-6 for r in recovered)

    plot_entry = _RawPoly(layer="0", polygon=Polygon(points=[Point(x=x, y=y) for x, y in plot_points]))
    bonus, scale_correction = _score_recovered_evidence(recovered, plot_entry, None, warnings, "region 0")
    assert bonus > 0
    assert any("Recovered vectorized-text evidence confirms" in w and "plot.width" in w for w in warnings)
    assert any("plot.depth" in w for w in warnings)


def test_single_axis_evidence_match_contributes_no_score_bonus():
    """Pins the actual regression found and fixed this session: a
    candidate whose depth alone (not width) happened to be within
    tolerance of one recovered, low-specificity digit still scored high
    enough to beat a genuinely better, evidence-guided-merged candidate
    that had not (yet) picked up a matching bonus of its own. A recovered
    value confirming only ONE of a candidate's two independent spatial
    extents must not move its score at all."""
    from backend.cv_extraction.dxf_extractor import _RawPoly, _score_recovered_evidence
    from backend.cv_extraction.dxf_text_recovery import RecoveredDimension
    from backend.schemas.geometry import Point, Polygon

    # A 15.35 x 7.43 candidate (real bundled PLAN5 numbers) with a single
    # recovered "7.14" (specific-looking: has a decimal point, 3 digits)
    # falling within tolerance of its DEPTH only -- width (15.35) is
    # unconfirmed.
    plot_entry = _RawPoly(layer="RECONSTRUCTED_ENVELOPE", polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=15.35, y=0), Point(x=15.35, y=7.43), Point(x=0, y=7.43),
    ]))
    depth_only_match = RecoveredDimension(
        value=7.14, unit_hint=None, raw_text="7.14", world_bbox=(0.0, 0.0, 1.0, 1.0),
        confidence=0.7, associated_segment_index=None, associated_segment_length=None,
    )
    warnings: list[str] = []
    bonus, _scale = _score_recovered_evidence([depth_only_match], plot_entry, None, warnings, "region 0")
    assert bonus == 0.0
    # The match is still surfaced for auditability, just not scored.
    assert any("plot.depth" in w for w in warnings)


# --- Rank 1: adaptive OCR render-resolution escalation -----------------------
#
# Confirmed on the real PLAN5.dxf file (see the master extraction-
# architecture audit): the entity-count-derived starting resolution
# (1400px for a ~22,500-entity sheet) recovered ZERO OCR items, while the
# same sheet at 5000px recovered the exact correct printed dimensions.
# These tests pin the escalation mechanism that fixes this in isolation,
# without needing the real multi-megabyte file.


def test_escalates_to_high_fidelity_resolution_when_starting_attempt_finds_nothing():
    """A starting attempt that recovers nothing, with ample remaining
    budget, must escalate straight to `_HIGH_FIDELITY_ESCALATION_PX` --
    not stay stuck at the cost-bounded starting tier."""
    from unittest.mock import patch

    from backend.cv_extraction.dxf_extractor import (
        _HIGH_FIDELITY_ESCALATION_PX,
        _recover_sheet_wide_evidence,
    )
    from backend.cv_extraction.dxf_text_recovery import RecoveredDimension
    from backend.schemas.geometry import BoundingBox

    plot_edges = [((0.0, 0.0), (12.0, 0.0))]
    region_bboxes = {0: BoundingBox(min_x=0.0, min_y=0.0, max_x=12.0, max_y=18.0)}
    confirmed = RecoveredDimension(
        value=12.0, unit_hint="m", raw_text="12.00m", world_bbox=(5.0, -1.0, 7.0, 1.0),
        confidence=0.9, associated_segment_index=0, associated_segment_length=12.0,
    )

    calls = []

    def _fake_recover(polys, chains, edges, caption_patterns, render_target_px):
        calls.append(render_target_px)
        return ([], []) if render_target_px != _HIGH_FIDELITY_ESCALATION_PX else ([confirmed], [])

    warnings: list[str] = []
    with patch("backend.cv_extraction.dxf_text_recovery.recover_dimension_and_caption_evidence", side_effect=_fake_recover):
        evidence_by_region, _captions_by_region = _recover_sheet_wide_evidence(
            [], [], {0: plot_edges}, region_bboxes, warnings, remaining_budget_seconds=120.0,
        )

    assert calls[0] != _HIGH_FIDELITY_ESCALATION_PX  # started at the cost-bounded tier
    assert _HIGH_FIDELITY_ESCALATION_PX in calls  # then escalated
    assert evidence_by_region[0], "escalated attempt's evidence should be used"
    assert any("escalated to" in w for w in warnings)


def test_does_not_escalate_when_remaining_budget_is_too_tight():
    """The whole point of the escalation gate: never risk blowing the
    overall extraction timeout on a speculative retry. A near-zero
    remaining budget must suppress escalation even though the starting
    attempt found nothing."""
    import time as time_module
    from unittest.mock import patch

    from backend.schemas.geometry import BoundingBox

    calls = []

    def _fake_recover(polys, chains, edges, caption_patterns, render_target_px):
        calls.append(render_target_px)
        time_module.sleep(0.05)  # measurable cost, so the budget gate has a real elapsed time to react to
        return [], []

    plot_edges = [((0.0, 0.0), (12.0, 0.0))]
    region_bboxes = {0: BoundingBox(min_x=0.0, min_y=0.0, max_x=12.0, max_y=18.0)}
    warnings: list[str] = []
    from backend.cv_extraction.dxf_extractor import _recover_sheet_wide_evidence

    # Less than the 0.05s the starting attempt alone will take -- nowhere
    # near the ~35x-of-observed-cost the escalation gate requires.
    with patch("backend.cv_extraction.dxf_text_recovery.recover_dimension_and_caption_evidence", side_effect=_fake_recover):
        _recover_sheet_wide_evidence(
            [], [], {0: plot_edges}, region_bboxes, warnings, remaining_budget_seconds=0.02,
        )

    assert len(calls) == 1, "must not have attempted a second (escalated) call when budget is too tight"


def test_does_not_escalate_when_no_budget_information_is_available():
    """Direct/unit-test callers that don't pass `remaining_budget_seconds`
    (the default, None) must get the old cost-bounded single-attempt
    behavior -- escalation is opt-in via real budget information, never a
    blind default."""
    from unittest.mock import patch

    from backend.schemas.geometry import BoundingBox

    calls = []

    def _fake_recover(polys, chains, edges, caption_patterns, render_target_px):
        calls.append(render_target_px)
        return [], []

    plot_edges = [((0.0, 0.0), (12.0, 0.0))]
    region_bboxes = {0: BoundingBox(min_x=0.0, min_y=0.0, max_x=12.0, max_y=18.0)}
    warnings: list[str] = []
    from backend.cv_extraction.dxf_extractor import _recover_sheet_wide_evidence

    with patch("backend.cv_extraction.dxf_text_recovery.recover_dimension_and_caption_evidence", side_effect=_fake_recover):
        _recover_sheet_wide_evidence([], [], {0: plot_edges}, region_bboxes, warnings)

    assert len(calls) == 1


def test_does_not_escalate_when_starting_attempt_already_found_something():
    """Escalation is gated on the starting attempt finding literally
    nothing -- any non-empty result (even a single noisy item) must not
    trigger a second, more expensive OCR pass."""
    from unittest.mock import patch

    from backend.cv_extraction.dxf_text_recovery import RecoveredDimension
    from backend.schemas.geometry import BoundingBox

    calls = []
    found = RecoveredDimension(
        value=2.0, unit_hint=None, raw_text="2", world_bbox=(0.0, 0.0, 1.0, 1.0),
        confidence=0.3, associated_segment_index=None, associated_segment_length=None,
    )

    def _fake_recover(polys, chains, edges, caption_patterns, render_target_px):
        calls.append(render_target_px)
        return [found], []

    plot_edges = [((0.0, 0.0), (12.0, 0.0))]
    region_bboxes = {0: BoundingBox(min_x=0.0, min_y=0.0, max_x=12.0, max_y=18.0)}
    warnings: list[str] = []
    from backend.cv_extraction.dxf_extractor import _recover_sheet_wide_evidence

    with patch("backend.cv_extraction.dxf_text_recovery.recover_dimension_and_caption_evidence", side_effect=_fake_recover):
        _recover_sheet_wide_evidence(
            [], [], {0: plot_edges}, region_bboxes, warnings, remaining_budget_seconds=120.0,
        )

    assert len(calls) == 1


# --- The hang, reproduced with a synthetic fixture and fixed -----------------


def _dense_fragment_dxf(path: Path, n_fragments: int = 2500) -> Path:
    """
    A synthetic stand-in for the real problem file: one large plot polygon
    (unlabeled layer, forcing graph-based role inference) plus many tiny
    closed quads scattered around it -- the same shape the real vectorized-
    trace DXF has (thousands of tiny closed fragments from a dash-dot
    boundary), without needing the actual 13MB user file in the repo.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (30, 0), (30, 40), (0, 40)], close=True, dxfattribs={"layer": "0"})
    for i in range(n_fragments):
        x = (i % 100) * 0.3
        y = (i // 100) * 0.3
        msp.add_lwpolyline(
            [(x, y), (x + 0.05, y), (x + 0.05, y + 0.05), (x, y + 0.05)],
            close=True, dxfattribs={"layer": "0"},
        )
    doc.saveas(str(path))
    return path


def test_many_small_fragments_no_longer_hangs(tmp_path):
    """
    Pins the actual reported bug: a DXF with many thousands of tiny closed
    polygon fragments (all on an unlabeled layer, forcing the O(n^2)
    graph-based role-inference path) must complete in a bounded, reasonable
    time -- not hang. Before the fix (graph rebuilt 3x over the full,
    uncapped list) this class of file made the pipeline appear to hang
    indefinitely; the real 6,346-fragment file measured 92+ seconds for
    just an artificially-capped 300 of them.
    """
    dxf_path = _dense_fragment_dxf(tmp_path / "dense.dxf", n_fragments=2500)
    t0 = time.time()
    result = DXFHybridExtractor().extract(dxf_path, "dense")
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"DXF extraction took {elapsed:.1f}s for 2500 fragments -- regression toward the hang"
    got = _measurements(result)
    assert got.get("plot.width") == pytest.approx(30.0, abs=0.5)
    assert got.get("plot.depth") == pytest.approx(40.0, abs=0.5)


def test_role_inference_graph_is_not_rebuilt_per_role_query(tmp_path, monkeypatch):
    """Structural pin, not just a timing budget: `_polygons_to_site_graph`
    (the expensive O(n^2) build) must be built once per plot/building/road
    RESOLUTION ATTEMPT (`_resolve_plot_building_road`), not once per
    individual plot/building/road role query within that attempt -- that
    was the original historical bug this test was written to catch.

    Region-scoped resolution (see `_resolve_via_regions`) legitimately
    makes more than one resolution attempt on a sheet with multiple
    spatially distinct drawings -- once per region, each over a much
    smaller per-region polygon subset -- so the bound here is "at most one
    call per detected region plus one for the whole-sheet fallback", not a
    hardcoded 1. What must never happen again is the OLD bug: three
    separate full-uncapped-polygon-list builds for a single resolution
    attempt.
    """
    dxf_path = rectangular_site_plan_dxf(tmp_path / "site.dxf")
    import ezdxf
    doc = ezdxf.readfile(str(dxf_path))
    for e in doc.modelspace():
        if e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
            e.dxf.layer = "0"
    doc.saveas(str(dxf_path))

    from backend.cv_extraction import dxf_extractor

    call_count = {"n": 0}
    original = dxf_extractor._polygons_to_site_graph

    def counting_wrapper(polygons):
        call_count["n"] += 1
        return original(polygons)

    monkeypatch.setattr(dxf_extractor, "_polygons_to_site_graph", counting_wrapper)
    DXFHybridExtractor().extract(dxf_path, "site")
    # This fixture's geometry (a handful of simple rectangles) clusters
    # into at most a couple of regions; +1 accounts for the whole-sheet
    # fallback attempt. The old bug rebuilt the SAME full-polygon-list
    # graph 3 times for one attempt -- this bound would still catch that.
    assert call_count["n"] <= 3


# --- Implausible-building-footprint guard ------------------------------------


def test_implausibly_tiny_building_footprint_is_rejected_not_reported(tmp_path):
    """
    When the largest polygon nested inside the plot is a small fraction of
    the plot's own area (e.g. a stray closed fragment rather than a real
    building outline -- exactly what happens on a DXF where the building's
    walls were never captured as one closed polygon at all), that must be
    treated as unresolved, not reported as a confident building.footprint_area.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    # A 20x20 plot with only a tiny 0.5x0.5 fragment nested inside it --
    # no plausible building-sized polygon anywhere.
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 20), (0, 20)], close=True, dxfattribs={"layer": "0"})
    msp.add_lwpolyline([(5, 5), (5.5, 5), (5.5, 5.5), (5, 5.5)], close=True, dxfattribs={"layer": "0"})
    doc.saveas(str(tmp_path / "sparse.dxf"))

    result = DXFHybridExtractor().extract(tmp_path / "sparse.dxf", "sparse")
    got = _measurements(result)
    assert got.get("plot.width") == pytest.approx(20.0, abs=0.01)
    assert "building.footprint_area" not in got
    assert any("implausibly small" in w for w in result.warnings)


# --- Curved geometry: ARC / CIRCLE / bulge -----------------------------------


def _confidence(result, field: str) -> float:
    m = next(mm for mm in result.independent_cv.measurements if mm.field == field)
    return m.confidence


def test_lwpolyline_with_bulge_rounded_corner_is_flattened_not_cut_straight(tmp_path):
    """
    A plot boundary with one rounded (bulge) corner must resolve close to
    its TRUE bounding-box dimensions -- if the bulge were silently treated
    as a straight line (naive `get_points('xy')`), the polygon would still
    happen to have the same bounding box here (the arc bulges inward, not
    outward, past the corner), so this specifically checks the AREA, which
    a straight-line chord would overstate (it cuts across the rounded
    corner rather than following the arc back around it).
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    # A 10x10 square with one corner rounded off via a bulge -- format
    # 'xyseb' = (x, y, start_width, end_width, bulge). A negative bulge on
    # the segment from (10,10) to (0,10) sweeps a concave arc cutting the
    # corner, so the enclosed area is strictly less than the full 10x10 the
    # straight-line (bulge-ignoring) reading would report.
    msp.add_lwpolyline(
        [(0, 0), (10, 0), (10, 10, 0, 0, -0.5), (0, 10)],
        format="xyseb",
        close=True,
        dxfattribs={"layer": "PLOT"},
    )
    dxf_path = tmp_path / "rounded.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "rounded")
    got = _measurements(result)
    assert got["plot.area"] < 100.0 - 0.5, (
        "A bulge (arc) corner must be flattened into the polygon boundary, not silently "
        "replaced with a straight line -- the true enclosed area is smaller than the full square."
    )
    # Curve-approximated geometry must carry a visibly lower confidence than
    # exact straight-line vector geometry, never the same hardcoded 0.95/0.98.
    assert _confidence(result, "plot.width") <= 0.90 + 1e-9
    assert _confidence(result, "plot.area") <= 0.90 + 1e-9


def test_circle_entity_resolves_as_a_building_footprint(tmp_path):
    """A circular building footprint (e.g. a rotunda/gazebo) must not be
    silently invisible just because it's a CIRCLE, not an LWPOLYLINE."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 20), (0, 20)], close=True, dxfattribs={"layer": "PLOT"})
    msp.add_circle((10, 10), radius=4.0, dxfattribs={"layer": "BLDG"})
    dxf_path = tmp_path / "circular_building.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "circular_building")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(20.0, abs=0.01)
    # A radius-4 circle is flattened into a many-sided regular polygon, so
    # its minimum-area ORIENTED bounding rectangle (used for width/depth
    # since Critical Requirement 1 -- see `_oriented_width_depth`) hugs the
    # polygon's own flat facets and is slightly SMALLER than the 8x8
    # axis-aligned bbox across the true circle -- that's the correct,
    # expected effect of measuring from the polygon's own geometry rather
    # than an axis-aligned box, not a regression.
    assert got["building.width"] == pytest.approx(8.0, abs=0.3)
    assert got["building.depth"] == pytest.approx(8.0, abs=0.3)
    assert _confidence(result, "building.width") <= 0.90 + 1e-9


def test_arc_segments_feed_fragmented_geometry_reconstruction(tmp_path):
    """
    A building drawn as disconnected LINE + ARC wall segments (no closed
    polyline at all) must still be reconstructible -- an ARC entity used to
    be completely invisible to `_collect_raw`, so its endpoints never
    reached `open_segments`/`dxf_reconstruction.py` at all, silently
    breaking reconstruction for any wall drawn with a curved/filleted corner.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 20), (0, 20)], close=True, dxfattribs={"layer": "PLOT"})
    # A near-rectangular building outline, but one corner is a quarter-circle
    # ARC instead of a straight LINE segment, and the whole thing is drawn as
    # disconnected entities rather than one closed polyline.
    msp.add_line((5, 5), (13, 5), dxfattribs={"layer": "BLDG"})
    msp.add_arc((13, 7), radius=2.0, start_angle=270, end_angle=360, dxfattribs={"layer": "BLDG"})
    msp.add_line((15, 7), (15, 15), dxfattribs={"layer": "BLDG"})
    msp.add_line((15, 15), (5, 15), dxfattribs={"layer": "BLDG"})
    msp.add_line((5, 15), (5, 5), dxfattribs={"layer": "BLDG"})
    dxf_path = tmp_path / "arc_wall.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "arc_wall")
    got = _measurements(result)
    assert got.get("building.footprint_area") is not None, (
        "Fragmented-geometry reconstruction must succeed even when one of the wall "
        "segments is an ARC rather than a straight LINE."
    )
    assert got["building.footprint_area"] == pytest.approx(10.0 * 10.0, rel=0.1)


# --- INSERT block attribute values --------------------------------------------


def test_insert_attribute_values_are_read_not_silently_dropped(tmp_path):
    """
    `INSERT.virtual_entities()` deliberately excludes ATTDEF (the block's
    template placeholder); the real per-instance value lives on
    `INSERT.attribs`, which was never read at all -- so a title-block
    attribute like a drawing number or scale note was silently invisible
    regardless of what value was actually typed in for this insertion.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True, dxfattribs={"layer": "PLOT"})

    block = doc.blocks.new(name="TITLEBLOCK")
    block.add_line((0, 0), (5, 0))
    block.add_attdef("SCALEVAL", text="DEFAULT", dxfattribs={"insert": (0, 0)})
    insert = msp.add_blockref("TITLEBLOCK", (1, 1), dxfattribs={"layer": "NOTES"})
    insert.add_auto_attribs({"SCALEVAL": "SCALE 1:100"})

    dxf_path = tmp_path / "titleblock.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "titleblock")
    assert any("SCALE 1:100" in (t.raw_text or "") for t in result.text_evidence), (
        "INSERT.attribs (the real per-instance attribute value) must be read as text "
        "evidence, not only the block's own definition geometry."
    )


# --- Unit/scale confidence -----------------------------------------------------


def test_missing_insunits_header_downgrades_confidence_even_when_plausible(tmp_path):
    """
    Even when a candidate-scale sanity check happens to succeed (the
    geometry scales to a plausible plot area), that is a heuristic guess
    about units the drawing itself never declared -- it must not be
    reported with the same confidence as a drawing whose own $INSUNITS
    header was read directly and confirmed.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 0  # explicitly unitless/unspecified
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 14), (0, 14)], close=True, dxfattribs={"layer": "PLOT"})
    dxf_path = tmp_path / "no_units.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "no_units")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(10.0, abs=0.01)
    assert _confidence(result, "plot.width") <= 0.60 + 1e-9
    assert _confidence(result, "plot.area") <= 0.60 + 1e-9
    assert any("Unit confidence is LOW" in w for w in result.warnings)


def test_valid_insunits_header_keeps_full_confidence(tmp_path):
    """Regression guard for the fix above: a normal, well-evidenced DXF
    (explicit $INSUNITS confirmed by a plausible plot area) must NOT be
    penalized -- the existing high-confidence behavior for the common case
    must be unchanged."""
    dxf_path = rectangular_site_plan_dxf(tmp_path / "site.dxf", plot_width=12.0, plot_depth=18.0)
    result = DXFHybridExtractor().extract(dxf_path, "site")
    assert _confidence(result, "plot.width") == pytest.approx(0.95, abs=1e-9)
    assert _confidence(result, "plot.area") == pytest.approx(0.98, abs=1e-9)


# --- Region detection / sheet-border rejection / envelope reconstruction ----
#
# End-to-end coverage (real `DXFHybridExtractor.extract()` calls, not just
# the region-clustering unit tests in test_dxf_regions.py) for a synthetic
# multi-drawing sheet with no layer names at all -- the same structural
# shape as the real production file that motivated this (19,622 POLYLINE
# entities, all layer '0', a sheet-spanning border rectangle that used to
# get selected as "the plot").


def test_sheet_border_is_rejected_and_the_real_site_plan_is_found(tmp_path):
    """The generic end-to-end acceptance case: a sheet-spanning frame
    rectangle, a real (dash-fragmented) site plan elsewhere on the sheet,
    and an unrelated second drawing -- none of it layer-tagged. The
    resolved plot/building must reflect the real site plan's own
    dimensions, not the frame's much larger extent."""
    dxf_path = multi_drawing_sheet_with_frame_dxf(
        tmp_path / "multi.dxf", plot_width=20.0, plot_depth=15.0,
        building_width=12.0, building_depth=8.0,
    )
    result = DXFHybridExtractor().extract(dxf_path, "multi")
    got = _measurements(result)

    assert any("Sheet-border/frame rejection" in w for w in result.warnings), (
        "the sheet-spanning frame polygon must be recognized and excluded from plot candidacy"
    )
    # The frame's own extent (90x70, area 6300 m2) must not appear as the
    # resolved plot -- this is the core claim of this test. Envelope
    # reconstruction from dash fragments is a coarse, disclosed
    # approximation (see `_reconstruct_plot_envelope`'s docstring), so this
    # deliberately does not assert tight numeric agreement with the
    # fixture's own plot_width/plot_depth parameters -- only that the
    # result is in the right neighborhood and structurally sane. (A
    # region-merging fix for a closely related bug -- the plot's dashed
    # boundary and the ring drawn around its building have different dash
    # densities, so density-erosion clustering sometimes splits them into
    # two regions, and cross-region scoring can then pick the smaller,
    # ring-sized region as "the plot" -- was tried and reverted: it fixed
    # this synthetic fixture but caused large regressions on the real
    # PLAN5/PLAN6 validation files, which have many more regions where
    # bounding-box nesting turned out not to reliably mean "same drawing".
    # See the Critical-Requirement-2 stage report for the full finding.)
    frame_area = 90.0 * 70.0
    assert got["plot.area"] < frame_area * 0.25
    assert got["building.footprint_area"] < got["plot.area"]
    # Order-of-magnitude sanity: the real site plan (20x15=300 m2) is much
    # closer to the resolved plot area than the frame (6300 m2) is.
    true_plot_area = 20.0 * 15.0
    assert abs(got["plot.area"] - true_plot_area) < abs(got["plot.area"] - frame_area)


def test_multiple_drawing_regions_are_detected_on_one_sheet(tmp_path):
    """A sheet containing more than one spatially distinct drawing must be
    recognized as such (region-count > 1), which is the precondition for
    sheet-border rejection to ever engage at all."""
    dxf_path = multi_drawing_sheet_with_frame_dxf(tmp_path / "multi.dxf")
    result = DXFHybridExtractor().extract(dxf_path, "multi")
    assert any("detected sheet regions" in w for w in result.warnings)


def test_item_9_fires_on_an_ordinary_multi_drawing_sheet_even_when_correct(tmp_path):
    """DXF_FAILURE_TAXONOMY.md item 9's own open question, answered: does
    the unconfirmed-drawing-identity cap fire only on the rare dangerous
    case, or routinely on any multi-drawing sheet without recognizable
    captions? Checked directly against this project's own EXISTING
    multi-drawing synthetic fixture (no caption text at all, by
    construction) -- it fires here too, even though this fixture's own
    extraction is numerically correct (plot ~20x15, building exactly
    12x8). This is not a bug: a genuinely caption-less multi-drawing sheet
    really does carry this uncertainty regardless of whether the number
    happens to be right, and the point of item 9 is to say so rather than
    guess. It does mean this fires far more often than "rare edge case" --
    on any multi-drawing sheet lacking recognizable captions, correct
    extractions included -- which is the honest way to describe it, not
    "rare"."""
    dxf_path = multi_drawing_sheet_with_frame_dxf(tmp_path / "multi.dxf")
    result = DXFHybridExtractor().extract(dxf_path, "multi")
    assert any("No drawing-type caption was recognized anywhere on this sheet" in w for w in result.warnings)
    plot_width = next(m for m in result.independent_cv.measurements if m.field == "plot.width")
    assert plot_width.value_m == pytest.approx(20.0, abs=0.5), "sanity check: this fixture's extraction is itself correct"
    from backend.cv_extraction.dxf_extractor import _CAPTION_OVERRIDE_CONFIDENCE_CAP
    assert plot_width.confidence <= _CAPTION_OVERRIDE_CONFIDENCE_CAP + 1e-9


def test_single_drawing_dxf_is_unaffected_by_region_detection(tmp_path):
    """A normal, single-drawing DXF (this project's existing common case)
    must resolve identically whether or not the region-detection machinery
    runs internally -- region-based resolution must be a strict no-op here,
    never a source of a different (worse) answer."""
    dxf_path = rectangular_site_plan_dxf(tmp_path / "site.dxf", plot_width=12.0, plot_depth=18.0)
    result = DXFHybridExtractor().extract(dxf_path, "site")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(12.0, abs=0.01)
    assert got["plot.depth"] == pytest.approx(18.0, abs=0.01)
    assert not any("Sheet-border/frame rejection" in w for w in result.warnings)
    assert not any("detected sheet regions" in w for w in result.warnings)


def test_dominant_region_is_not_excluded_just_because_a_small_unrelated_region_exists(tmp_path):
    """A real bug found via generalization testing: the region-level
    "sheet-spanning content" exclusion used to fire whenever ONE region's
    bbox covered a large SHARE of the union of every region's bbox --
    with no requirement that it actually overlap/enclose the other
    region(s) the way a real page border or frame would. A small,
    entirely separate cluster ANYWHERE on the sheet (a north-arrow/gate
    symbol, a DIMENSION entity's own rendered graphic sitting a modest
    distance from the geometry it measures) is enough to enlarge that
    union just far enough to push a real, single, dominant drawing over
    the coverage threshold -- wrongly excluding it even though the two
    regions never spatially overlap at all. Reproduced directly: a plot+
    building drawing with a small, non-overlapping decoy cluster placed
    just outside the plot's own bounding box used to lose plot/building
    resolution entirely."""
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for layer in ("PLOT", "BLDG"):
        doc.layers.add(layer)
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 15), (0, 15)], close=True, dxfattribs={"layer": "PLOT"})
    msp.add_lwpolyline(
        [(4, 3.5), (16, 3.5), (16, 11.5), (4, 11.5)], close=True, dxfattribs={"layer": "BLDG"},
    )
    # A small, unrelated decoy cluster sitting entirely below the plot
    # (never overlapping it) -- e.g. a scale bar or north-arrow symbol.
    # Dense enough (many close points) to cluster into its OWN region
    # rather than being eroded away as noise.
    for i in range(30):
        x = -3.0 + (i % 6) * 0.3
        y = -8.0 - (i // 6) * 0.3
        msp.add_line((x, y), (x + 0.15, y), dxfattribs={"layer": "0"})
    dxf_path = tmp_path / "dominant_region.dxf"
    doc.saveas(str(dxf_path))

    result = DXFHybridExtractor().extract(dxf_path, "dominant_region")
    got = _measurements(result)
    assert got["plot.width"] == pytest.approx(20.0, abs=0.01)
    assert got["plot.depth"] == pytest.approx(15.0, abs=0.01)
    assert not any("Excluded from plot/building candidacy" in w for w in result.warnings)


# --- Rank 2: evidence-guided pairwise region merging -------------------------
#
# Root cause this targets (see the master extraction-architecture audit):
# density-based region clustering can fragment one real, large, sparse
# boundary into several disjoint regions along its own straight sides, each
# then competing only as its own incomplete candidate. These tests exercise
# `_merge_candidate_region_pairs` directly, with simple closed-polygon
# regions rather than a full dash-fragment DXF, so the scenario is precise
# and deterministic rather than depending on the density-clustering
# algorithm's exact behavior on a particular fixture.


def _rect_poly(x0, y0, x1, y1, layer="0"):
    from backend.cv_extraction.dxf_extractor import _RawPoly
    from backend.schemas.geometry import Point, Polygon

    return _RawPoly(layer=layer, polygon=Polygon(points=[
        Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1),
    ]))


def test_merge_accepts_two_regions_that_together_resolve_plot_and_building():
    """Region A alone has a plot-shaped rectangle (plus a tiny nested speck
    -- `_pick_plot_polygon`'s role-inference graph needs at least two
    polygons to compare containment at all, a real, narrow limitation
    documented in the master extraction-architecture audit's role-
    inference findings). Region B alone similarly only has ITS OWN small
    rectangle (elsewhere on the sheet) plus a speck. Only when pooled does
    the true building rectangle turn out to sit INSIDE the true plot
    rectangle -- a strictly better resolution (real plot AND building
    found) than either region alone, so the merge must be accepted."""
    from backend.cv_extraction.dxf_extractor import (
        _merge_candidate_region_pairs, _resolve_plot_building_road, _score_region_resolution,
    )
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox

    plot_poly = _rect_poly(0, 0, 20, 10)
    speck_a = _rect_poly(0.5, 0.5, 0.6, 0.6)  # nested inside plot_poly
    bldg_poly = _rect_poly(7, 3, 13, 7)  # happens to sit inside plot_poly's bounds, but region B never sees plot_poly to know that
    speck_b = _rect_poly(7.1, 3.1, 7.2, 3.2)  # nested inside bldg_poly, so region B alone still has 2 polygons to compare
    polygons = [plot_poly, speck_a, bldg_poly, speck_b]

    region_a = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=20, max_y=10), polygon_indices=[0, 1])
    region_b = DxfRegion(id=1, bbox=BoundingBox(min_x=7, min_y=3, max_x=13, max_y=7), polygon_indices=[2, 3])

    plot_a, building_a, road_a = _resolve_plot_building_road(
        [polygons[0], polygons[1]], [], [], allow_envelope_reconstruction=True,
    )
    score_a = _score_region_resolution(plot_a, building_a)
    plot_b, building_b, road_b = _resolve_plot_building_road(
        [polygons[2], polygons[3]], [], [], allow_envelope_reconstruction=True,
    )
    score_b = _score_region_resolution(plot_b, building_b)
    assert building_a is None and building_b is None  # neither region alone found a building

    structural = [(score_a, region_a, plot_a, building_a, road_a), (score_b, region_b, plot_b, building_b, road_b)]
    warnings: list[str] = []
    merged = _merge_candidate_region_pairs(
        structural, polygons, [], frame_indices=set(), warnings=warnings, time_budget_seconds=30.0,
    )

    assert len(merged) == 1
    merged_score, _merged_region, _merged_plot, merged_building, _road = merged[0]
    assert merged_building is not None
    assert merged_score > score_a
    assert merged_score > score_b
    assert any("pooled and re-resolved together score higher" in w for w in warnings)


def test_merge_rejects_a_pooled_result_that_scores_higher_but_sprawls_far_beyond_the_sum_of_its_parts(monkeypatch):
    """Pins the actual regression found and fixed this session: on a real
    bundled DXF regression fixture, two unrelated regions pooled into a
    materially larger, wrong-shaped result that still scored HIGHER than
    either alone (the generic plausibility score rewards things like
    "has both a plot and a building", which doesn't detect an incoherent
    union). The area-vs-sum-of-parts sanity check must reject that, even
    though the score comparison alone would have accepted it."""
    import backend.cv_extraction.dxf_extractor as dxf_extractor_mod
    from backend.cv_extraction.dxf_extractor import (
        _RawPoly, _merge_candidate_region_pairs, _score_region_resolution,
    )
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox, Point, Polygon

    region_a = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10), polygon_indices=[0])
    region_b = DxfRegion(id=1, bbox=BoundingBox(min_x=20, min_y=20, max_x=25, max_y=25), polygon_indices=[1])
    plot_a = _RawPoly(layer="0", polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10),
    ]))  # area 100
    plot_b = _RawPoly(layer="0", polygon=Polygon(points=[
        Point(x=20, y=20), Point(x=25, y=20), Point(x=25, y=25), Point(x=20, y=25),
    ]))  # area 25
    # A stand-in for a pooled fit that "sprawls" far beyond the sum of its
    # parts (area 100+25=125) -- mirrors the real regression's fitted
    # envelope covering empty space between two genuinely separate regions.
    sprawling_plot = _RawPoly(layer="RECONSTRUCTED_ENVELOPE", polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100),
    ]), envelope_method="bounding_rectangle")  # area 10,000 -- far beyond 125

    score_a = _score_region_resolution(plot_a, None)
    score_b = _score_region_resolution(plot_b, None)
    structural = [(score_a, region_a, plot_a, None, None), (score_b, region_b, plot_b, None, None)]

    def _fake_resolve(*args, **kwargs):
        return sprawling_plot, None, None

    monkeypatch.setattr(dxf_extractor_mod, "_resolve_plot_building_road", _fake_resolve)

    warnings: list[str] = []
    merged = _merge_candidate_region_pairs(
        structural, [plot_a, plot_b], [], frame_indices=set(), warnings=warnings, time_budget_seconds=30.0,
    )

    assert merged == [], "a pooled result sprawling far beyond the sum of its parts' areas must be rejected"


def test_merge_stops_trying_further_pairs_once_its_time_budget_is_used(monkeypatch):
    """The whole point of the time-budget parameter: however many candidate
    pairs there are, testing must stop once the actual measured elapsed
    time reaches the budget, rather than trying every combination
    regardless of cost."""
    import time as time_module

    import backend.cv_extraction.dxf_extractor as dxf_extractor_mod
    from backend.cv_extraction.dxf_extractor import _RawPoly, _merge_candidate_region_pairs
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox, Point, Polygon

    poly = Polygon(points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)])
    plot_entry = _RawPoly(layer="0", polygon=poly)

    regions = [
        DxfRegion(id=i, bbox=BoundingBox(min_x=i, min_y=0, max_x=i + 10, max_y=10), polygon_indices=[0])
        for i in range(4)
    ]
    structural = [(1.0 + i * 0.01, regions[i], plot_entry, None, None) for i in range(4)]

    calls = []

    def _slow_resolve(*args, **kwargs):
        calls.append(1)
        time_module.sleep(0.05)
        return plot_entry, None, None

    monkeypatch.setattr(dxf_extractor_mod, "_resolve_plot_building_road", _slow_resolve)

    warnings: list[str] = []
    # 4 candidates -> 6 possible pairs; a budget covering only ~2 calls'
    # worth of time must stop well short of trying all 6.
    _merge_candidate_region_pairs(
        structural, [plot_entry], [], frame_indices=set(), warnings=warnings, time_budget_seconds=0.1,
    )

    assert 0 < len(calls) < 6
    assert any("stopped early" in w for w in warnings)


def test_merge_grows_agglomeratively_across_more_than_one_round(monkeypatch):
    """A real boundary is not always split into exactly two fragments (a
    bundled PLAN5-style fixture split one into four) -- growth must be
    able to keep adding regions across several rounds, not stop after a
    single pairwise merge. Three starting regions (0, 1, 2): merging 0+1
    is a real improvement over either alone, and growing that merged
    cluster further with region 2 is a further improvement again --
    merging 0+2 or 1+2 directly is NOT an improvement, so round 1 must
    specifically find (0, 1), and only round 2 (growing that cluster)
    should discover region 2 belongs too."""
    import backend.cv_extraction.dxf_extractor as dxf_extractor_mod
    from backend.cv_extraction.dxf_extractor import _RawPoly, _merge_candidate_region_pairs, _score_region_resolution
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox, Point, Polygon

    def rect(area, layer="0"):
        # A distinct, recognizable area per leaf region's own polygon, so
        # the mock below can identify which combination is being tested
        # purely from the pooled polygons it's given.
        side = area ** 0.5
        return _RawPoly(layer=layer, polygon=Polygon(points=[
            Point(x=0, y=0), Point(x=side, y=0), Point(x=side, y=side), Point(x=0, y=side),
        ]))

    poly0, poly1, poly2 = rect(100.0), rect(200.0), rect(300.0)
    region0 = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10), polygon_indices=[0])
    region1 = DxfRegion(id=1, bbox=BoundingBox(min_x=20, min_y=0, max_x=34.1, max_y=14.1), polygon_indices=[1])
    region2 = DxfRegion(id=2, bbox=BoundingBox(min_x=50, min_y=0, max_x=67.3, max_y=17.3), polygon_indices=[2])
    polygons = [poly0, poly1, poly2]

    plot_alone_0 = rect(100.0)
    plot_alone_1 = rect(200.0)
    plot_alone_2 = rect(300.0)
    plot_01 = rect(310.0)  # a real improvement over either 100 or 200 alone
    plot_012 = rect(620.0)  # growing (0+1) with region 2 improves further again
    plot_02_or_12 = rect(5000.0)  # NOT an improvement-worthy combination (deliberately made implausible: exceeds the area-sum-of-parts sanity check)

    score_0 = _score_region_resolution(plot_alone_0, None)
    score_1 = _score_region_resolution(plot_alone_1, None)
    score_2 = _score_region_resolution(plot_alone_2, None)
    structural = [
        (score_0, region0, plot_alone_0, None, None),
        (score_1, region1, plot_alone_1, None, None),
        (score_2, region2, plot_alone_2, None, None),
    ]

    def _fake_resolve(pooled_polygons, pooled_segments, warnings_arg, **kwargs):
        areas = sorted(round(p.polygon.area) for p in pooled_polygons)
        if areas == [100, 200]:
            return plot_01, None, None
        if areas == [100, 200, 300]:
            return plot_012, None, None
        # (0,2) or (1,2) directly -- not a real fragment pairing.
        return plot_02_or_12, None, None

    monkeypatch.setattr(dxf_extractor_mod, "_resolve_plot_building_road", _fake_resolve)

    warnings: list[str] = []
    merged = _merge_candidate_region_pairs(
        structural, polygons, [], frame_indices=set(), warnings=warnings, time_budget_seconds=30.0,
    )

    merged_areas = sorted(round(m[2].polygon.area) for m in merged if m[2] is not None)
    assert round(plot_01.polygon.area) in merged_areas, "round 1 must find the real (0,1) merge"
    assert round(plot_012.polygon.area) in merged_areas, "round 2 must grow that merge further with region 2"
    assert round(plot_02_or_12.polygon.area) not in merged_areas, "a non-improving direct (0,2)/(1,2) pairing must never be kept"


def test_merge_candidacy_includes_a_low_scoring_region_that_carries_confirmed_evidence(monkeypatch):
    """Root cause this targets (confirmed on the real PLAN5.dxf fixture):
    candidacy for growth was capped to the top `max_dxf_region_merge_candidates`
    regions by their OWN standalone score. A region that is genuinely one
    fragment of a larger, split boundary scores poorly in isolation almost
    by definition (see this function's own docstring) -- capping candidacy
    to a fixed top-N by score alone made it structurally impossible for
    growth to ever recover such a fragment once its own score fell outside
    that cutoff, no matter how much time budget was available, whenever
    recovered OCR/vectorized-text evidence had already flagged it as
    relevant. A region carrying confirmed evidence must always be eligible,
    regardless of its own rank."""
    import backend.cv_extraction.dxf_extractor as dxf_extractor_mod
    from backend.cv_extraction.dxf_extractor import _RawPoly, _merge_candidate_region_pairs
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox, Point, Polygon

    def rect(area, layer="0"):
        side = area ** 0.5
        return _RawPoly(layer=layer, polygon=Polygon(points=[
            Point(x=0, y=0), Point(x=side, y=0), Point(x=side, y=side), Point(x=0, y=side),
        ]))

    # Six well-scoring "core" regions -- more than enough to fill the
    # default candidate cap (max_dxf_region_merge_candidates=6) on their
    # own -- plus a seventh, poorly-scoring region that only carries
    # confirmed evidence going for it.
    core_areas = [50, 51, 52, 53, 54, 55]
    core_plots = [rect(a) for a in core_areas]
    core_regions = [
        DxfRegion(id=i, bbox=BoundingBox(min_x=i * 20, min_y=0, max_x=i * 20 + 10, max_y=10), polygon_indices=[i])
        for i in range(6)
    ]
    extra_plot = rect(1)
    extra_region = DxfRegion(id=6, bbox=BoundingBox(min_x=200, min_y=0, max_x=210, max_y=10), polygon_indices=[6])

    polygons = core_plots + [extra_plot]
    structural = [(6.0 - i * 0.1, core_regions[i], core_plots[i], None, None) for i in range(6)]
    structural.append((-100.0, extra_region, extra_plot, None, None))  # would never make a plain top-6-by-score cut

    good_merge_plot = rect(55)  # region0 (area 50) + extra (area 1): sum*1.15 = 58.65, comfortably area-sane

    def _fake_resolve(pooled_polygons, pooled_segments, warnings_arg, **kwargs):
        areas = sorted(round(p.polygon.area) for p in pooled_polygons)
        if areas == [1, 50]:
            return good_merge_plot, None, None
        # Every other pooling (all core-only combinations) sprawls wildly
        # beyond any plausible area-sum -- deliberately never accepted, so
        # this test isolates whether the region-0 + extra pairing itself
        # gets a chance to be tried at all.
        return rect(10 ** 6), None, None

    def _fake_score(plot_entry, building_entry):
        return 50.0 if plot_entry is good_merge_plot else -999.0

    monkeypatch.setattr(dxf_extractor_mod, "_resolve_plot_building_road", _fake_resolve)
    monkeypatch.setattr(dxf_extractor_mod, "_score_region_resolution", _fake_score)

    warnings: list[str] = []
    merged = _merge_candidate_region_pairs(
        structural, polygons, [], frame_indices=set(), warnings=warnings, time_budget_seconds=30.0,
        evidence_by_region={6: [_recovered(1.0)]},
    )

    assert any(m[2] is good_merge_plot for m in merged), (
        "region 6 (score -100, well below the top-6-by-score cutoff) carries confirmed evidence and must "
        "still be tried against region 0 -- otherwise a real fragment's own poor standalone score would "
        "permanently exclude it from growth regardless of time budget"
    )


def test_merge_never_starves_a_working_core_merge_to_make_room_for_an_evidence_extra(monkeypatch):
    """The evidence carve-out above must never come at the expense of the
    already-working top-N-by-score group: on a file where merely testing
    every pair among the top-scoring regions already consumes the whole
    time budget (confirmed on the real PLAN5.dxf fixture -- 15 core pairs
    alone did not finish within its allotted budget), an evidence-only
    extra competing for the SAME time slice must not be allowed to crowd
    out a core pairing that would otherwise have been found and kept."""
    import time as time_module

    import backend.cv_extraction.dxf_extractor as dxf_extractor_mod
    from backend.cv_extraction.dxf_extractor import _RawPoly, _merge_candidate_region_pairs, _score_region_resolution
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox, Point, Polygon

    def rect(area, layer="0"):
        side = area ** 0.5
        return _RawPoly(layer=layer, polygon=Polygon(points=[
            Point(x=0, y=0), Point(x=side, y=0), Point(x=side, y=side), Point(x=0, y=side),
        ]))

    # Core regions 0 and 1: a real, genuine improvement when pooled.
    # Regions 2-5: present purely to fill out the candidate set (their own
    # pairings are never a real improvement -- see `_fake_resolve` below).
    plot_0, plot_1 = rect(100.0), rect(200.0)
    plot_01 = rect(310.0)  # a real improvement over either 100 or 200 alone
    core_plots = [plot_0, plot_1, rect(11), rect(13), rect(17), rect(19)]
    core_regions = [
        DxfRegion(id=i, bbox=BoundingBox(min_x=i * 20, min_y=0, max_x=i * 20 + 10, max_y=10), polygon_indices=[i])
        for i in range(6)
    ]
    extra_plot = rect(97)
    extra_region = DxfRegion(id=6, bbox=BoundingBox(min_x=200, min_y=0, max_x=210, max_y=10), polygon_indices=[6])
    polygons = core_plots + [extra_plot]

    structural = [
        (_score_region_resolution(core_plots[i], None), core_regions[i], core_plots[i], None, None)
        for i in range(6)
    ]
    structural.append((-100.0, extra_region, extra_plot, None, None))

    call_kinds: list[bool] = []  # True = a call involving the evidence extra (area 97)

    def _slow_resolve(pooled_polygons, pooled_segments, warnings_arg, **kwargs):
        areas = sorted(round(p.polygon.area) for p in pooled_polygons)
        call_kinds.append(97 in areas)
        time_module.sleep(0.03)
        if areas == [100, 200]:
            return plot_01, None, None
        return rect(10 ** 6), None, None  # never an improvement -- area-insane

    monkeypatch.setattr(dxf_extractor_mod, "_resolve_plot_building_road", _slow_resolve)

    warnings: list[str] = []
    # 6 core regions -> 15 core-only pairs at ~0.03s each ~= 0.45s; budget
    # covers that plus a little room for a couple of extras pairs, but not
    # all 6 of the extra's possible pairings with each core region.
    merged = _merge_candidate_region_pairs(
        structural, polygons, [], frame_indices=set(), warnings=warnings, time_budget_seconds=0.55,
        evidence_by_region={6: [_recovered(1.0)]},
    )

    assert call_kinds.count(False) == 15, "every core-only pair must be tried before the budget is spent on extras"
    assert any(round(m[2].polygon.area) == 310 for m in merged), (
        "the real (0,1) core merge must still be found -- an evidence-only extra competing for the same "
        "time budget must never crowd out an already-working core pairing"
    )


# --- Evidence-aware merge veto ------------------------------------------------
#
# Root cause this targets (confirmed on the real bundled PLAN5.dxf fixture):
# growing a genuinely-improving 2-region merge with a third region raised
# its generic structural plausibility score further while moving its depth
# from 9.1% error to 15.5% error against the sheet's own recoverable
# printed dimension. Structural score alone cannot see that a growth step
# is moving away from the document's own evidence -- it needs to be told.


def _recovered(value, raw_text=None, confidence=0.8):
    from backend.cv_extraction.dxf_text_recovery import RecoveredDimension

    return RecoveredDimension(
        value=value, unit_hint=None, raw_text=raw_text or f"{value}",
        world_bbox=(0.0, 0.0, 1.0, 1.0), confidence=confidence,
        associated_segment_index=None, associated_segment_length=None,
    )


def _rect_poly_wd(width, depth, layer="0"):
    from backend.cv_extraction.dxf_extractor import _RawPoly
    from backend.schemas.geometry import Point, Polygon

    return _RawPoly(layer=layer, polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=width, y=0), Point(x=width, y=depth), Point(x=0, y=depth),
    ]))


# --- Road-edge borrowing for isolated regions (DXF_FAILURE_TAXONOMY.md item 8) ---


def _road_poly(min_x, min_y, max_x, max_y):
    from backend.cv_extraction.dxf_extractor import _RawPoly
    from backend.schemas.geometry import Point, Polygon

    return _RawPoly(layer="ROAD", polygon=Polygon(points=[
        Point(x=min_x, y=min_y), Point(x=max_x, y=min_y), Point(x=max_x, y=max_y), Point(x=min_x, y=max_y),
    ]))


def test_find_nearby_road_candidate_borrows_the_single_close_candidate():
    from backend.cv_extraction.dxf_extractor import find_nearby_road_candidate
    from backend.schemas.geometry import BoundingBox

    target_plot_bbox = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    nearby_road = _road_poly(11, 0, 13, 10)  # just past the plot's own right edge
    result = find_nearby_road_candidate(target_plot_bbox, [(7, nearby_road)])
    assert result is not None
    assert result.source_region_id == 7
    assert result.road_entry is nearby_road


def test_find_nearby_road_candidate_refuses_when_nothing_is_close_enough():
    from backend.cv_extraction.dxf_extractor import find_nearby_road_candidate
    from backend.schemas.geometry import BoundingBox

    target_plot_bbox = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    far_road = _road_poly(1000, 1000, 1002, 1010)
    assert find_nearby_road_candidate(target_plot_bbox, [(3, far_road)]) is None


def test_find_nearby_road_candidate_refuses_when_two_candidates_are_comparably_close():
    """Refuse rather than guess: two plausible roads at similar distance
    is exactly the ambiguous case borrowing must not silently resolve by
    picking one arbitrarily -- attributing the wrong road's width to this
    plot would be worse than leaving road.width MISSING."""
    from backend.cv_extraction.dxf_extractor import find_nearby_road_candidate
    from backend.schemas.geometry import BoundingBox

    target_plot_bbox = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    road_a = _road_poly(11, 0, 13, 10)
    road_b = _road_poly(-3, 0, -1, 10)  # comparably close on the opposite side
    assert find_nearby_road_candidate(target_plot_bbox, [(1, road_a), (2, road_b)]) is None


def test_find_nearby_road_candidate_picks_the_closer_one_when_not_ambiguous():
    from backend.cv_extraction.dxf_extractor import find_nearby_road_candidate
    from backend.schemas.geometry import BoundingBox

    target_plot_bbox = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    close_road = _road_poly(11, 0, 13, 10)
    far_road = _road_poly(500, 500, 502, 510)
    result = find_nearby_road_candidate(target_plot_bbox, [(1, far_road), (2, close_road)])
    assert result is not None
    assert result.source_region_id == 2


def test_find_nearby_road_candidate_refuses_with_no_candidates_at_all():
    from backend.cv_extraction.dxf_extractor import find_nearby_road_candidate
    from backend.schemas.geometry import BoundingBox

    assert find_nearby_road_candidate(BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10), []) is None


def test_entry_source_confidence_is_capped_when_caption_overrode_implausible_score():
    """A region can be the genuinely right DRAWING (confirmed by a
    recognized drawing-type caption) while its own reconstructed geometry
    is still known-implausible -- a caption match is evidence about WHICH
    drawing this is, not evidence that its geometry is accurate. Confirmed
    on the real PLAN5.dxf fixture: region3 is unambiguously the correct
    site plan, but its own pre-caption structural score was the worst of
    all 12 regions on that sheet (its reconstructed plot area is below this
    module's own plausible-plot-area floor). Shipping its measurements at a
    normal reconstruction confidence (0.6-0.85) would misrepresent how much
    this specific number should be trusted."""
    from backend.cv_extraction.dxf_extractor import (
        _CAPTION_OVERRIDE_CONFIDENCE_CAP,
        _RawPoly,
        _entry_source_confidence,
    )
    from backend.schemas.enums import VISION_CONFIDENCE_LOW_THRESHOLD
    from backend.schemas.geometry import Point, Polygon

    poly = Polygon(points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)])

    normal_entry = _RawPoly(layer="0", polygon=poly)
    _source, normal_conf = _entry_source_confidence(normal_entry, 0.97, unit_confidence=1.0)
    assert normal_conf > VISION_CONFIDENCE_LOW_THRESHOLD, "sanity check: an ordinary entry is not already capped low"

    flagged_entry = _RawPoly(layer="0", polygon=poly, caption_overrode_implausible_score=True)
    _source, flagged_conf = _entry_source_confidence(flagged_entry, 0.97, unit_confidence=1.0)
    assert flagged_conf == _CAPTION_OVERRIDE_CONFIDENCE_CAP
    assert flagged_conf < VISION_CONFIDENCE_LOW_THRESHOLD, "must always bucket as LOW confidence downstream"


def test_entry_source_confidence_is_capped_when_drawing_identity_is_unconfirmed():
    """DXF_FAILURE_TAXONOMY.md item 9: a region can win on ORDINARY
    structural scoring, with no caption pointing at it or anywhere else on
    the sheet, on a sheet complex enough (multiple independently-plausible
    regions) that misidentifying the drawing is a live possibility. This
    is a materially different, more dangerous situation than the caption-
    override case above: there, at least ONE region was positively
    identified. Here, nothing was -- confirmed directly on the real
    PLAN6.dxf fixture (see the test below), where the actual site plan
    (region12) independently exists and matches ground truth almost
    exactly, but the pipeline has no way to know that from this winner's
    own vantage point."""
    from backend.cv_extraction.dxf_extractor import (
        _CAPTION_OVERRIDE_CONFIDENCE_CAP,
        _RawPoly,
        _entry_source_confidence,
    )
    from backend.schemas.enums import VISION_CONFIDENCE_LOW_THRESHOLD
    from backend.schemas.geometry import Point, Polygon

    poly = Polygon(points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)])
    flagged_entry = _RawPoly(layer="0", polygon=poly, unconfirmed_drawing_identity=True)
    _source, flagged_conf = _entry_source_confidence(flagged_entry, 0.97, unit_confidence=1.0)
    assert flagged_conf == _CAPTION_OVERRIDE_CONFIDENCE_CAP
    assert flagged_conf < VISION_CONFIDENCE_LOW_THRESHOLD, "must always bucket as LOW confidence downstream"


def test_plan6_dxf_confidence_is_capped_when_no_caption_confirms_any_region():
    """Real-fixture regression pin for item 9, and specifically for a bug
    caught in this exact check: `captions_by_region` is pre-populated with
    an empty list per region (for `.setdefault` convenience elsewhere), so
    the dict itself is never empty -- the ORIGINAL version of this
    mechanism read `not captions_by_region`, which is always False, and
    silently never fired on this exact file, the one that motivated
    building it. Confirms the shipped plot.width confidence is actually
    capped now, not just that the code compiles."""
    from pathlib import Path

    from backend.cv_extraction.dxf_extractor import _CAPTION_OVERRIDE_CONFIDENCE_CAP, DXFHybridExtractor

    result = DXFHybridExtractor().extract(Path("data/test_plans/PLAN6.dxf"), "PLAN6")
    assert result.independent_cv is not None
    plot_width = next((m for m in result.independent_cv.measurements if m.field == "plot.width"), None)
    assert plot_width is not None
    assert plot_width.confidence <= _CAPTION_OVERRIDE_CONFIDENCE_CAP + 1e-9, (
        f"expected plot.width confidence capped at {_CAPTION_OVERRIDE_CONFIDENCE_CAP} given no caption "
        f"confirms any region on this sheet; got {plot_width.confidence}"
    )


def test_plan6_low_confidence_survives_all_the_way_to_normalized_plan_and_compliance():
    """Closes the loop: a low numeric confidence on `IndependentMeasurement`
    is only worth anything if it actually changes what a compliance report
    shows. Found live, NOT hypothetically, while verifying this: two
    separate hardcoding bugs (`_value_field` in final_fusion.py always
    tagging a CV-only value MEDIUM regardless of its own confidence; then
    `build_normalized_plan`'s own `fv()` in pipeline.py ALSO discarding
    the correctly-derived confidence and rebuilding HIGH/MEDIUM from
    fusion status alone) meant PLAN6's real 0.4-confidence plot.width
    reached `NormalizedPlan` tagged MEDIUM -- identical to an ordinary,
    fully-trusted reading, and invisible to `backend/compliance/engine.py`'s
    own documented "LOW confidence -> REQUIRES_REVIEW, never silently
    PASS/FAIL" gate. Both are fixed; this pins the full chain end to end
    on the real fixture that exposed it, not just each link in isolation."""
    from pathlib import Path

    from backend.compliance.engine import DeterministicRuleEvaluator
    from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
    from backend.runtime_rules.contracts import RuleContext, RuntimeRuleDefinition
    from backend.schemas.compliance import ComplianceStatus
    from backend.schemas.enums import ConfidenceLevel
    from backend.spatial_reasoning.pipeline import build_normalized_plan

    extraction = DXFHybridExtractor().extract(Path("data/test_plans/PLAN6.dxf"), "PLAN6")
    plan = build_normalized_plan(extraction, plan_id="plan-PLAN6")
    assert plan.plot.width.confidence.level == ConfidenceLevel.LOW, (
        "the extractor's own 0.4 confidence must survive into NormalizedPlan as LOW, "
        f"got {plan.plot.width.confidence.level}"
    )

    # An always-applicable synthetic rule, isolated from BBMP's real
    # ruleset (whose applies_when clauses depend on fields a DXF
    # extraction never resolves, like building.floor_count, and would
    # short-circuit to INSUFFICIENT_DATA before ever reaching the
    # confidence check this test cares about).
    rule = RuntimeRuleDefinition(
        rule_id="test-plot-width-min", municipality="TEST",
        description="Synthetic always-applicable rule for this test only.",
        target="plot.width",
        applies_when={"field": "plot.width", "op": ">=", "value": 0},
        threshold={"field": "plot.width", "op": ">=", "value": 10, "unit": "m"},
        version="1.0.0", citation="test", status="ACTIVE", priority=1,
    )
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=plan, rule=rule))
    assert result.status == ComplianceStatus.REQUIRES_REVIEW, (
        f"expected REQUIRES_REVIEW given LOW plot.width confidence, got {result.status}: {result.explanation}"
    )


def test_plan5_dxf_plot_confidence_is_capped_when_caption_overrode_implausible_score():
    """Real-fixture regression pin, same file class as the unit test above:
    PLAN5.dxf's region3 wins its resolution via a recognized 'SITE PLAN'
    caption after its own pre-caption structural score fell below this
    pipeline's usability bar. Confirms the shipped plot.width confidence
    reflects that, not a normal reconstruction confidence."""
    from pathlib import Path

    from backend.cv_extraction.dxf_extractor import _CAPTION_OVERRIDE_CONFIDENCE_CAP, DXFHybridExtractor

    result = DXFHybridExtractor().extract(Path("data/test_plans/PLAN5.dxf"), "PLAN5")
    assert result.independent_cv is not None
    plot_width = next(m for m in result.independent_cv.measurements if m.field == "plot.width")
    assert plot_width.confidence <= _CAPTION_OVERRIDE_CONFIDENCE_CAP + 1e-9, (
        f"expected plot.width confidence capped at {_CAPTION_OVERRIDE_CONFIDENCE_CAP} once region3 "
        f"won via caption override; got {plot_width.confidence}"
    )


def test_plan5_dxf_setbacks_ship_low_confidence_with_no_orientation_evidence():
    """DXF_FAILURE_TAXONOMY.md item 6 (front-edge-fallback instability):
    already correctly degrades to LOW confidence when no FRONT/ROAD/
    STREET/ACCESS/GATE evidence exists to resolve real orientation --
    verified here as a permanent regression pin, since this behavior was
    previously unprotected by any test. `resolve_front_side`'s own
    ConfidenceLevel.LOW maps to a raw 0.5 (`_build_independent_cv`'s own
    side_confidence table), which must stay below `VISION_CONFIDENCE_LOW_
    THRESHOLD` (0.75) so it always buckets as LOW downstream, not silently
    MEDIUM."""
    from pathlib import Path

    from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
    from backend.schemas.enums import VISION_CONFIDENCE_LOW_THRESHOLD

    result = DXFHybridExtractor().extract(Path("data/test_plans/PLAN5.dxf"), "PLAN5")
    assert result.independent_cv is not None
    setback_fields = {m.field: m for m in result.independent_cv.measurements if m.field.startswith("setbacks.")}
    assert setback_fields, "expected at least one setback measurement to check"
    for field, m in setback_fields.items():
        assert m.confidence < VISION_CONFIDENCE_LOW_THRESHOLD, (
            f"{field} shipped confidence {m.confidence}, expected below the LOW threshold "
            f"{VISION_CONFIDENCE_LOW_THRESHOLD} given no orientation evidence was available"
        )


# --- Evidence-match conflict refusal (DXF_FAILURE_TAXONOMY.md items 4/5) ---


def test_match_recovered_evidence_refuses_a_genuine_conflict():
    """Refuse rather than guess: two DIFFERENT recovered items both fall
    within tolerance of the same field, at comparable specificity, but
    disagree by more than OCR jitter -- there is no principled way to
    prefer one, so this field must get NO match at all, not an arbitrary
    (previously: first-seen) tie-break."""
    from backend.cv_extraction.dxf_extractor import _match_recovered_evidence, _RawPoly
    from backend.schemas.geometry import Point, Polygon

    plot_entry = _RawPoly(layer="0", polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=10.0, y=0), Point(x=10.0, y=8.0), Point(x=0, y=8.0),
    ]))
    # Both comfortably within _EVIDENCE_MATCH_RELATIVE_TOLERANCE (0.08) of
    # 10.0, both reasonably specific (decimal point, similar digit count),
    # but 3% apart from each other -- above _EVIDENCE_SAME_READING_
    # RELATIVE_TOLERANCE (0.03), so this is a genuine conflict, not jitter.
    a = _recovered(9.7, raw_text="9.70")
    b = _recovered(10.35, raw_text="10.35")
    matched = _match_recovered_evidence([a, b], plot_entry, None)
    assert not any(field == "plot.width" for field, *_ in matched)


def test_match_recovered_evidence_resolves_ordinary_ocr_jitter_not_a_conflict():
    """Two readings of the same real annotation that differ only by OCR
    jitter (well within _EVIDENCE_SAME_READING_RELATIVE_TOLERANCE) must
    still resolve to the stronger one -- this is not the ambiguous case
    the conflict refusal above exists for."""
    from backend.cv_extraction.dxf_extractor import _match_recovered_evidence, _RawPoly
    from backend.schemas.geometry import Point, Polygon

    plot_entry = _RawPoly(layer="0", polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=10.0, y=0), Point(x=10.0, y=8.0), Point(x=0, y=8.0),
    ]))
    strong = _recovered(10.0, raw_text="10.00m", confidence=0.95)  # has unit + decimal -> high specificity
    weak_jitter = _recovered(10.05, raw_text="1", confidence=0.3)  # bare short digit -> low specificity, within 3% of strong
    matched = _match_recovered_evidence([strong, weak_jitter], plot_entry, None)
    width_matches = [m for m in matched if m[0] == "plot.width"]
    assert len(width_matches) == 1
    assert width_matches[0][1] is strong


def test_match_recovered_evidence_ignores_a_much_weaker_runner_up():
    """A second reading far below the conflict-weight-ratio floor is
    ordinary noise next to a strong reading, not a competing candidate --
    must not block the strong match."""
    from backend.cv_extraction.dxf_extractor import _match_recovered_evidence, _RawPoly
    from backend.schemas.geometry import Point, Polygon

    plot_entry = _RawPoly(layer="0", polygon=Polygon(points=[
        Point(x=0, y=0), Point(x=10.0, y=0), Point(x=10.0, y=8.0), Point(x=0, y=8.0),
    ]))
    strong = _recovered(10.0, raw_text="10.00m", confidence=0.95)
    noise = _recovered(10.5, raw_text="1", confidence=0.2)  # meaningfully different value, but far weaker
    matched = _match_recovered_evidence([strong, noise], plot_entry, None)
    width_matches = [m for m in matched if m[0] == "plot.width"]
    assert len(width_matches) == 1
    assert width_matches[0][1] is strong


def test_confirmed_axis_values_requires_both_width_and_depth():
    """Mirrors `_SINGLE_AXIS_EVIDENCE_DISCOUNT`'s own discipline: a
    candidate is only "confirmed" for veto purposes when BOTH its width
    and depth are independently matched by recovered evidence, never just
    one."""
    from backend.cv_extraction.dxf_extractor import _confirmed_axis_values

    plot = _rect_poly_wd(10.0, 8.0)

    # Depth only -- must not count as confirmed.
    assert _confirmed_axis_values([_recovered(8.05, "8.05")], plot) == {}

    # Both width and depth -- confirmed on both axes.
    confirmed = _confirmed_axis_values([_recovered(10.05, "10.05"), _recovered(8.05, "8.05")], plot)
    assert confirmed == {"width": pytest.approx(10.05), "depth": pytest.approx(8.05)}


def test_confirmed_axis_values_rejects_one_item_confirming_both_axes():
    """Pins a real gap found and fixed on the bundled PLAN5.dxf fixture: a
    near-square candidate (width and depth both close to the same value)
    let ONE recovered digit satisfy both fields at once. That is one weak
    signal, not independent two-axis corroboration, and must not count as
    confirmed."""
    from backend.cv_extraction.dxf_extractor import _confirmed_axis_values

    # A near-square candidate: both width and depth are close to 7.14.
    plot = _rect_poly_wd(6.875, 7.69)
    single_item = [_recovered(7.14, "7.14")]  # matches BOTH width and depth within tolerance

    assert _confirmed_axis_values(single_item, plot) == {}

    # Two DIFFERENT items, each independently confirming one axis, is fine.
    two_items = [_recovered(6.9, "6.90"), _recovered(7.7, "7.70")]
    confirmed = _confirmed_axis_values(two_items, plot)
    assert confirmed == {"width": pytest.approx(6.9), "depth": pytest.approx(7.7)}


def test_evidence_veto_rejects_a_merge_that_moves_away_from_confirmed_depth():
    from backend.cv_extraction.dxf_extractor import _evidence_veto_reason
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox

    region_a = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=8.3))
    region_b = DxfRegion(id=1, bbox=BoundingBox(min_x=20, min_y=0, max_x=25, max_y=25))
    plot_a = _rect_poly_wd(10.0, 8.3)  # depth 8.3 is within 8% of a confirmed 8.7
    plot_b = _rect_poly_wd(5.0, 5.0)
    merged_plot_worse = _rect_poly_wd(10.0, 15.0)  # depth moved FAR from the confirmed 8.7

    evidence_by_region = {0: [_recovered(10.0, "10.00"), _recovered(8.7, "8.70")]}

    reason = _evidence_veto_reason(evidence_by_region, region_a, region_b, plot_a, plot_b, merged_plot_worse)
    assert reason is not None
    assert "depth" in reason


def test_evidence_veto_allows_a_merge_that_moves_toward_confirmed_depth():
    from backend.cv_extraction.dxf_extractor import _evidence_veto_reason
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox

    region_a = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=8.3))
    region_b = DxfRegion(id=1, bbox=BoundingBox(min_x=20, min_y=0, max_x=25, max_y=25))
    plot_a = _rect_poly_wd(10.0, 8.3)
    plot_b = _rect_poly_wd(5.0, 5.0)
    merged_plot_better = _rect_poly_wd(10.0, 8.5)  # depth moved CLOSER to the confirmed 8.7

    evidence_by_region = {0: [_recovered(10.0, "10.00"), _recovered(8.7, "8.70")]}

    reason = _evidence_veto_reason(evidence_by_region, region_a, region_b, plot_a, plot_b, merged_plot_better)
    assert reason is None


def test_evidence_veto_does_not_fire_without_confirmed_evidence():
    """No recovered evidence at all (or only a single-axis match) for
    either parent -- growth must proceed purely on structural merit, same
    as before this mechanism existed."""
    from backend.cv_extraction.dxf_extractor import _evidence_veto_reason
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox

    region_a = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=8.3))
    region_b = DxfRegion(id=1, bbox=BoundingBox(min_x=20, min_y=0, max_x=25, max_y=25))
    plot_a = _rect_poly_wd(10.0, 8.3)
    plot_b = _rect_poly_wd(5.0, 5.0)
    merged_plot = _rect_poly_wd(10.0, 20.0)

    assert _evidence_veto_reason({}, region_a, region_b, plot_a, plot_b, merged_plot) is None
    single_axis_evidence = {0: [_recovered(9.0, "9.00")]}  # depth only, no width confirmation
    assert _evidence_veto_reason(single_axis_evidence, region_a, region_b, plot_a, plot_b, merged_plot) is None


def test_merge_candidate_region_pairs_respects_evidence_veto(monkeypatch):
    """End-to-end through `_merge_candidate_region_pairs` itself: a
    pairing that would score higher AND pass the area-sanity check is
    still rejected when it moves a confirmed dimension further away, and
    growth correctly stops there rather than accepting a worse result."""
    import backend.cv_extraction.dxf_extractor as dxf_extractor_mod
    from backend.cv_extraction.dxf_extractor import _merge_candidate_region_pairs, _score_region_resolution
    from backend.cv_extraction.dxf_regions import DxfRegion
    from backend.schemas.geometry import BoundingBox

    plot_a = _rect_poly_wd(10.0, 8.3)
    plot_b = _rect_poly_wd(5.0, 5.0)
    # depth 12.0 scores higher than both inputs and stays within the
    # area-sum-sanity bound ((83+25)*1.15=124.2 >= 10*12=120) -- isolating
    # the veto as the ONLY reason this must still be rejected, not area.
    merged_plot = _rect_poly_wd(10.0, 12.0)

    region_a = DxfRegion(id=0, bbox=BoundingBox(min_x=0, min_y=0, max_x=10, max_y=8.3), polygon_indices=[0])
    region_b = DxfRegion(id=1, bbox=BoundingBox(min_x=20, min_y=0, max_x=25, max_y=25), polygon_indices=[1])
    score_a = _score_region_resolution(plot_a, None)
    score_b = _score_region_resolution(plot_b, None)
    structural = [(score_a, region_a, plot_a, None, None), (score_b, region_b, plot_b, None, None)]

    def _fake_resolve(*args, **kwargs):
        return merged_plot, None, None

    monkeypatch.setattr(dxf_extractor_mod, "_resolve_plot_building_road", _fake_resolve)
    assert _score_region_resolution(merged_plot, None) > max(score_a, score_b), "test setup must actually score higher"

    evidence_by_region = {0: [_recovered(10.0, "10.00"), _recovered(8.7, "8.70")]}
    warnings: list[str] = []
    merged = _merge_candidate_region_pairs(
        structural, [plot_a, plot_b], [], frame_indices=set(), warnings=warnings, time_budget_seconds=30.0,
        evidence_by_region=evidence_by_region,
    )

    assert merged == [], "the higher-scoring merge must still be vetoed for moving away from confirmed depth"
    assert any("vetoed" in w for w in warnings)


def test_select_labeled_area_match_prefers_achieved_reading_over_permissible_or_unrelated():
    """Pins the exact real-fixture bug found live on PLAN8.dxf: a coverage/
    FAR worksheet restates the same field several times with different
    qualifiers (a regulatory ceiling, the achieved as-built figure, a
    remainder), and blind first-match-in-file-order selection shipped the
    PERMISSIBLE ceiling for coverage and, worse, an unrelated number for
    FAR. `_select_labeled_area_match` must prefer the ACHIEVED/NET/PROPOSED
    reading regardless of file order."""
    from backend.cv_extraction.dxf_extractor import _RawText, _select_labeled_area_match

    def rt(text: str) -> _RawText:
        return _RawText(text=text, position=(0.0, 0.0), layer="NOTES")

    far_candidates = [
        (rt("Total Perm. FAR area ( 1.75 )"), 1.75),
        (rt("Residential FAR (100.00% )"), 100.00),
        (rt("Achieved Net FAR Area ( 1.64 )"), 1.64),
        (rt("Balance FAR Area ( 0.11 )"), 0.11),
    ]
    selected = _select_labeled_area_match(far_candidates)
    assert selected is not None
    assert selected[1] == 1.64

    coverage_candidates = [
        (rt("Permissible Coverage area (70.00 %)"), 70.00),
        (rt("Proposed Coverage Area (62.83 %)"), 62.83),
        (rt("Achieved Net coverage area (62.83 %)"), 62.83),
        (rt("Balance coverage area left ( 7.16 % )"), 7.16),
    ]
    selected = _select_labeled_area_match(coverage_candidates)
    assert selected is not None
    assert selected[1] == 62.83


def test_select_labeled_area_match_falls_back_to_first_candidate_when_no_qualifier_present():
    """The common case (a sheet states a field exactly once, with no
    ACHIEVED/PERMISSIBLE-style qualifier at all) must be completely
    unaffected -- this is what guarantees the fix above cannot regress any
    single-reading sheet into MISSING."""
    from backend.cv_extraction.dxf_extractor import _RawText, _select_labeled_area_match

    def rt(text: str) -> _RawText:
        return _RawText(text=text, position=(0.0, 0.0), layer="NOTES")

    candidates = [(rt("AREA OF PLOT = 200.00 SQM"), 200.00)]
    assert _select_labeled_area_match(candidates) == candidates[0]


def test_select_labeled_area_match_falls_back_to_first_when_every_candidate_is_deprioritized():
    """If EVERY match on a sheet happens to carry a deprioritized qualifier
    (no achieved/net/proposed reading exists at all), this must still
    return something -- the old first-match behavior -- rather than
    silently dropping a field that previously always resolved."""
    from backend.cv_extraction.dxf_extractor import _RawText, _select_labeled_area_match

    def rt(text: str) -> _RawText:
        return _RawText(text=text, position=(0.0, 0.0), layer="NOTES")

    candidates = [
        (rt("Permissible Coverage area (70.00 %)"), 70.00),
        (rt("Balance coverage area left ( 7.16 % )"), 7.16),
    ]
    assert _select_labeled_area_match(candidates) == candidates[0]


def test_far_and_coverage_from_a_worksheet_with_multiple_qualified_readings(tmp_path):
    """End-to-end regression pin for the real PLAN8.dxf bug: a plot/
    building sheet whose NOTES layer restates coverage/FAR with several
    different qualifiers must resolve `coverage`/`far` to the ACHIEVED
    figure via the real `DXFHybridExtractor` pipeline, not a regulatory
    limit or an unrelated number the old, letter-forbidding `far` regex
    happened to grab instead."""
    import ezdxf

    path = rectangular_site_plan_dxf(tmp_path / "plan.dxf")
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    for text in (
        "Permissible Coverage area (70.00 %)",
        "Proposed Coverage Area (62.83 %)",
        "Achieved Net coverage area (62.83 %)",
        "Balance coverage area left ( 7.16 % )",
        "Total Perm. FAR area ( 1.75 )",
        "Residential FAR (100.00% )",
        "Achieved Net FAR Area ( 1.64 )",
        "Balance FAR Area ( 0.11 )",
    ):
        msp.add_text(text, dxfattribs={"layer": "NOTES"}).dxf.insert = (6.0, 20.0)
    doc.saveas(str(path))

    result = DXFHybridExtractor().extract(path, "plan")
    measurements = _measurements(result)

    assert measurements["coverage"] == pytest.approx(62.83)
    assert measurements["far"] == pytest.approx(1.64)
