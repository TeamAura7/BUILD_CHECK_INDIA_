"""Tests for `backend.cv_extraction.skewed_plot` (plots with slanted boundary edges)."""

from __future__ import annotations

import math

import pytest

from backend.cv_extraction import site_plan, skewed_plot
from backend.cv_extraction.raw_types import RawLine, RawTextItem, SourceKind
from backend.schemas.geometry import BoundingBox, Line, Point

SCALE = 20.0   # points per metre in the synthetic drawing
REGION = BoundingBox(min_x=0, min_y=0, max_x=600, max_y=600)

# A skewed plot: top leans 1.1 degrees, left edge 7.6 degrees, right and bottom are straight.
TL, TR, BR, BL = (100.0, 100.0), (300.0, 104.0), (300.0, 400.0), (140.0, 400.0)
BUILDING = BoundingBox(min_x=160, min_y=160, max_x=260, max_y=340)   # 5.0 x 9.0 m


def _line(a, b, stroke=0.85):
    return RawLine(
        line=Line(start=Point(x=a[0], y=a[1]), end=Point(x=b[0], y=b[1])),
        page=0, source=SourceKind.VECTOR_PDF, stroke_width=stroke,
    )


def _num(value, x, y):
    item = RawTextItem(
        text=f"{value:.2f}M", bounding_box=BoundingBox(min_x=x - 12, min_y=y - 5, max_x=x + 12, max_y=y + 5),
        page=0, source=SourceKind.PDF_TEXT,
    )
    return (value, item, "m")


def _plot_lines(dashed_left=False):
    lines = [_line(TL, TR), _line(TR, BR), _line(BL, BR)]
    if dashed_left:
        n = 24
        for i in range(n):
            t0, t1 = i / n, (i + 0.6) / n
            lines.append(_line((TL[0] + (BL[0] - TL[0]) * t0, TL[1] + (BL[1] - TL[1]) * t0),
                               (TL[0] + (BL[0] - TL[0]) * t1, TL[1] + (BL[1] - TL[1]) * t1)))
    else:
        lines.append(_line(TL, BL))
    b = BUILDING
    lines += [_line((b.min_x, b.min_y), (b.max_x, b.min_y), 0.7), _line((b.max_x, b.min_y), (b.max_x, b.max_y), 0.7),
              _line((b.min_x, b.max_y), (b.max_x, b.max_y), 0.7), _line((b.min_x, b.min_y), (b.min_x, b.max_y), 0.7)]
    return lines


def _edge_labels():
    return [_num(10.0, 200, 84), _num(8.0, 220, 420), _num(14.8, 322, 250), _num(15.1, 88, 250)]


def _building_labels():
    return [_num(5.0, 210, 150), _num(9.0, 150, 250)]


def _analyse(lines, nums, **kw):
    return skewed_plot.analyse(
        lines, REGION, nums=nums, rect_candidates=kw.pop("rects", [BUILDING]), **kw,
    )


def test_a_sheet_drawn_with_exactly_axis_aligned_lines_has_no_slanted_evidence():
    rect = [_line((100, 100), (300, 100)), _line((300, 100), (300, 400)), _line((100, 400), (300, 400)), _line((100, 100), (100, 400))]
    assert skewed_plot.slanted_boundary_evidence(rect, REGION) == 0
    state, plot, _ = _analyse(rect, _edge_labels())
    assert state == "not_skewed" and plot is None


def test_thin_or_short_slanted_lines_are_not_boundary_evidence():
    thin = [_line((100, 100), (140, 400), stroke=0.28)]
    short = [_line((100, 100), (108, 160))]
    assert skewed_plot.slanted_boundary_evidence(thin, REGION) == 0
    assert skewed_plot.slanted_boundary_evidence(short, REGION) == 0


def test_a_skewed_quadrilateral_is_recovered_and_validated_by_its_labels():
    state, plot, _ = _analyse(_plot_lines(), _edge_labels() + _building_labels())
    assert state == "resolved"
    assert plot.scale_pts_per_m == pytest.approx(SCALE, rel=0.02)
    assert plot.label_agreement >= 3
    assert plot.edge_length_m["top"] == pytest.approx(10.0, abs=0.15)
    assert plot.edge_length_m["bottom"] == pytest.approx(8.0, abs=0.15)
    assert plot.edge_length_m["left"] == pytest.approx(15.13, abs=0.15)
    assert plot.tilt_deg == pytest.approx(math.degrees(math.atan(40 / 300)), abs=0.5)


def test_setbacks_are_minimum_perpendicular_distances_to_the_slanted_edges():
    """The left gap is 2.6 m at the top and 1.4 m at the bottom: the compliance
    value is the smaller, and it is measured perpendicular to the leaning edge."""
    _, plot, _ = _analyse(_plot_lines(), _edge_labels() + _building_labels())
    assert plot.building == BUILDING
    assert plot.setback_geometry_m["left"] == pytest.approx(27.75 / SCALE, abs=0.05)
    assert plot.setback_geometry_m["top"] == pytest.approx(56.8 / SCALE, abs=0.06)
    assert plot.setback_geometry_m["right"] == pytest.approx(2.0, abs=0.03)
    assert plot.setback_geometry_m["bottom"] == pytest.approx(3.0, abs=0.03)
    # 52 pt / 20 = 2.6 m at the wide end: the minimum must not be that.
    assert plot.setback_geometry_m["left"] < 2.0


def test_a_dashed_slanted_edge_is_merged_into_one_edge():
    state, plot, _ = _analyse(_plot_lines(dashed_left=True), _edge_labels() + _building_labels())
    assert state == "resolved"
    assert plot.edge_length_m["left"] == pytest.approx(15.13, abs=0.3)


def test_thin_dimension_lines_beside_the_boundary_do_not_replace_it():
    dims = [_line((TL[0] - 30, TL[1]), (BL[0] - 30, BL[1]), stroke=0.28),
            _line((TL[0], TL[1] - 20), (TR[0], TR[1] - 20), stroke=0.28)]
    state, plot, _ = _analyse(_plot_lines() + dims, _edge_labels() + _building_labels())
    assert state == "resolved"
    assert plot.edge_length_m["top"] == pytest.approx(10.0, abs=0.15)


def test_labels_that_disagree_on_scale_leave_the_plot_unresolved():
    wrong = [_num(10.0, 200, 84), _num(9.1, 220, 420), _num(31.0, 322, 250), _num(6.2, 88, 250)]
    state, plot, notes = _analyse(_plot_lines(), wrong)
    assert state == "unresolved" and plot is None
    assert any("Nothing is asserted" in n for n in notes)


def test_small_setback_callouts_are_not_mistaken_for_plot_edge_labels():
    """A ~1 m 'quadrilateral' once validated at 176 pt/m on setback callouts."""
    callouts = [_num(1.0, 200, 84), _num(1.1, 220, 420), _num(1.5, 322, 250), _num(1.65, 88, 250)]
    state, plot, _ = _analyse(_plot_lines(), callouts)
    assert state == "unresolved"


def test_two_agreeing_edges_alone_are_not_enough_without_the_stated_area():
    two = [_num(10.0, 200, 84), _num(15.1, 88, 250)]
    assert _analyse(_plot_lines(), two)[0] == "unresolved"
    quad_area_m2 = skewed_plot._polygon_area([TL, TR, BR, BL]) / SCALE ** 2
    assert _analyse(_plot_lines(), two, plot_area_target_m2=quad_area_m2)[0] == "resolved"


def test_a_printed_scale_close_to_the_label_scale_wins():
    state, plot, _ = _analyse(_plot_lines(), _edge_labels(), printed_scales=[20.4])
    assert state == "resolved"
    assert plot.scale_pts_per_m == 20.4


def test_building_is_found_by_stated_footprint_area_when_no_dimension_labels_exist():
    state, plot, _ = _analyse(_plot_lines(), _edge_labels(), footprint_area_target_m2=45.0)
    assert state == "resolved" and plot.building == BUILDING
    state, plot, _ = _analyse(_plot_lines(), _edge_labels(), footprint_area_target_m2=80.0)
    assert plot.building is None


def test_a_building_outside_the_plot_is_never_chosen():
    outside = BoundingBox(min_x=400, min_y=160, max_x=500, max_y=340)
    _, plot, _ = _analyse(_plot_lines(), _edge_labels() + _building_labels(), rects=[outside])
    assert plot.building is None


def test_unresolved_slanted_plot_caps_the_rectangle_models_confidence_at_low(monkeypatch):
    """One heavy slanted line and no validated quadrilateral: the rectangle model
    may still run, but its assumption is known to be violated."""
    from backend.schemas.independent_measurements import IndependentMeasurement

    anchor = RawTextItem(text="SITE PLAN", bounding_box=BoundingBox(min_x=200, min_y=450, max_x=260, max_y=470),
                         page=0, source=SourceKind.PDF_TEXT)
    measurement = IndependentMeasurement(field="plot.width", value_m=10.0, source="NATIVE_TEXT", confidence=0.99,
                                         evidence=[], note="", page=1)
    monkeypatch.setattr(site_plan, "_extract_site_plan_rectangles",
                        lambda *a, **k: ([measurement], BoundingBox(min_x=0, min_y=0, max_x=1, max_y=1), 20.0, 0.99, []))
    monkeypatch.setattr(site_plan, "_extract_from_skewed_plot", lambda *a, **k: None)
    monkeypatch.setattr(site_plan.skewed_plot, "slanted_boundary_evidence", lambda *a, **k: 1)
    out, _bbox, _scale, _conf, _notes = site_plan.extract_site_plan_measurements(
        [anchor], [], page_number=1, page_width=600, page_height=600,
    )
    assert out[0].confidence <= 0.4
    assert "confidence capped" in out[0].note


def test_without_slanted_evidence_the_rectangle_model_is_called_unchanged(monkeypatch):
    from backend.schemas.independent_measurements import IndependentMeasurement

    anchor = RawTextItem(text="SITE PLAN", bounding_box=BoundingBox(min_x=200, min_y=450, max_x=260, max_y=470),
                         page=0, source=SourceKind.PDF_TEXT)
    measurement = IndependentMeasurement(field="plot.width", value_m=10.0, source="NATIVE_TEXT", confidence=0.99,
                                         evidence=[], note="", page=1)
    monkeypatch.setattr(site_plan, "_extract_site_plan_rectangles",
                        lambda *a, **k: ([measurement], None, 20.0, 0.99, []))
    monkeypatch.setattr(site_plan.skewed_plot, "slanted_boundary_evidence", lambda *a, **k: 0)
    out, *_ = site_plan.extract_site_plan_measurements([anchor], [], page_number=1, page_width=600, page_height=600)
    assert out[0].confidence == 0.99
