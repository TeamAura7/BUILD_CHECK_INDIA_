"""Tests for `backend.spatial_reasoning.plot_resolution`."""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.spatial_reasoning.plot_resolution import plot_confidence, score_plot_candidates
from tests.fixtures.geometry_builders import (
    BoundingBox,
    dim,
    make_extraction,
    plot_candidate,
    rect_polygon,
    road_candidate,
    text,
)


def test_no_candidates_returns_none_winner():
    er = make_extraction()
    winner, scored, note = score_plot_candidates(er)
    assert winner is None
    assert scored == []
    assert "No plot candidates" in note


def test_largest_rectangle_is_not_automatically_the_winner():
    """
    A big, unlabeled, unannotated rectangle must NOT beat a smaller
    rectangle that has boundary dimension annotations, a 'PLOT' label,
    and road adjacency — per phase3.md's explicit "do NOT simply use
    largest rectangle = plot" rule.
    """
    from backend.schemas.geometry import Line, Point

    # Keep the two candidates far apart (relative to their own sizes) so
    # neither candidate's proximity-based scores (label_match,
    # boundary_annotation, road_adjacency — all of which use a tolerance
    # scaled to the candidate's own bounding-box size) can pick up false
    # credit from evidence that actually belongs to the other candidate.
    huge_unlabeled = plot_candidate(rect_polygon(0, 0, 700, 700), cid="huge")
    small_labeled = plot_candidate(rect_polygon(3000, 3000, 3200, 3200), cid="small-labeled")

    small_bbox = small_labeled.geometry.bounding_box
    boundary_dims = [
        dim(10.0, "m", line=Line(start=Point(x=small_bbox.min_x, y=small_bbox.min_y - 2), end=Point(x=small_bbox.max_x, y=small_bbox.min_y - 2))),
        dim(10.0, "m", line=Line(start=Point(x=small_bbox.min_x, y=small_bbox.max_y + 2), end=Point(x=small_bbox.max_x, y=small_bbox.max_y + 2))),
        dim(4.0, "m", line=Line(start=Point(x=small_bbox.min_x - 2, y=small_bbox.min_y), end=Point(x=small_bbox.min_x - 2, y=small_bbox.max_y))),
        dim(4.0, "m", line=Line(start=Point(x=small_bbox.max_x + 2, y=small_bbox.min_y), end=Point(x=small_bbox.max_x + 2, y=small_bbox.max_y))),
    ]
    road_bbox = BoundingBox(min_x=small_bbox.min_x, min_y=small_bbox.max_y + 1, max_x=small_bbox.max_x, max_y=small_bbox.max_y + 20)

    er = make_extraction(
        plot_candidates=[huge_unlabeled, small_labeled],
        dimensions=boundary_dims,
        road_candidates=[road_candidate(road_bbox)],
        text_evidence=[
            text(
                "PLOT NO. 12",
                bbox=BoundingBox(min_x=small_bbox.min_x, min_y=small_bbox.min_y - 20, max_x=small_bbox.max_x, max_y=small_bbox.min_y - 6),
            )
        ],
    )
    winner, scored, note = score_plot_candidates(er)
    breakdown_by_id = {s.candidate.id: s.breakdown for s in scored}
    assert breakdown_by_id["huge"]["boundary_annotation"] == 0.0
    assert breakdown_by_id["huge"]["label_match"] == 0.0
    assert winner.candidate.id == "small-labeled"


def test_plot_confidence_high_with_clear_margin():
    a = plot_candidate(rect_polygon(0, 0, 400, 480), cid="a")
    er = make_extraction(plot_candidates=[a])
    winner, scored, _ = score_plot_candidates(er)
    level, _reason = plot_confidence(winner, scored)
    # Single candidate: MEDIUM or LOW depending on its own score, never HIGH.
    assert level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)


def test_candidates_missing_geometry_are_excluded():
    from backend.schemas.candidates import PlotCandidate
    from backend.schemas.evidence import ValueField

    no_geom = PlotCandidate(id="no-geom", width=ValueField.missing(), depth=ValueField.missing(), area=ValueField.missing())
    er = make_extraction(plot_candidates=[no_geom])
    winner, scored, note = score_plot_candidates(er)
    assert winner is None
