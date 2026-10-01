"""Tests for `backend.spatial_reasoning.road_access` and `.front_side`."""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.spatial_reasoning.front_side import resolve_front_side
from backend.spatial_reasoning.road_access import best_road_candidate, collect_access_evidence
from tests.fixtures.geometry_builders import BoundingBox, make_extraction, rect_polygon, road_candidate, text


def test_road_never_assumed_top_of_page():
    """
    NEVER assume PDF top = front. With a ROAD candidate positioned at the
    BOTTOM of the plot (south edge), front must resolve to the bottom
    edge, not the top edge just because it's first in page order.
    """
    plot = rect_polygon(0, 0, 400, 480)
    road_bbox = BoundingBox(min_x=0, min_y=500, max_x=400, max_y=520)  # below (south of) the plot
    fsr = resolve_front_side(plot, road_bbox=road_bbox, access_evidence=[])
    assert fsr.evidence_level == "ROAD candidate"
    # front edge should be the bottom edge (y ~ 480), not the top (y ~ 0)
    front_mid_y = sum(p.y for p in [fsr.front_edges[0].start, fsr.front_edges[0].end]) / 2
    assert front_mid_y > 240  # closer to the bottom (480) than the top (0)


def test_front_label_takes_priority_over_road():
    from backend.spatial_reasoning.road_access import AccessEvidence
    from tests.fixtures.geometry_builders import text as text_ev

    plot = rect_polygon(0, 0, 400, 480)
    # Road is at the bottom, but an explicit FRONT label sits at the top —
    # FRONT label must win per phase3.md's priority order.
    road_bbox = BoundingBox(min_x=0, min_y=500, max_x=400, max_y=520)
    front_label = AccessEvidence(kind="front", text_evidence=text_ev("FRONT", bbox=BoundingBox(min_x=180, min_y=-20, max_x=220, max_y=-5)))
    fsr = resolve_front_side(plot, road_bbox=road_bbox, access_evidence=[front_label])
    assert fsr.evidence_level == "FRONT label"
    assert fsr.confidence == ConfidenceLevel.HIGH
    front_mid_y = sum(p.y for p in [fsr.front_edges[0].start, fsr.front_edges[0].end]) / 2
    assert front_mid_y < 240  # near the top, where the FRONT label is


def test_no_evidence_falls_back_at_low_confidence_not_silently():
    plot = rect_polygon(0, 0, 400, 480)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    assert fsr.confidence == ConfidenceLevel.LOW
    assert fsr.evidence_level is None
    assert "fallback" in fsr.reasoning.lower() or "not an assumption" in fsr.reasoning.lower()


def test_rear_is_farthest_edge_from_front():
    plot = rect_polygon(0, 0, 400, 480)
    road_bbox = BoundingBox(min_x=0, min_y=500, max_x=400, max_y=520)
    fsr = resolve_front_side(plot, road_bbox=road_bbox, access_evidence=[])
    front_mid_y = sum(p.y for p in [fsr.front_edges[0].start, fsr.front_edges[0].end]) / 2
    rear_mid_y = sum(p.y for p in [fsr.rear_edges[0].start, fsr.rear_edges[0].end]) / 2
    assert rear_mid_y < front_mid_y  # rear (top, y~0) is farthest from front (bottom, y~480)


def test_left_right_are_the_two_lateral_edges():
    plot = rect_polygon(0, 0, 400, 480)
    road_bbox = BoundingBox(min_x=0, min_y=500, max_x=400, max_y=520)
    fsr = resolve_front_side(plot, road_bbox=road_bbox, access_evidence=[])
    assert len(fsr.left_edges) == 1
    assert len(fsr.right_edges) == 1
    assert fsr.left_edges[0] is not fsr.right_edges[0]


def test_access_evidence_collects_gate_and_entry_keywords():
    text_evs = [
        text("MAIN ENTRY GATE", bbox=BoundingBox(min_x=0, min_y=0, max_x=50, max_y=10)),
        text("BEDROOM", bbox=BoundingBox(min_x=100, min_y=100, max_x=150, max_y=110)),
    ]
    evidence = collect_access_evidence(text_evs, page=0)
    kinds = {e.kind for e in evidence}
    assert "main_entry" in kinds or "gate" in kinds
    assert not any(e.kind == "gate" and "BEDROOM" in e.text_evidence.raw_text for e in evidence)


def test_best_road_candidate_prefers_labeled_and_larger():
    small_unlabeled = road_candidate(BoundingBox(min_x=0, min_y=0, max_x=10, max_y=5), cid="r1")
    large_labeled = road_candidate(BoundingBox(min_x=0, min_y=0, max_x=400, max_y=20), cid="r2", label="12M ROAD")
    er = make_extraction(road_candidates=[small_unlabeled, large_labeled])
    best = best_road_candidate(er, page=0)
    assert best.id == "r2"


def test_no_road_candidates_returns_none():
    er = make_extraction()
    assert best_road_candidate(er, page=0) is None
