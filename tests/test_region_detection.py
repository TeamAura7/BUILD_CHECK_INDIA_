"""
Tests for `backend.cv_extraction.region_detection` and the Vision focus-pass
region cross-check it feeds (`BaseArchitecturalPlanExtractor._select_focus_bbox`
in `backend.vision_extraction.base`).

The concrete bug this guards against was reproduced live against
`data/test_plans/PLAN5.pdf`: Vision's own first-pass `SITE_PLAN` region bbox
was `[50, 50, 450, 350]` on the model's normalized 0-1000 grid -- the
top-left of the page -- while the sheet's actual site plan is at the
bottom-right (independently confirmed by
`backend.cv_extraction.site_plan.extract_independent_cv`'s own
`site_plan_bbox_pts` and corroborated by the sheet's printed "SITE AREA"
statement). The old, ungrounded focus-crop pass blindly cropped from
Vision's bbox and fed the ground-floor plan into a prompt asking for site-
plan setbacks; its own returned region evidence literally read "GROUND
FLOOR PLAN". These tests pin the fix: a caption-anchored bbox that strongly
disagrees with Vision's own guess must win.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cv_extraction import region_detection
from backend.schemas.geometry import BoundingBox
from backend.schemas.regions import DetectedRegion
from backend.schemas.vision import VisionRegion
from backend.vision_extraction.base import BaseArchitecturalPlanExtractor

PLANS_DIR = Path(__file__).parent.parent / "data" / "test_plans"


# --- region_detection: pure geometry helpers --------------------------------


def test_iou_of_non_overlapping_boxes_is_zero():
    a = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    b = BoundingBox(min_x=100, min_y=100, max_x=110, max_y=110)
    assert region_detection.iou(a, b) == 0.0


def test_iou_of_identical_boxes_is_one():
    a = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    assert region_detection.iou(a, a) == pytest.approx(1.0)


def test_iou_of_partially_overlapping_boxes():
    a = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)  # area 100
    b = BoundingBox(min_x=5, min_y=0, max_x=15, max_y=10)  # area 100, overlap 5x10=50
    # union = 100 + 100 - 50 = 150; iou = 50/150
    assert region_detection.iou(a, b) == pytest.approx(50 / 150)


def test_best_match_picks_highest_confidence_of_matching_type():
    regions = [
        DetectedRegion(type="SITE_PLAN", bbox_pts=BoundingBox(min_x=0, min_y=0, max_x=1, max_y=1), confidence=0.5, source="vision"),
        DetectedRegion(type="SITE_PLAN", bbox_pts=BoundingBox(min_x=2, min_y=2, max_x=3, max_y=3), confidence=0.9, source="caption_anchor"),
        DetectedRegion(type="ELEVATION", bbox_pts=BoundingBox(min_x=4, min_y=4, max_x=5, max_y=5), confidence=1.0, source="caption_anchor"),
    ]
    best = region_detection.best_match(regions, "SITE_PLAN")
    assert best is not None
    assert best.confidence == 0.9
    assert region_detection.best_match(regions, "AREA_STATEMENT") is None


# --- region_detection: real PDF captions -------------------------------------


def test_detects_site_plan_and_area_statement_captions_on_plan2():
    pdf_path = PLANS_DIR / "PLAN2.pdf"
    if not pdf_path.exists():
        pytest.skip("PLAN2.pdf fixture not present")
    regions = region_detection.detect_regions(pdf_path, 0)
    types = {r.type for r in regions}
    assert "SITE_PLAN" in types
    assert "AREA_STATEMENT" in types
    for r in regions:
        assert r.source == "caption_anchor"
        assert r.confidence > 0


def test_plan5_site_plan_caption_anchor_locates_the_real_site_plan_not_the_top_left():
    """
    The exact case that exposed the bug: PLAN5 is rotated and its "SITE
    PLAN SCALE 1:200" caption is a tall, narrow, vertically-set OCR item.
    The deterministic locator's window must contain the independently-
    verified true site-plan bbox
    (`site_plan.extract_independent_cv`'s `site_plan_bbox_pts`,
    corroborated by the sheet's own area statement in
    tests/test_real_plan_regression.py), and must NOT be the top-left
    quadrant of the page Vision's own first pass mistakenly claimed.
    """
    pdf_path = PLANS_DIR / "PLAN5.pdf"
    if not pdf_path.exists():
        pytest.skip("PLAN5.pdf fixture not present")
    from backend.cv_extraction.site_plan import extract_independent_cv

    cv_result = extract_independent_cv(pdf_path, "PLAN5")
    assert cv_result.site_plan_bbox_pts is not None, "independent CV must resolve a site-plan bbox for this test to be meaningful"
    true_bbox = BoundingBox(
        min_x=cv_result.site_plan_bbox_pts[0], min_y=cv_result.site_plan_bbox_pts[1],
        max_x=cv_result.site_plan_bbox_pts[2], max_y=cv_result.site_plan_bbox_pts[3],
    )

    regions = region_detection.detect_regions(pdf_path, 0)
    detected = region_detection.best_match(regions, "SITE_PLAN")
    assert detected is not None, "caption-anchored SITE_PLAN window not found on PLAN5"

    # The caption-anchored window is a generous SEARCH window (by design --
    # see region_detection's module docstring), not a tight bbox, so it is
    # expected to be much larger in area than the true site-plan rectangle;
    # raw IoU would be misleadingly small even for perfect containment.
    # What matters is that it actually contains the true rectangle.
    window = detected.bbox_pts
    ix0, iy0 = max(window.min_x, true_bbox.min_x), max(window.min_y, true_bbox.min_y)
    ix1, iy1 = min(window.max_x, true_bbox.max_x), min(window.max_y, true_bbox.max_y)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    true_bbox_area = true_bbox.width * true_bbox.height
    contained_fraction = intersection / true_bbox_area if true_bbox_area else 0.0
    assert contained_fraction > 0.95, (
        f"caption-anchored window {window} does not contain the true site-plan "
        f"bbox {true_bbox} (only {contained_fraction:.1%} of it falls inside the window)"
    )

    # And it must NOT be the wrong-region bbox Vision's own first pass
    # actually returned live for this exact sheet.
    vision_wrong_bbox = BoundingBox(min_x=59.6, min_y=84.2, max_x=536.1, max_y=589.4)  # ~[50,50,450,350]/1000 in page-points
    assert region_detection.iou(detected.bbox_pts, vision_wrong_bbox) < 0.05


# --- base.py: the cross-check that decides which bbox a focus pass crops ----


def _vision_region(region_type: str, bbox_0_1000: list[float]) -> VisionRegion:
    return VisionRegion(id="region_1", type=region_type, bbox=bbox_0_1000, confidence=0.95, label=region_type, evidence=region_type)


def test_select_focus_bbox_prefers_caption_anchor_when_vision_bbox_disagrees():
    """Pins the PLAN5 bug: a wildly-disagreeing Vision bbox must lose to the caption anchor."""
    page_w, page_h = 1191.24, 1684.08
    # The actual bbox Vision returned live for PLAN5's SITE_PLAN region.
    vision_regions = [_vision_region("SITE_PLAN", [50, 50, 450, 350])]
    # The actual bbox region_detection finds for PLAN5's SITE_PLAN caption.
    true_window = BoundingBox(min_x=407.3, min_y=990.9, max_x=1121.9, max_y=1597.1)
    detected = [DetectedRegion(type="SITE_PLAN", bbox_pts=true_window, confidence=0.9, source="caption_anchor")]

    selection = BaseArchitecturalPlanExtractor._select_focus_bbox(
        vision_regions, detected, "SITE_PLAN", page_w, page_h
    )
    assert selection is not None
    chosen_bbox, note = selection
    assert chosen_bbox == true_window
    assert "disagreed" in note.lower()


def test_select_focus_bbox_uses_vision_bbox_when_they_agree():
    page_w, page_h = 1000.0, 1000.0
    # Vision's normalized bbox [400,400,600,600] maps to page-points [400,400,600,600] on a 1000x1000 page.
    vision_regions = [_vision_region("SITE_PLAN", [400, 400, 600, 600])]
    detected = [DetectedRegion(
        type="SITE_PLAN", bbox_pts=BoundingBox(min_x=390, min_y=390, max_x=610, max_y=610),
        confidence=0.9, source="caption_anchor",
    )]
    selection = BaseArchitecturalPlanExtractor._select_focus_bbox(
        vision_regions, detected, "SITE_PLAN", page_w, page_h
    )
    assert selection is not None
    chosen_bbox, note = selection
    assert "agrees" in note.lower()
    # Vision's own bbox is used when the two sources agree.
    assert chosen_bbox.min_x == pytest.approx(400.0)


def test_select_focus_bbox_falls_back_to_vision_when_no_caption_anchor_found():
    page_w, page_h = 1000.0, 1000.0
    vision_regions = [_vision_region("SITE_PLAN", [100, 100, 300, 300])]
    selection = BaseArchitecturalPlanExtractor._select_focus_bbox(
        vision_regions, [], "SITE_PLAN", page_w, page_h
    )
    assert selection is not None
    chosen_bbox, note = selection
    assert "caption unreadable" in note.lower() or "no caption" in note.lower()
    assert chosen_bbox.min_x == pytest.approx(100.0)


def test_select_focus_bbox_returns_none_when_neither_source_has_the_region():
    selection = BaseArchitecturalPlanExtractor._select_focus_bbox([], [], "SITE_PLAN", 1000.0, 1000.0)
    assert selection is None
