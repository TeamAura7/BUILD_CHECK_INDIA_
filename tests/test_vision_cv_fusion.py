"""
Regression tests for two Phase 3.1 bugs found while integrating vision
and CV extraction against a real plan:

1. `vision_bbox_to_page_points` assumed raw pixels at `vision_render_dpi`,
   when the model actually returns a normalized 0-1000 grid. On a
   large-format sheet this shrank every vision bbox into a sliver near
   the page origin, so it never overlapped real CV geometry.
2. Vision output could only ever relabel a `Dimension` the deterministic
   CV/OCR pipeline had already produced -- a value the vision model read
   correctly but CV/OCR missed entirely had no path into the final plan.
"""

from __future__ import annotations

from backend.schemas.geometry import Dimension
from backend.schemas.vision import VisionDimension, VisionPageResult, VisionRegion
from backend.spatial_reasoning.dimension_classification import (
    DimensionSemanticType,
    vision_only_dimensions,
)
from backend.vision_extraction.spatial import region_score, vision_bbox_to_page_points


# --- Bug #1: coordinate-space auto-detection ------------------------------


def test_normalized_grid_bbox_is_detected_and_scaled_to_real_page_size():
    # A large architectural sheet: 3024x2160pt page, rendered at 200 DPI
    # -> an 8400x6000px image. A bbox reported near the middle of a
    # normalized 0-1000 grid should land near the middle of the real page,
    # not near the origin.
    box = vision_bbox_to_page_points([400.0, 400.0, 600.0, 600.0], page_width_pts=3024.0, page_height_pts=2160.0)
    assert box is not None
    # Old (buggy) behavior would have produced a box around x=1.44-2.16pt
    # (400 * 72/200), i.e. essentially at the page origin. The fixed
    # behavior should land near the page's actual centre.
    assert 1000.0 < box.center.x < 2000.0
    assert 700.0 < box.center.y < 1500.0


def test_raw_pixel_bbox_still_handled_when_values_exceed_normalized_range():
    # If a bbox clearly exceeds the 0-1000 grid, it must be real pixels on
    # the rendered image, not the normalized grid -- must not be
    # (mis)treated as normalized.
    box = vision_bbox_to_page_points([4000.0, 3000.0, 4200.0, 3200.0], page_width_pts=3024.0, page_height_pts=2160.0)
    assert box is not None
    assert 0.0 <= box.min_x <= 3024.0
    assert 0.0 <= box.min_y <= 2160.0


def test_unknown_page_size_falls_back_without_crashing():
    box = vision_bbox_to_page_points([10.0, 10.0, 20.0, 20.0])
    assert box is not None


def test_region_score_matches_after_coordinate_fix():
    # A SITE_PLAN region roughly covering the left half of a 3024x2160pt
    # page, reported on the normalized 0-1000 grid (as real model output
    # does). A plot candidate bbox actually located in that area of the
    # page should now score highly against it.
    site_region_normalized = [0.0, 0.0, 500.0, 1000.0]  # left half, full height
    region_box = vision_bbox_to_page_points(
        site_region_normalized, page_width_pts=3024.0, page_height_pts=2160.0
    )
    from backend.schemas.geometry import BoundingBox

    plot_bbox = BoundingBox(min_x=100.0, min_y=100.0, max_x=1400.0, max_y=2000.0)
    score = region_score(plot_bbox, [region_box])
    assert score > 0.3  # meaningfully overlapping, not ~0 as the old bug produced


# --- Bug #2: vision-only dimension fusion ---------------------------------


def _vision_pages_with_plot_width(value: float, confidence: float = 0.9, page_number: int = 1):
    return [
        VisionPageResult(
            page_number=page_number,
            regions=[VisionRegion(id="r1", type="SITE_PLAN", bbox=[0, 0, 500, 500], confidence=0.9)],
            dimensions=[
                VisionDimension(
                    value=value,
                    unit="m",
                    type="PLOT_WIDTH",
                    region_id="r1",
                    bbox=[100, 100, 200, 120],
                    evidence=str(value),
                    confidence=confidence,
                )
            ],
            page_width_pts=3024.0,
            page_height_pts=2160.0,
        )
    ]


def test_vision_only_dimension_admitted_when_no_native_candidate_exists():
    vision_pages = _vision_pages_with_plot_width(9.14)
    result = vision_only_dimensions(vision_pages, existing_dimensions=[], page=0)
    assert len(result) == 1
    assert result[0].semantic_type is DimensionSemanticType.PLOT_WIDTH
    assert abs(result[0].value_metres - 9.14) < 1e-6
    assert result[0].dimension.geometry is None  # no CV geometry backs it


def test_vision_only_dimension_skipped_when_native_candidate_already_has_the_value():
    vision_pages = _vision_pages_with_plot_width(9.14)
    native = Dimension(label="9.14 m", magnitude=9.14, unit="m", geometry=None, page=0)
    result = vision_only_dimensions(vision_pages, existing_dimensions=[native], page=0)
    # Already represented by a native Dimension -- must not double-count
    # the same physical reading as two independent evidence sources.
    assert result == []


def test_vision_only_dimension_rejected_below_confidence_threshold():
    vision_pages = _vision_pages_with_plot_width(9.14, confidence=0.2)
    result = vision_only_dimensions(vision_pages, existing_dimensions=[], page=0)
    assert result == []


def test_vision_only_dimension_ignores_wrong_page():
    vision_pages = _vision_pages_with_plot_width(9.14, page_number=2)
    result = vision_only_dimensions(vision_pages, existing_dimensions=[], page=0)  # page 0 -> vision page 1
    assert result == []
