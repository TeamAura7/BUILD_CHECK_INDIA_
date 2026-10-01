"""
Tests for `backend.spatial_reasoning.vision_consistency`, the cross-region
numeric consistency guard.

Reproduces the exact PLAN5 pattern found live: a `SITE_PLAN` region and a
`GROUND_FLOOR_PLAN` region report the same two numbers for two different
semantic dimension types (PLOT_WIDTH/PLOT_DEPTH vs BUILDING_WIDTH/
BUILDING_DEPTH).
"""

from __future__ import annotations

from backend.schemas.vision import VisionArea, VisionDimension, VisionPageResult, VisionRegion
from backend.spatial_reasoning.vision_consistency import (
    DUPLICATE_MARKER,
    flag_cross_region_duplicates,
    is_flagged_duplicate,
    sanitize_vision_result,
)


def _page_like_plan5() -> VisionPageResult:
    return VisionPageResult(
        page_number=1,
        regions=[
            VisionRegion(id="region_1", type="SITE_PLAN", confidence=0.95),
            VisionRegion(id="region_2", type="GROUND_FLOOR_PLAN", confidence=0.95),
        ],
        dimensions=[
            VisionDimension(value=17.59, unit="m", type="PLOT_WIDTH", region_id="region_1", evidence="17.59", confidence=0.95),
            VisionDimension(value=9.14, unit="m", type="PLOT_DEPTH", region_id="region_1", evidence="9.14", confidence=0.95),
            VisionDimension(value=13.09, unit="m", type="BUILDING_WIDTH", region_id="region_1", evidence="13.09", confidence=0.95),
            # The pattern-completion bug: GROUND_FLOOR_PLAN repeats the
            # plot's own width/depth as if they were its building's.
            VisionDimension(value=17.59, unit="m", type="BUILDING_WIDTH", region_id="region_2", evidence="17.59", confidence=0.95),
            VisionDimension(value=9.14, unit="m", type="BUILDING_DEPTH", region_id="region_2", evidence="9.14", confidence=0.95),
        ],
    )


def test_flags_the_plan5_cross_region_duplicate_pattern():
    page = _page_like_plan5()
    notes = flag_cross_region_duplicates(page)
    assert len(notes) == 2  # PLOT_WIDTH<->BUILDING_WIDTH and PLOT_DEPTH<->BUILDING_DEPTH

    by_type_region = {(d.type, d.region_id): d for d in page.dimensions}

    # The correct SITE_PLAN BUILDING_WIDTH (13.09) is untouched: it never
    # matched a different-type value in a different region.
    assert by_type_region[("BUILDING_WIDTH", "region_1")].confidence == 0.95
    assert not is_flagged_duplicate(by_type_region[("BUILDING_WIDTH", "region_1")].evidence)

    # The flagged pair is downgraded on BOTH sides.
    assert by_type_region[("PLOT_WIDTH", "region_1")].confidence < 0.95
    assert by_type_region[("BUILDING_WIDTH", "region_2")].confidence < 0.95
    assert is_flagged_duplicate(by_type_region[("PLOT_WIDTH", "region_1")].evidence)
    assert is_flagged_duplicate(by_type_region[("BUILDING_WIDTH", "region_2")].evidence)
    assert DUPLICATE_MARKER in by_type_region[("BUILDING_WIDTH", "region_2")].evidence


def test_same_semantic_type_across_regions_is_not_flagged():
    """Two floors legitimately sharing a room dimension is not suspicious."""
    page = VisionPageResult(
        page_number=1,
        regions=[
            VisionRegion(id="r1", type="GROUND_FLOOR_PLAN"),
            VisionRegion(id="r2", type="FIRST_FLOOR_PLAN"),
        ],
        dimensions=[
            VisionDimension(value=3.0, unit="m", type="ROOM_DIMENSION", region_id="r1", confidence=0.9),
            VisionDimension(value=3.0, unit="m", type="ROOM_DIMENSION", region_id="r2", confidence=0.9),
        ],
    )
    notes = flag_cross_region_duplicates(page)
    assert notes == []
    assert all(d.confidence == 0.9 for d in page.dimensions)


def test_same_region_duplicate_values_are_not_flagged():
    """Two different fields in the SAME region sharing a value is not the pattern-completion signature."""
    page = VisionPageResult(
        page_number=1,
        regions=[VisionRegion(id="r1", type="SITE_PLAN")],
        dimensions=[
            VisionDimension(value=1.0, unit="m", type="LEFT_SETBACK", region_id="r1", confidence=0.9),
            VisionDimension(value=1.0, unit="m", type="RIGHT_SETBACK", region_id="r1", confidence=0.9),
        ],
    )
    notes = flag_cross_region_duplicates(page)
    assert notes == []


def test_areas_are_checked_too():
    page = VisionPageResult(
        page_number=1,
        regions=[
            VisionRegion(id="r1", type="AREA_STATEMENT"),
            VisionRegion(id="r2", type="GROUND_FLOOR_PLAN"),
        ],
        areas=[
            VisionArea(value=160.77, unit="m2", type="PLOT_AREA", region_id="r1", confidence=0.9),
            VisionArea(value=160.77, unit="m2", type="BUILDING_FOOTPRINT_AREA", region_id="r2", confidence=0.9),
        ],
    )
    notes = flag_cross_region_duplicates(page)
    assert len(notes) == 1
    assert all(a.confidence < 0.9 for a in page.areas)


def test_sanitize_vision_result_adds_warnings_and_handles_none():
    assert sanitize_vision_result(None) is None

    from backend.schemas.vision import VisionDocumentResult

    doc = VisionDocumentResult(pages=[_page_like_plan5()], model_name="test")
    sanitized = sanitize_vision_result(doc)
    assert sanitized is doc
    assert any("CROSS_REGION_CONSISTENCY" in w for w in doc.pages[0].warnings)


def test_downgraded_confidence_falls_below_the_production_pipeline_grounding_threshold():
    """
    `spatial_reasoning.vision_semantics.vision_values` (the production
    pipeline.py consumption path, separate from final_fusion.py) requires
    confidence >= 0.75 by default. A flagged duplicate starting at a typical
    0.95 must drop below that threshold so it can no longer single-handedly
    determine a field's value there.
    """
    page = _page_like_plan5()
    flag_cross_region_duplicates(page)
    flagged = [d for d in page.dimensions if d.region_id == "region_2" and d.type == "BUILDING_WIDTH"][0]
    assert flagged.confidence < 0.75
