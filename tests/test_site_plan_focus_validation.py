from __future__ import annotations

from backend.vision_extraction.base import BaseArchitecturalPlanExtractor
from backend.vision_extraction.prompts import SITE_PLAN_FOCUS_PROMPT


def test_site_plan_focus_prompt_requires_spatial_setback_assignment_and_preserves_duplicates():
    text = SITE_PLAN_FOCUS_PROMPT.upper()
    for token in (
        "FRONT_SETBACK",
        "REAR_SETBACK",
        "LEFT_SETBACK",
        "RIGHT_SETBACK",
        "SPATIAL POSITION",
        "PRESERVE REPEATED VALUES",
        "DO NOT REUSE PLOT/BUILDING DIMENSIONS",
    ):
        assert token in text


def test_focus_bbox_remapping_keeps_crop_geometry_in_full_page_normalized_space():
    bbox = BaseArchitecturalPlanExtractor._remap_focus_bbox(
        [0, 0, 1000, 1000],
        (200.0, 300.0, 600.0, 700.0),
        1000.0,
        1000.0,
    )
    assert bbox == [200.0, 300.0, 600.0, 700.0]
