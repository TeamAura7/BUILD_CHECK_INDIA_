"""
Regression test for a confirmed live bug in `NormalizedPlan`'s
`_derive_context_from_metadata` model validator
(`backend/schemas/normalized_plan.py`): the `building_height_excluding_stilt`
branch builds a `UnitValue(...)`, but `UnitValue` was not imported in that
module -- constructing a `NormalizedPlan` with a raw numeric
`building_height_excluding_stilt` already present in `metadata` at
construction time would raise `NameError: UnitValue` instead of producing the
expected `ValueField`.
"""
from __future__ import annotations

from backend.schemas.evidence import ValueField
from backend.schemas.normalized_plan import (
    BuildingSection,
    NormalizedPlan,
    PlotSection,
    RoadSection,
    SetbackSection,
)


def _missing() -> ValueField[float]:
    return ValueField[float].missing("test fixture")


def test_building_height_excluding_stilt_from_metadata_does_not_raise():
    plan = NormalizedPlan(
        plan_id="p",
        source_document_id="doc",
        plot=PlotSection(width=_missing(), depth=_missing(), area=_missing()),
        building=BuildingSection(width=_missing(), depth=_missing(), footprint_area=_missing()),
        road=RoadSection(width=_missing()),
        setbacks=SetbackSection(front=_missing(), rear=_missing(), left=_missing(), right=_missing()),
        coverage=_missing(),
        far=_missing(),
        metadata={"building_height_excluding_stilt": 12.5},
    )
    assert plan.building_height_excluding_stilt is not None
    assert plan.building_height_excluding_stilt.value == 12.5
    assert plan.building_height_excluding_stilt.normalized_value is not None
    assert plan.building_height_excluding_stilt.normalized_value.unit == "m"
