from __future__ import annotations

import pytest

from backend.schemas.candidates import SetbackMeasurement
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, TextEvidence, ValueField
from backend.schemas.normalized_plan import (
    BuildingSection,
    NormalizedPlan,
    PlotSection,
    RoadSection,
    SetbackSection,
)
from backend.schemas.units import CanonicalUnit, UnitValue


def make_value_field(
    value: float,
    level: ConfidenceLevel = ConfidenceLevel.HIGH,
    raw_magnitude: float | None = None,
    raw_unit: str = "m",
) -> ValueField[float]:
    raw = None
    if raw_magnitude is not None:
        raw = UnitValue(magnitude=raw_magnitude, unit=raw_unit)
    normalized = UnitValue(magnitude=value, unit=CanonicalUnit.METRE.value)
    return ValueField[float](
        value=value,
        raw_value=raw,
        normalized_value=normalized,
        confidence=Confidence(level=level),
        source="test fixture",
        evidence=[TextEvidence(raw_text=f"{value} m", page=1)],
    )


@pytest.fixture
def sample_normalized_plan() -> NormalizedPlan:
    return NormalizedPlan(
        plan_id="plan-001",
        source_document_id="doc-001",
        plot=PlotSection(
            width=make_value_field(12.0, raw_magnitude=39.37, raw_unit="ft"),
            depth=make_value_field(18.0),
            area=make_value_field(216.0),
        ),
        building=BuildingSection(
            width=make_value_field(9.0),
            depth=make_value_field(14.0),
            footprint_area=make_value_field(126.0),
        ),
        road=RoadSection(width=make_value_field(9.0)),
        setbacks=SetbackSection(
            front=make_value_field(3.0),
            rear=make_value_field(2.0),
            left=make_value_field(1.5),
            right=make_value_field(1.5),
        ),
        coverage=make_value_field(58.3),
        far=make_value_field(1.75),
    )
