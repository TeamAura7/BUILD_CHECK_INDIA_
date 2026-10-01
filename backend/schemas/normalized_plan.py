"""
NormalizedPlan — the single, resolved, municipality-independent
representation of a building plan.

This is THE contract. Every teammate downstream of extraction (RAG, RASE,
RuleEngine, report generation, frontend) builds against this shape and
NOTHING else. It intentionally contains no municipality-specific
thresholds — those live in externally-loaded runtime rules (see
backend/runtime_rules/contracts.py) and are matched against these fields.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from backend.schemas.candidates import SetbackMeasurement
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, Conflict, ValueField
from backend.schemas.geometry import NormalizedGeometry
from backend.schemas.units import UnitValue


class PlotSection(BaseModel):
    geometry: Optional[NormalizedGeometry] = None
    width: ValueField[float]
    depth: ValueField[float]
    area: ValueField[float]


class BuildingSection(BaseModel):
    geometry: Optional[NormalizedGeometry] = None
    width: ValueField[float]
    depth: ValueField[float]
    footprint_area: ValueField[float]
    floor_count: Optional[ValueField[int]] = None


class RoadSection(BaseModel):
    geometry: Optional[NormalizedGeometry] = None
    width: ValueField[float]


class SetbackSection(BaseModel):
    front: ValueField[float]
    rear: ValueField[float]
    left: ValueField[float]
    right: ValueField[float]

    def as_measurements(self) -> list[SetbackMeasurement]:
        return [
            SetbackMeasurement(side=side, distance=getattr(self, side))
            for side in ("front", "rear", "left", "right")
        ]


class NormalizedPlan(BaseModel):
    """
    Fully resolved plan, one instance per submitted building plan.

    `evidence` and `confidence` at the top level summarize plan-wide
    extraction quality; per-field evidence/confidence lives inside each
    ValueField. `conflicts` aggregates every Conflict raised anywhere in
    the plan so the RuleEngine (or a human reviewer) can act on them
    without walking the whole tree.
    """

    plan_id: str
    source_document_id: str

    plot: PlotSection
    building: BuildingSection
    road: RoadSection
    setbacks: SetbackSection

    coverage: ValueField[float] = Field(
        ..., description="Ground coverage, canonical unit = %"
    )
    far: ValueField[float] = Field(
        ..., description="Floor Area Ratio, canonical unit = ratio"
    )

    conflicts: list[Conflict] = Field(default_factory=list)
    overall_confidence_note: Optional[str] = None

    # Extended plan information used by the UI/reporting layer. These are
    # intentionally optional so older rule sets remain compatible.
    floor_areas: dict[str, ValueField[float]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    building_height_estimated: Optional[ValueField[float]] = None
    # Regulatory height used by the 2025 draft Table 8, which explicitly
    # excludes the stilt floor. It is separate from the generic estimated
    # height so the draft cannot silently assume a different height definition.
    building_height_excluding_stilt: Optional[ValueField[float]] = None
    # Explicit planning-area classification required by the 2003 Table 6
    # coverage/FAR table. The engine never guesses A/B/C from geometry.
    development_area: Optional[ValueField[str]] = None
    # Canonical building-use classification used by municipality rules.
    # Expected values include residential, commercial, public_semi_public,
    # traffic_transportation, public_utility, and named special uses such as
    # hospital/college. The engine never guesses this classification.
    building_use: Optional[ValueField[str]] = None

    @model_validator(mode="after")
    def _derive_context_from_metadata(self) -> "NormalizedPlan":
        # Keep canonical rule-context fields separate from free-form metadata.
        # These values are never inferred from geometry or an LLM.
        if self.development_area is None:
            raw = self.metadata.get("development_area")
            if isinstance(raw, str) and raw.strip():
                self.development_area = ValueField[str](
                    value=raw.strip().upper(),
                    confidence=Confidence(
                        level=ConfidenceLevel.HIGH,
                        reason="Development-area class supplied explicitly in plan metadata.",
                    ),
                    source="plan metadata",
                )
        if self.building_use is None:
            raw = self.metadata.get("building_use", self.metadata.get("building_type"))
            if isinstance(raw, str) and raw.strip():
                normalized = self._normalize_building_use(raw)
                if normalized:
                    self.building_use = ValueField[str](
                        value=normalized,
                        confidence=Confidence(
                            level=ConfidenceLevel.HIGH,
                            reason="Building-use classification supplied explicitly in plan metadata.",
                        ),
                        source="plan metadata",
                    )
        if self.building_height_excluding_stilt is None:
            raw_height = self.metadata.get("building_height_excluding_stilt")
            if isinstance(raw_height, (int, float)):
                self.building_height_excluding_stilt = ValueField[float](
                    value=float(raw_height),
                    normalized_value=UnitValue(magnitude=float(raw_height), unit="m"),
                    confidence=Confidence(
                        level=ConfidenceLevel.HIGH,
                        reason="Regulatory height excluding stilt supplied explicitly in plan metadata.",
                    ),
                    source="plan metadata",
                )
        return self

    @staticmethod
    def _normalize_building_use(raw: str) -> Optional[str]:
        text = " ".join(raw.strip().lower().replace("-", " ").split())
        aliases = {
            "residential building": "residential",
            "residential": "residential",
            "house": "residential",
            "commercial building": "commercial",
            "commercial": "commercial",
            "public and semi public": "public_semi_public",
            "public & semi public": "public_semi_public",
            "public/semi-public": "public_semi_public",
            "public semi public": "public_semi_public",
            "traffic and transportation": "traffic_transportation",
            "traffic transportation": "traffic_transportation",
            "public utility": "public_utility",
            "hospital": "hospital",
            "health centre": "health_centre_nursing_home",
            "health center": "health_centre_nursing_home",
            "nursing home": "health_centre_nursing_home",
            "health centre/nursing home": "health_centre_nursing_home",
            "nursery school": "nursery_primary_school",
            "primary school": "nursery_primary_school",
            "nursery/primary school": "nursery_primary_school",
            "secondary school": "secondary_school",
            "college": "college",
        }
        return aliases.get(text, text if text in {
            "residential", "commercial", "public_semi_public",
            "traffic_transportation", "public_utility", "hospital",
            "health_centre_nursing_home", "nursery_primary_school",
            "secondary_school", "college"
        } else None)

    def missing_field_names(self) -> list[str]:
        """Convenience for report generation / REQUIRES_REVIEW triage."""
        missing = []
        candidates = {
            "plot.width": self.plot.width,
            "plot.depth": self.plot.depth,
            "plot.area": self.plot.area,
            "building.width": self.building.width,
            "building.depth": self.building.depth,
            "building.footprint_area": self.building.footprint_area,
            "road.width": self.road.width,
            "setbacks.front": self.setbacks.front,
            "setbacks.rear": self.setbacks.rear,
            "setbacks.left": self.setbacks.left,
            "setbacks.right": self.setbacks.right,
            "coverage": self.coverage,
            "far": self.far,
            "building_use": self.building_use,
            "development_area": self.development_area,
            "building_height_estimated": self.building_height_estimated,
            "building.floor_count": self.building.floor_count,
        }
        for name, field in candidates.items():
            # Some entries here (building_use, development_area,
            # building_height_estimated, building.floor_count) are plain
            # Optional[...] fields on the model, not always a ValueField --
            # they're metadata that often can't be inferred from geometry at
            # all, so an unset one is skipped here rather than reported
            # missing. A field that got at least this far as a ValueField.missing(...)
            # is a different situation (something was resolved-to-absent)
            # and is what this method exists to surface.
            if field is not None and field.confidence.level == ConfidenceLevel.MISSING:
                missing.append(name)
        return missing
