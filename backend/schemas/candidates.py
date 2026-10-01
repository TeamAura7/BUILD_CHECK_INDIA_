"""
Candidate entities produced during spatial reasoning, BEFORE they are
resolved into the single authoritative NormalizedPlan.

A "candidate" may be one of several competing interpretations of the same
region of the drawing (e.g. two possible plot boundary polygons). The
NormalizedPlan holds the resolved winner per field; candidates are kept
around for auditability and for the frontend to show "other
interpretations we considered".
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.enums import EntityKind
from backend.schemas.evidence import ValueField
from backend.schemas.geometry import NormalizedGeometry


class _CandidateBase(BaseModel):
    id: str
    kind: EntityKind
    geometry: Optional[NormalizedGeometry] = None
    confidence_note: Optional[str] = None


class PlotCandidate(_CandidateBase):
    kind: EntityKind = EntityKind.PLOT
    width: ValueField[float]
    depth: ValueField[float]
    area: ValueField[float]


class BuildingCandidate(_CandidateBase):
    kind: EntityKind = EntityKind.BUILDING
    width: ValueField[float]
    depth: ValueField[float]
    footprint_area: ValueField[float]
    floor_count: Optional[ValueField[int]] = None


class RoadCandidate(_CandidateBase):
    kind: EntityKind = EntityKind.ROAD
    width: ValueField[float]
    name_or_label: Optional[str] = None


class SetbackMeasurement(BaseModel):
    """A single directional setback (front/rear/left/right)."""

    side: str = Field(..., description="'front' | 'rear' | 'left' | 'right'")
    distance: ValueField[float]
    measured_between: Optional[str] = Field(
        default=None, description="e.g. 'building footprint edge to plot boundary'"
    )
