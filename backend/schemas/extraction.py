"""
Contract for raw extraction output — what the CV/PDF/OCR modules produce,
BEFORE spatial reasoning resolves it into a NormalizedPlan.

This is deliberately loose/flat: extraction modules append whatever
candidates and text they found. Spatial reasoning (a later phase) is
responsible for turning this into the strongly-resolved NormalizedPlan.

Also defines forward-looking, NOT-implemented-in-Phase-1 abstractions for
multi-document cross-verification (DocumentSet / DocumentEntity /
Contradiction / ContradictionReport), so later phases have a stable
interface to build against without Phase 1 needing to implement the logic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.candidates import BuildingCandidate, PlotCandidate, RoadCandidate
from backend.schemas.enums import DocumentType
from backend.schemas.evidence import TextEvidence
from backend.schemas.geometry import Dimension, SpatialRelation
from backend.schemas.vision import VisionPageResult
from backend.schemas.independent_measurements import IndependentCVResult


class ExtractionResult(BaseModel):
    """
    Output of the raw extraction stage for a single source document.

    Downstream (spatial reasoning) consumes this to build a NormalizedPlan.
    RuleEngine and everything after it NEVER sees this directly.
    """

    document_id: str
    document_type: DocumentType
    page_count: int = Field(..., ge=1)

    plot_candidates: list[PlotCandidate] = Field(default_factory=list)
    building_candidates: list[BuildingCandidate] = Field(default_factory=list)
    road_candidates: list[RoadCandidate] = Field(default_factory=list)

    dimensions: list[Dimension] = Field(default_factory=list)
    text_evidence: list[TextEvidence] = Field(default_factory=list)
    spatial_relations: list[SpatialRelation] = Field(default_factory=list)

    # Optional multimodal semantic evidence. Empty when vision extraction is disabled.
    vision_pages: list[VisionPageResult] = Field(default_factory=list)
    # Independent site-plan CV/native/OCR evidence used by the final CV+Vision agreement layer.
    independent_cv: Optional[IndependentCVResult] = None

    warnings: list[str] = Field(default_factory=list)
    extractor_name: Optional[str] = None
    extractor_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Forward-looking abstractions — INTERFACES ONLY.
# Cross-document verification and DXF/BIM ingestion are future phases.
# Do not implement logic against these in Phase 1.
# ---------------------------------------------------------------------------


class DocumentEntity(BaseModel):
    """A single named entity as understood within one document's context."""

    document_id: str
    entity_id: str
    kind: str
    description: Optional[str] = None


class DocumentSet(BaseModel):
    """A collection of related documents describing the same project/plot."""

    project_id: str
    document_ids: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    """A single disagreement detected between two DocumentEntity records."""

    entity_a: DocumentEntity
    entity_b: DocumentEntity
    field_name: str
    description: str


class ContradictionReport(BaseModel):
    """Aggregate report of contradictions found across a DocumentSet."""

    project_id: str
    contradictions: list[Contradiction] = Field(default_factory=list)

    @property
    def has_contradictions(self) -> bool:
        return len(self.contradictions) > 0
