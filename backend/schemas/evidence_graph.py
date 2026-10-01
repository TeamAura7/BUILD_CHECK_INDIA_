"""
Drawing Evidence Graph contract (Architecture V2, Phase 2).

See ARCHITECTURE_V2.md, Deliverables C/D, for the full design rationale.
Short version: perception modules (DXF entity walking, PDF native-text
scanning, OCR, vision) should emit `EvidenceCandidate` records instead of
each privately resolving a winner and writing straight into a semantic
field. `DrawingNode`/`DrawingRelationship` connect those candidates (and raw
geometry) into a graph that hypothesis generation and constraint scoring
read from.

This module is purely additive: nothing here is consumed by
`ExtractionResult`/`NormalizedPlan`/`ComplianceResult` yet (the same
"additive, not yet wired in" status `backend/schemas/regions.py`'s
`DetectedRegion` already has). No existing extractor is required to
construct these yet — that migration happens in later phases, behind a
feature flag, per ARCHITECTURE_V2.md's Deliverable E.

`DrawingNode`/`DrawingRelationship` generalize
`backend.spatial_reasoning.site_graph.SiteGraphNode`/`SiteGraphEdge` (already
format-agnostic, already unifies several duplicated "is this
nested/adjacent" implementations) to also cover non-geometric evidence
nodes and scored/typed relationship edges, rather than introducing a third,
competing graph representation alongside `SiteGraph` and
`gnn_extraction.graph_builder.EntityGraph`.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from backend.schemas.enums import EntityKind, EvidenceKind, RelationType
from backend.schemas.geometry import BoundingBox, Line, Point, Polygon
from backend.schemas.units import UnitValue


class EvidenceCandidate(BaseModel):
    """
    A single, not-yet-decided observation from one source.

    This is what perception emits instead of privately resolving a value.
    Many `EvidenceCandidate`s may compete to support (or contradict) the
    same field; the evidence decision engine (Architecture V2, Phase 7)
    is the only thing that turns a set of these into an authoritative
    `ValueField`. Deliberately distinct from `TextEvidence`/`GeometryEvidence`
    (`backend/schemas/evidence.py`), which remain the "evidence attached to
    an already-accepted `ValueField`" shape — an accepted `EvidenceCandidate`
    is projected into a `TextEvidence`/`GeometryEvidence` entry at decision
    time, it does not replace them.
    """

    id: str
    kind: EvidenceKind
    entity_kind_hint: Optional[EntityKind] = Field(
        default=None, description="What this MIGHT be evidence for, not a commitment"
    )
    raw_value: Optional[UnitValue] = None
    text: Optional[str] = Field(default=None, description="Raw OCR/native text, if applicable")
    geometry: Optional[Union[Point, Line, Polygon, BoundingBox]] = None

    source_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "This source's own reliability (0-1). Never defaulted silently — "
            "every constructor of an EvidenceCandidate must supply a real "
            "number; use backend.schemas.enums.confidence_from_source when "
            "converting this into a ValueField-level Confidence later."
        ),
    )
    source_module: str = Field(..., description="e.g. 'dxf_text_recovery.recover_caption'")
    render_resolution: Optional[int] = Field(
        default=None, description="Render DPI/px used to produce this candidate, for adaptive-computation audit trail"
    )
    provenance_note: str = Field(..., description="Human-readable explanation, mandatory (not optional)")


class DrawingNode(BaseModel):
    """
    One node in the Drawing Evidence Graph: either a piece of raw geometry,
    an `EvidenceCandidate`, or a region-clustering fragment awaiting
    hypothesis assembly.
    """

    id: str
    node_kind: str = Field(
        ..., description="'geometry' | 'evidence_candidate' | 'region_fragment'"
    )
    geometry_ref: Optional[Polygon] = Field(
        default=None, description="Populated for 'geometry'/'region_fragment' nodes"
    )
    evidence_ref: Optional[str] = Field(
        default=None, description="EvidenceCandidate.id, populated for 'evidence_candidate' nodes"
    )
    role_hint: Optional[EntityKind] = Field(
        default=None, description="Widened form of SiteGraphNode.NodeRole; not a commitment"
    )


class DrawingRelationship(BaseModel):
    """
    One typed, scored edge in the Drawing Evidence Graph.

    Deterministic geometric relations (NEAR/COLLINEAR/PARALLEL/
    PERPENDICULAR/INTERSECTS/CONTINUES/ENCLOSES/INSIDE/ADJACENT/
    ALIGNED_WITH) are computed directly from geometry and carry
    confidence=1.0. Association relations (MEASURES/BELONGS_TO) and
    detected disagreements (CONFLICTS_WITH) are produced by scoring logic
    and carry a real, sub-1.0 confidence. A future GNN relationship-inference
    pass (Architecture V2 Phase 12) writes rows here too, tagged
    `derivation="gnn_inference"` — it is one more scored relationship
    source, never a final value, by construction of this schema rather than
    by convention.
    """

    id: str
    relation: RelationType
    source_node_id: str
    target_node_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    derivation: str = Field(
        ..., description="'deterministic_geometry' | 'association_scoring' | 'gnn_inference'"
    )
    note: str = ""


__all__ = ["EvidenceCandidate", "DrawingNode", "DrawingRelationship"]
