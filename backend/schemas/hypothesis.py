"""
Structural hypothesis contract (Architecture V2, Phase 2).

See ARCHITECTURE_V2.md, Deliverables C/D, for the full design rationale.
Short version: a region-clustering fragment (or a group of fragments) is
never directly a "plot" or "building" — it is a `StructuralHypothesis`
claiming to be one, scored against a small, named, auditable set of
constraints (closure, side count, orthogonality, area plausibility,
dimension agreement, sheet-frame exclusion, contradiction penalties). The
evidence decision engine (Phase 7) compares competing `StructuralHypothesis`
records for the same role and accepts, rejects, or abstains — it never picks
the first plausible-looking one.

This module is purely additive and does not change `ExtractionResult`/
`NormalizedPlan`/`candidate_geometry.PlotCandidate` today. Once a hypothesis
is accepted, later phases project it into the existing
`PlotCandidate`/`BuildingCandidate`/`RoadCandidate` shape
(`backend/schemas/candidates.py`) so the pipeline's existing
`NormalizedPlan`-building code receives real candidates for the first time
(see ARCHITECTURE_V2.md Deliverable D.6) — this module does not attempt that
bridge itself.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.enums import HypothesisIdentity
from backend.schemas.geometry import Polygon


class ConstraintResult(BaseModel):
    """One named, auditable scoring term contributing to a hypothesis's total score.

    Every term is named (`name`) so a rejection can be explained in the
    vocabulary a compliance reviewer can read, not just as an opaque number
    — mirrors the discipline `dxf_extractor._score_region_resolution` and
    `document_evidence.EVIDENCE_SCORE_WEIGHTS` already apply independently
    (see ARCHITECTURE_V2.md Deliverable B.6/C.2), generalized into one shape
    both can share.
    """

    name: str = Field(..., description="e.g. 'boundary_closure', 'dimension_agreement'")
    score: float = Field(..., description="Signed contribution to the hypothesis's total_score")
    passed: bool
    detail: str = Field(..., description="Human-readable explanation, mandatory")


class StructuralHypothesis(BaseModel):
    """
    One candidate interpretation of a group of drawing fragments as a single
    structural object (a plot boundary, a building, a road, or sheet
    furniture to be excluded).

    `identity` widens `EntityKind` via `HypothesisIdentity` rather than
    inventing a parallel vocabulary (ARCHITECTURE_V2.md Deliverable B.9/D.3).
    `supporting_node_ids` references `DrawingNode` ids
    (`backend/schemas/evidence_graph.py`) rather than embedding geometry
    directly, so a hypothesis can be re-scored cheaply as new evidence nodes
    are added without re-serializing the whole graph.
    """

    id: str
    identity: HypothesisIdentity
    supporting_node_ids: list[str] = Field(
        default_factory=list, description="DrawingNode ids composing this hypothesis"
    )
    geometry: Optional[Polygon] = Field(default=None, description="The reconstructed/pooled geometry, if any")
    constraint_results: list[ConstraintResult] = Field(default_factory=list)
    total_score: float = 0.0
    document_evidence_ids: list[str] = Field(
        default_factory=list,
        description="EvidenceCandidate ids of kind OCR_TEXT/NATIVE_TEXT/DOCUMENT_STATEMENT",
    )
    vision_evidence_ids: list[str] = Field(
        default_factory=list, description="EvidenceCandidate ids of kind VISION_SEMANTIC"
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Human-readable, e.g. 'geometry 15.35m vs printed 17.59m, unassociated'",
    )
    rejected_alternative_ids: list[str] = Field(
        default_factory=list,
        description="Sibling StructuralHypothesis ids this one beat (their own constraint_results carry why)",
    )


__all__ = ["ConstraintResult", "StructuralHypothesis"]
