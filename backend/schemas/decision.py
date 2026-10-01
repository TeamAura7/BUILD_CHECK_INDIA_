"""
Evidence decision contract (Architecture V2, Phase 2).

See ARCHITECTURE_V2.md, Deliverables C/D, for the full design rationale.
`Decision` is the record the (future, Phase 7) evidence decision engine
produces for each field: which candidate/hypothesis was accepted, which
competitors were considered and why they lost, and how the resulting
confidence was derived. `ValueField.decision_id`
(`backend/schemas/evidence.py`) points back at one of these once the engine
is wired in.

`Decision` replaces the `Contradiction`/`ContradictionReport` placeholders
in `backend/schemas/extraction.py` as the load-bearing schema for this idea.
Those two types were confirmed (ARCHITECTURE_V2.md Deliverable B.9) to have
zero constructors anywhere in the codebase — pure forward-looking
scaffolding for a "Phase 1: interfaces only" contract that was never built
against — so repurposing the concept here carries no migration risk. This
module does not delete `Contradiction`/`ContradictionReport`; a later phase
can retire them once nothing references them, which today is already true.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.enums import DecisionStatus
from backend.schemas.hypothesis import ConstraintResult


class Decision(BaseModel):
    """
    One field-level authoritative decision, with full provenance.

    Exactly one of ACCEPT / REJECT / CONFLICT / ABSTAIN
    (`backend.schemas.enums.DecisionStatus`) — never a residual "else"
    branch. `rejected` reuses `ConstraintResult`'s named/auditable shape so a
    losing candidate's rejection reads the same way a hypothesis's own
    internal scoring does, rather than as a free-text afterthought.
    """

    id: str = Field(..., description="Stable id, referenced by ValueField.decision_id")
    field_name: str = Field(..., description="e.g. 'plot.width'")
    status: DecisionStatus
    accepted_candidate_id: Optional[str] = Field(
        default=None, description="EvidenceCandidate or StructuralHypothesis id, when status == ACCEPT"
    )
    considered_candidate_ids: list[str] = Field(default_factory=list)
    rejected: list[ConstraintResult] = Field(
        default_factory=list, description="Why each loser lost, one or more ConstraintResult per rejected candidate"
    )
    resulting_value_field: Optional[str] = Field(
        default=None, description="Field path on NormalizedPlan this decision produced, e.g. 'plot.width'"
    )
    confidence_derivation: str = Field(
        ..., description="Human-readable: how the resulting confidence was composed, mandatory"
    )


__all__ = ["Decision"]
