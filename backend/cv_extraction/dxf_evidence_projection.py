"""
Architecture V2, Phase DXF-1: additive projection of `dxf_extractor.py`'s
existing per-region resolved plot/building/road candidates into
`EvidenceCandidate`/`StructuralHypothesis` records.

See `ARCHITECTURE_V2.md` (Deliverables C/D) and the DXF redesign plan for the
full rationale. Short version: `dxf_extractor._resolve_via_regions` already
computes everything a `StructuralHypothesis` needs for every region it
considers (a structural plausibility score via `hypothesis_scoring.
score_plot_building_resolution`, a caption-confirmation signal, an
"unconfirmed drawing identity" ambiguity flag, a recovered-evidence bonus) --
it just discards everything except the single highest-scoring region's own
resolution (`best = ...` in `_resolve_via_regions`'s Pass 3). This module
does not change that selection at all; it only re-packages the SAME numbers,
for EVERY region considered, into the shared Evidence Graph schema, so a
future decision engine (Phase DXF-2/3) can arbitrate between them explicitly
instead of the winner-take-all `best = ...` loop being the only place this
information ever existed.

Nothing in `dxf_extractor.py` calls into this module yet as anything other
than an optional, additive sink (`_dxf_hypothesis_sink` parameter) -- see
`_resolve_via_regions`'s own docstring for the call site. No existing caller
passes it, so this file changes zero observable behavior on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from backend.schemas.enums import EntityKind, EvidenceKind, HypothesisIdentity
from backend.schemas.evidence_graph import EvidenceCandidate
from backend.schemas.hypothesis import ConstraintResult, StructuralHypothesis

if TYPE_CHECKING:
    # Forward-ref only, to avoid a circular import -- `dxf_extractor.py` is
    # the (optional, additive) CALLER of this module, not the other way
    # around. Matches the existing forward-ref style `dxf_extractor.py`
    # itself already uses for `_RawPoly` in its own type hints.
    from backend.cv_extraction.dxf_extractor import _RawPoly

# Layers `_entry_source_confidence` already treats as "not exact vector
# geometry" (Vision fallback, or one of the three fragment-reconstruction
# methods) -- reused here, not re-derived, to classify an EvidenceCandidate's
# `kind` the same way that function already classifies confidence.
_RECONSTRUCTED_LAYER_NAMES = frozenset({
    "VISION_FALLBACK",
    "RECONSTRUCTED_FROM_LINEWORK",
    "RECONSTRUCTED_WALL_UNION",
    "RECONSTRUCTED_ENVELOPE",
})

_IDENTITY_ENTITY_KIND_HINT: dict[HypothesisIdentity, EntityKind] = {
    HypothesisIdentity.PLOT_BOUNDARY: EntityKind.PLOT,
    HypothesisIdentity.BUILDING: EntityKind.BUILDING,
    HypothesisIdentity.ROAD: EntityKind.ROAD,
}


def _identity_entries(
    plot_entry: Optional["_RawPoly"], building_entry: Optional["_RawPoly"], road_entry: Optional["_RawPoly"],
) -> dict[HypothesisIdentity, Optional["_RawPoly"]]:
    return {
        HypothesisIdentity.PLOT_BOUNDARY: plot_entry,
        HypothesisIdentity.BUILDING: building_entry,
        HypothesisIdentity.ROAD: road_entry,
    }


def region_candidate_to_evidence_candidates(
    *,
    region_id: int,
    plot_entry: Optional["_RawPoly"],
    building_entry: Optional["_RawPoly"],
    road_entry: Optional["_RawPoly"],
    source_confidence: dict[HypothesisIdentity, float],
    id_prefix: str,
) -> list[EvidenceCandidate]:
    """One `EvidenceCandidate` per non-None entry among plot/building/road.

    `source_confidence` must be keyed by the SAME `HypothesisIdentity` this
    function checks, with values coming from `_entry_source_confidence`'s own
    already-computed float for that entry -- never re-derived here, so this
    projection can never drift from what `dxf_extractor.py` itself already
    decided to trust.
    """
    candidates: list[EvidenceCandidate] = []
    for identity, entry in _identity_entries(plot_entry, building_entry, road_entry).items():
        if entry is None:
            continue
        kind = EvidenceKind.RECONSTRUCTED_GEOMETRY if entry.layer in _RECONSTRUCTED_LAYER_NAMES else EvidenceKind.GEOMETRY
        candidates.append(EvidenceCandidate(
            id=f"{id_prefix}-{identity.value.lower()}-geom",
            kind=kind,
            entity_kind_hint=_IDENTITY_ENTITY_KIND_HINT[identity],
            geometry=entry.polygon,
            source_confidence=source_confidence.get(identity, 0.5),
            source_module="backend.cv_extraction.dxf_extractor._resolve_via_regions",
            provenance_note=(
                f"Region {region_id}'s resolved {identity.value.lower()} geometry (layer '{entry.layer}')."
            ),
        ))
    return candidates


def caption_match_to_evidence_candidate(
    region_id: int, captions: list[str], *, id_prefix: str,
) -> Optional[EvidenceCandidate]:
    """One `EvidenceCandidate(kind=OCR_TEXT)` for a recognized drawing-type
    caption on this region (`dxf_text_recovery.py`'s sheet-wide caption
    recovery pass — see `DXF_FAILURE_TAXONOMY.md` item 0). This is the
    evidence the `caption_confirmation` `ConstraintResult` below is actually
    based on. `None` when no caption was recognized for this region.
    """
    if not captions:
        return None
    return EvidenceCandidate(
        id=f"{id_prefix}-caption",
        kind=EvidenceKind.OCR_TEXT,
        text="; ".join(captions),
        # Matches the confidence already assigned elsewhere in dxf_extractor.py
        # to other native/recovered-text reads (e.g. the 0.9 hardcoded for
        # _AREA_FIELD_PATTERNS matches) -- not a new, independently-guessed number.
        source_confidence=0.9,
        source_module="dxf_text_recovery.recover_dimension_and_caption_evidence",
        provenance_note=f"Drawing-type caption(s) recognized on region {region_id}: {', '.join(captions)}.",
    )


def region_resolution_to_structural_hypotheses(
    *,
    region_id: int,
    plot_entry: Optional["_RawPoly"],
    building_entry: Optional["_RawPoly"],
    road_entry: Optional["_RawPoly"],
    constraint_results: list[ConstraintResult],
    structural_score: float,
    caption_matched: bool,
    caption_bonus_applied: float,
    plausible_candidate_count: int,
    min_plausible_candidates_for_concern: int,
    evidence_bonus: float,
    caption_evidence_id: Optional[str],
    id_prefix: str,
) -> list[StructuralHypothesis]:
    """One `StructuralHypothesis` per non-None entry among plot/building/road,
    all sharing the same constraint terms and `total_score` -- a region's
    plot/building/road resolution is scored and won/lost JOINTLY today (see
    `_score_region_resolution`'s own docstring), so splitting the hypotheses
    by identity while keeping their scoring shared reflects that accurately,
    rather than pretending they were scored independently.

    `constraint_results`/`structural_score` MUST come from
    `hypothesis_scoring.score_plot_building_resolution` (via `dxf_extractor.
    _score_region_resolution_detailed`) for the SAME `plot_entry`/
    `building_entry` -- passed through, never recomputed with different
    logic. Two more named terms are appended here, generalizing two
    already-existing ad-hoc mechanisms into the same auditable
    `ConstraintResult` shape instead of a side-channel boolean flag:

      - `caption_confirmation`: generalizes `_SITE_PLAN_CAPTION_SCORE_BONUS`
        (DXF_FAILURE_TAXONOMY.md item 0).
      - `drawing_identity_confirmed`: generalizes `_RawPoly.
        unconfirmed_drawing_identity` (DXF_FAILURE_TAXONOMY.md item 9).

    `total_score = structural_score + caption_bonus_applied + evidence_bonus`
    -- this MUST equal whatever `_resolve_via_regions`'s own `best[0]`
    accumulates for this same region, by construction (same three additive
    terms, same order); a hand-computed regression test pins this.
    """
    terms = list(constraint_results)
    terms.append(ConstraintResult(
        name="caption_confirmation",
        score=caption_bonus_applied,
        passed=caption_matched,
        detail=(
            f"a drawing-type caption was recognized on region {region_id}, confirming it as the site plan"
            if caption_matched else
            f"no drawing-type caption was recognized on region {region_id}"
        ),
    ))
    identity_confirmed = not (
        plausible_candidate_count >= min_plausible_candidates_for_concern and not caption_matched
    )
    terms.append(ConstraintResult(
        name="drawing_identity_confirmed",
        score=0.0,
        passed=identity_confirmed,
        detail=(
            "drawing identity is either the only plausible candidate on this sheet or independently "
            "confirmed by a recognized caption"
            if identity_confirmed else
            f"{plausible_candidate_count} region(s) on this sheet independently scored as plausible "
            "drawings and no caption confirmed any of them -- this region's identity as the actual "
            "site plan is unconfirmed"
        ),
    ))
    if evidence_bonus:
        terms.append(ConstraintResult(
            name="recovered_evidence_bonus",
            score=evidence_bonus,
            passed=evidence_bonus > 0,
            detail=f"recovered dimension/caption evidence contributed {evidence_bonus:+.2f} to this region's score",
        ))

    total_score = structural_score + caption_bonus_applied + evidence_bonus
    document_evidence_ids = [caption_evidence_id] if caption_evidence_id else []

    hypotheses: list[StructuralHypothesis] = []
    for identity, entry in _identity_entries(plot_entry, building_entry, road_entry).items():
        if entry is None:
            continue
        hypotheses.append(StructuralHypothesis(
            id=f"{id_prefix}-{identity.value.lower()}",
            identity=identity,
            supporting_node_ids=[],
            geometry=entry.polygon,
            constraint_results=terms,
            total_score=total_score,
            document_evidence_ids=document_evidence_ids,
        ))
    return hypotheses


__all__ = [
    "region_candidate_to_evidence_candidates",
    "caption_match_to_evidence_candidate",
    "region_resolution_to_structural_hypotheses",
]
