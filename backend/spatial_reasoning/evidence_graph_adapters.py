"""
Drawing Evidence Graph adapters for OCR/document evidence (Architecture V2,
Phase 6). See ARCHITECTURE_V2.md, Deliverable C.2 item 2, and the migration
table entries for `dimension_classification.py`/`areas.py`.

This module converts the ALREADY-COMPUTED output of the existing, good
semantic classifiers (`dimension_classification.classify_dimensions`,
`areas.find_labeled_areas`) into `EvidenceCandidate` records
(`backend/schemas/evidence_graph.py`). It does not change, re-implement, or
call into either classifier's own logic -- both remain exactly as they are;
this is a pure, additive translation layer at their output boundary.

What this module deliberately does NOT do: emit `DrawingRelationship
(MEASURES)` edges connecting an `EvidenceCandidate` to a real plot/building/
road `DrawingNode`. A `MEASURES` edge's entire value is that it points at an
ACTUAL node in a shared graph -- and no live extraction path yet threads a
real `DrawingNode` registry through the pipeline (that requires
`dxf_extractor.py`'s and `site_plan.py`'s own "Step 2" migration, deferred
per ARCHITECTURE_V2.md's Deliverable E). Fabricating a target node id here
just to populate a `MEASURES` edge would misrepresent an association that
was never actually computed against a real graph -- "wrong is worse than
missing" applies to architecture scaffolding, not only to extracted
numbers. Each `EvidenceCandidate` below instead carries its own associated
geometry directly (via `EvidenceCandidate.geometry`, already supported),
which is honest about what is actually known today: "this text is
associated with this line," not "this text measures that specific,
already-identified plot side."
"""

from __future__ import annotations

from typing import Optional

from backend.schemas.enums import ConfidenceLevel, EntityKind, EvidenceKind
from backend.schemas.evidence_graph import EvidenceCandidate
from backend.schemas.units import UnitValue
from backend.spatial_reasoning.areas import LabeledArea
from backend.spatial_reasoning.dimension_classification import ClassifiedDimension, DimensionSemanticType

# `ClassifiedDimension`/`LabeledArea` carry a categorical `ConfidenceLevel`,
# not a raw 0-1 score, but `EvidenceCandidate.source_confidence` requires a
# real float (never defaulted silently -- see its own docstring). These are
# representative point estimates for each category, not a rescored
# probability -- deliberately reusing this project's own already-established
# HIGH/LOW threshold band (`enums.VISION_CONFIDENCE_HIGH_THRESHOLD`/
# `_LOW_THRESHOLD`, 0.85/0.75) as anchors rather than inventing new numbers:
# HIGH sits above the high threshold, MEDIUM between the two, LOW at the
# band's own low anchor.
_CONFIDENCE_LEVEL_TO_REPRESENTATIVE_SCORE = {
    ConfidenceLevel.HIGH: 0.9,
    ConfidenceLevel.MEDIUM: 0.8,
    ConfidenceLevel.LOW: 0.5,
    ConfidenceLevel.CONFLICTING: 0.0,
    ConfidenceLevel.MISSING: 0.0,
}

_SEMANTIC_TYPE_TO_ENTITY_KIND = {
    DimensionSemanticType.PLOT_WIDTH: EntityKind.PLOT,
    DimensionSemanticType.PLOT_DEPTH: EntityKind.PLOT,
    DimensionSemanticType.BUILDING_WIDTH: EntityKind.BUILDING,
    DimensionSemanticType.BUILDING_DEPTH: EntityKind.BUILDING,
    DimensionSemanticType.FRONT_SETBACK: EntityKind.SETBACK,
    DimensionSemanticType.REAR_SETBACK: EntityKind.SETBACK,
    DimensionSemanticType.LEFT_SETBACK: EntityKind.SETBACK,
    DimensionSemanticType.RIGHT_SETBACK: EntityKind.SETBACK,
    DimensionSemanticType.ROAD_WIDTH: EntityKind.ROAD,
    DimensionSemanticType.ROOM_DIMENSION: EntityKind.UNKNOWN,
    DimensionSemanticType.OTHER: EntityKind.UNKNOWN,
}


def classified_dimension_to_evidence_candidate(
    classified: ClassifiedDimension,
    candidate_id: str,
    *,
    kind: EvidenceKind = EvidenceKind.NATIVE_TEXT,
    source_module: str = "dimension_classification.classify_dimensions",
) -> EvidenceCandidate:
    """
    Project one `ClassifiedDimension` (already scored by geometry+keyword
    agreement) into an `EvidenceCandidate`. The dimension's own associated
    line (`classified.dimension.geometry`), if any, is carried directly on
    the candidate -- see module docstring for why no `MEASURES` edge is
    emitted separately.

    `kind` must be supplied by the caller, not guessed here: a `Dimension`
    (`backend/schemas/geometry.py`) carries no field distinguishing native
    PDF text from OCR/vectorized-text recovery -- both the PDF-native and
    DXF-OCR-recovery paths produce identical `Dimension` shapes -- so the
    caller (which knows which extractor produced this batch) is the only
    honest source for this distinction. Defaults to `NATIVE_TEXT` because
    that is `classify_dimensions`'s own most common real caller (PDF native
    text); pass `EvidenceKind.OCR_TEXT` explicitly for OCR-sourced batches.
    """
    dim = classified.dimension
    return EvidenceCandidate(
        id=candidate_id,
        kind=kind,
        entity_kind_hint=_SEMANTIC_TYPE_TO_ENTITY_KIND.get(classified.semantic_type, EntityKind.UNKNOWN),
        raw_value=UnitValue(magnitude=dim.magnitude, unit=dim.unit) if dim.unit else None,
        text=dim.label,
        geometry=dim.geometry,
        source_confidence=_CONFIDENCE_LEVEL_TO_REPRESENTATIVE_SCORE[classified.confidence],
        source_module=source_module,
        provenance_note=(
            f"classified as {classified.semantic_type.value} ({classified.confidence.value}): {classified.reasoning}"
        ),
    )


def classified_dimensions_to_evidence_candidates(
    classified: list[ClassifiedDimension], id_prefix: str, *, kind: EvidenceKind = EvidenceKind.NATIVE_TEXT,
) -> list[EvidenceCandidate]:
    """Batch form of `classified_dimension_to_evidence_candidate`, minting
    stable, ordinal ids (`f"{id_prefix}-{i}"`) for each input."""
    return [
        classified_dimension_to_evidence_candidate(cd, candidate_id=f"{id_prefix}-{i}", kind=kind)
        for i, cd in enumerate(classified)
    ]


def labeled_area_to_evidence_candidate(area: LabeledArea, candidate_id: str) -> EvidenceCandidate:
    """
    Project one `LabeledArea` (an explicit carpet/built-up/plinth/floor
    area statement, per `areas.find_labeled_areas`'s own docstring, kept
    deliberately separate from footprint/plot area) into an
    `EvidenceCandidate` of kind `DOCUMENT_STATEMENT`.

    Confidence is not categorical here (unlike `ClassifiedDimension`) --
    `find_labeled_areas` is a direct regex match on printed text, which
    this project's own existing convention (see `dxf_extractor.py`'s
    native-TEXT-entity area matches) treats as high-confidence *evidence*
    (a real printed statement exists) without implying it is the
    authoritative value for any specific NormalizedPlan field -- that
    remains the evidence decision engine's job, not this adapter's.
    """
    return EvidenceCandidate(
        id=candidate_id,
        kind=EvidenceKind.DOCUMENT_STATEMENT,
        entity_kind_hint=None,  # a labeled area (carpet/built-up/etc.) has no single EntityKind of its own
        raw_value=UnitValue(magnitude=area.value, unit=area.unit) if area.unit else None,
        text=area.source_text,
        geometry=None,
        source_confidence=0.9,  # a direct regex match on printed document text, per this project's existing convention
        source_module="areas.find_labeled_areas",
        provenance_note=f"labeled area statement ({area.label}): {area.source_text!r}",
    )


def labeled_areas_to_evidence_candidates(areas: list[LabeledArea], id_prefix: str) -> list[EvidenceCandidate]:
    """Batch form of `labeled_area_to_evidence_candidate`."""
    return [
        labeled_area_to_evidence_candidate(area, candidate_id=f"{id_prefix}-{i}")
        for i, area in enumerate(areas)
    ]


__all__ = [
    "classified_dimension_to_evidence_candidate",
    "classified_dimensions_to_evidence_candidates",
    "labeled_area_to_evidence_candidate",
    "labeled_areas_to_evidence_candidates",
]
