"""
Architecture V2, Phase 6 -- tests for
`backend.spatial_reasoning.evidence_graph_adapters`, the additive
translation layer from `dimension_classification.ClassifiedDimension`/
`areas.LabeledArea` into `EvidenceCandidate` records.
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel, EntityKind, EvidenceKind
from backend.schemas.geometry import Line, Point
from backend.spatial_reasoning.areas import LabeledArea
from backend.spatial_reasoning.dimension_classification import ClassifiedDimension, DimensionSemanticType
from backend.spatial_reasoning.evidence_graph_adapters import (
    classified_dimension_to_evidence_candidate,
    classified_dimensions_to_evidence_candidates,
    labeled_area_to_evidence_candidate,
    labeled_areas_to_evidence_candidates,
)
from backend.schemas.geometry import Dimension


def _classified(
    semantic_type=DimensionSemanticType.PLOT_WIDTH,
    confidence=ConfidenceLevel.HIGH,
    magnitude=17.59,
    unit="m",
    geometry=None,
    label="17.59",
) -> ClassifiedDimension:
    dim = Dimension(label=label, magnitude=magnitude, unit=unit, geometry=geometry)
    return ClassifiedDimension(
        dimension=dim, semantic_type=semantic_type, confidence=confidence,
        reasoning="geometry and label text both indicate PLOT_WIDTH.",
    )


def test_classified_dimension_carries_its_own_geometry_directly():
    line = Line(start=Point(x=0, y=0), end=Point(x=17.59, y=0))
    classified = _classified(geometry=line)
    candidate = classified_dimension_to_evidence_candidate(classified, "ec-1")
    assert candidate.geometry == line
    assert candidate.raw_value.magnitude == 17.59
    assert candidate.entity_kind_hint == EntityKind.PLOT


def test_confidence_level_maps_to_a_real_nonzero_score_for_high_and_medium():
    high = classified_dimension_to_evidence_candidate(_classified(confidence=ConfidenceLevel.HIGH), "ec-1")
    medium = classified_dimension_to_evidence_candidate(_classified(confidence=ConfidenceLevel.MEDIUM), "ec-2")
    low = classified_dimension_to_evidence_candidate(_classified(confidence=ConfidenceLevel.LOW), "ec-3")
    assert high.source_confidence > medium.source_confidence > low.source_confidence
    assert high.source_confidence >= 0.85  # above enums.VISION_CONFIDENCE_HIGH_THRESHOLD


def test_kind_defaults_to_native_text_and_is_caller_overridable():
    default_kind = classified_dimension_to_evidence_candidate(_classified(), "ec-1")
    assert default_kind.kind == EvidenceKind.NATIVE_TEXT
    ocr_kind = classified_dimension_to_evidence_candidate(_classified(), "ec-2", kind=EvidenceKind.OCR_TEXT)
    assert ocr_kind.kind == EvidenceKind.OCR_TEXT


def test_semantic_type_maps_to_entity_kind_hint():
    setback = classified_dimension_to_evidence_candidate(
        _classified(semantic_type=DimensionSemanticType.FRONT_SETBACK), "ec-1",
    )
    assert setback.entity_kind_hint == EntityKind.SETBACK
    road = classified_dimension_to_evidence_candidate(
        _classified(semantic_type=DimensionSemanticType.ROAD_WIDTH), "ec-2",
    )
    assert road.entity_kind_hint == EntityKind.ROAD


def test_batch_conversion_mints_stable_ordinal_ids():
    classified = [_classified(), _classified(semantic_type=DimensionSemanticType.PLOT_DEPTH)]
    candidates = classified_dimensions_to_evidence_candidates(classified, id_prefix="dim")
    assert [c.id for c in candidates] == ["dim-0", "dim-1"]


def test_labeled_area_becomes_a_document_statement_candidate():
    area = LabeledArea(label="built-up area", value=85.09, unit="sq_m", source_text="Built-up Area: 85.09 Sq.m")
    candidate = labeled_area_to_evidence_candidate(area, "area-1")
    assert candidate.kind == EvidenceKind.DOCUMENT_STATEMENT
    assert candidate.raw_value.magnitude == 85.09
    assert candidate.entity_kind_hint is None
    assert "built-up area" in candidate.provenance_note


def test_labeled_areas_batch_conversion():
    areas = [
        LabeledArea(label="carpet area", value=70.0, unit="sq_m", source_text="Carpet Area: 70.0 Sq.m"),
        LabeledArea(label="plinth area", value=90.0, unit="sq_m", source_text="Plinth Area: 90.0 Sq.m"),
    ]
    candidates = labeled_areas_to_evidence_candidates(areas, id_prefix="area")
    assert [c.id for c in candidates] == ["area-0", "area-1"]
    assert all(c.kind == EvidenceKind.DOCUMENT_STATEMENT for c in candidates)


def test_every_candidate_has_a_mandatory_provenance_note():
    classified_candidate = classified_dimension_to_evidence_candidate(_classified(), "ec-1")
    area_candidate = labeled_area_to_evidence_candidate(
        LabeledArea(label="carpet area", value=70.0, unit="sq_m", source_text="x"), "area-1",
    )
    assert classified_candidate.provenance_note
    assert area_candidate.provenance_note
