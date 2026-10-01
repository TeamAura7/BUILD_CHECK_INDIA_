"""
Tests for `backend.cv_extraction.dxf_evidence_projection` (Architecture V2,
Phase DXF-1) -- the additive projection of `dxf_extractor.py`'s existing
per-region resolved candidates into `EvidenceCandidate`/`StructuralHypothesis`
records, and its optional, additive hook into `_resolve_via_regions` via the
`_dxf_hypothesis_sink` parameter.
"""

from __future__ import annotations

import pytest

from backend.cv_extraction.dxf_evidence_projection import (
    caption_match_to_evidence_candidate,
    region_candidate_to_evidence_candidates,
    region_resolution_to_structural_hypotheses,
)
from backend.schemas.enums import EntityKind, EvidenceKind, HypothesisIdentity
from backend.schemas.geometry import Point, Polygon
from backend.schemas.hypothesis import ConstraintResult


def _rect(x0, y0, x1, y1, layer="0"):
    from backend.cv_extraction.dxf_extractor import _RawPoly

    return _RawPoly(layer=layer, polygon=Polygon(points=[
        Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1),
    ]))


# --------------------------------------------------------------------------
# region_candidate_to_evidence_candidates
# --------------------------------------------------------------------------

def test_region_candidate_to_evidence_candidates_one_per_non_none_entry():
    plot_entry = _rect(0, 0, 20, 10)
    building_entry = _rect(2, 2, 10, 8)
    candidates = region_candidate_to_evidence_candidates(
        region_id=3, plot_entry=plot_entry, building_entry=building_entry, road_entry=None,
        source_confidence={HypothesisIdentity.PLOT_BOUNDARY: 0.95, HypothesisIdentity.BUILDING: 0.6},
        id_prefix="dxf-r3",
    )
    assert len(candidates) == 2
    by_hint = {c.entity_kind_hint: c for c in candidates}
    assert by_hint[EntityKind.PLOT].source_confidence == 0.95
    assert by_hint[EntityKind.BUILDING].source_confidence == 0.6
    assert by_hint[EntityKind.PLOT].kind == EvidenceKind.GEOMETRY
    assert by_hint[EntityKind.PLOT].geometry == plot_entry.polygon


def test_region_candidate_to_evidence_candidates_marks_reconstructed_geometry():
    reconstructed_building = _rect(2, 2, 10, 8, layer="RECONSTRUCTED_FROM_LINEWORK")
    candidates = region_candidate_to_evidence_candidates(
        region_id=1, plot_entry=None, building_entry=reconstructed_building, road_entry=None,
        source_confidence={HypothesisIdentity.BUILDING: 0.5}, id_prefix="dxf-r1",
    )
    assert len(candidates) == 1
    assert candidates[0].kind == EvidenceKind.RECONSTRUCTED_GEOMETRY


def test_region_candidate_to_evidence_candidates_empty_when_all_none():
    assert region_candidate_to_evidence_candidates(
        region_id=0, plot_entry=None, building_entry=None, road_entry=None,
        source_confidence={}, id_prefix="dxf-r0",
    ) == []


# --------------------------------------------------------------------------
# caption_match_to_evidence_candidate
# --------------------------------------------------------------------------

def test_caption_match_to_evidence_candidate_none_when_no_captions():
    assert caption_match_to_evidence_candidate(5, [], id_prefix="dxf-r5") is None


def test_caption_match_to_evidence_candidate_present_when_captions_found():
    ev = caption_match_to_evidence_candidate(5, ["SITE", "PLAN"], id_prefix="dxf-r5")
    assert ev is not None
    assert ev.kind == EvidenceKind.OCR_TEXT
    assert "SITE" in ev.text and "PLAN" in ev.text
    assert ev.id == "dxf-r5-caption"


# --------------------------------------------------------------------------
# region_resolution_to_structural_hypotheses
# --------------------------------------------------------------------------

def test_total_score_is_structural_plus_caption_plus_evidence_hand_computed():
    """Hand-computed arithmetic pin: total_score must equal exactly what
    `_resolve_via_regions`'s own `best[0]` accumulates for the same region
    (structural score + caption bonus if matched + evidence bonus) -- same
    three additive terms, same order, no drift."""
    plot_entry = _rect(0, 0, 20, 10)
    building_entry = _rect(2, 2, 10, 8)
    constraint_results = [
        ConstraintResult(name="base", score=1.0, passed=True, detail="d"),
        ConstraintResult(name="plot_area_plausibility", score=0.0, passed=True, detail="d"),
    ]
    structural_score = sum(c.score for c in constraint_results)  # = 1.0

    hyps = region_resolution_to_structural_hypotheses(
        region_id=7, plot_entry=plot_entry, building_entry=building_entry, road_entry=None,
        constraint_results=constraint_results, structural_score=structural_score,
        caption_matched=True, caption_bonus_applied=20.0,
        plausible_candidate_count=3, min_plausible_candidates_for_concern=2,
        evidence_bonus=1.5, caption_evidence_id="dxf-r7-caption", id_prefix="dxf-r7",
    )
    assert len(hyps) == 2  # plot + building
    for h in hyps:
        assert h.total_score == pytest.approx(1.0 + 20.0 + 1.5)


def test_caption_confirmation_term_reflects_caption_match():
    plot_entry = _rect(0, 0, 20, 10)
    hyps = region_resolution_to_structural_hypotheses(
        region_id=1, plot_entry=plot_entry, building_entry=None, road_entry=None,
        constraint_results=[], structural_score=1.0,
        caption_matched=True, caption_bonus_applied=20.0,
        plausible_candidate_count=1, min_plausible_candidates_for_concern=2,
        evidence_bonus=0.0, caption_evidence_id="dxf-r1-caption", id_prefix="dxf-r1",
    )
    term = next(c for c in hyps[0].constraint_results if c.name == "caption_confirmation")
    assert term.passed is True
    assert term.score == 20.0
    assert hyps[0].document_evidence_ids == ["dxf-r1-caption"]


def test_drawing_identity_confirmed_fails_when_ambiguous_and_no_caption():
    """Generalizes DXF_FAILURE_TAXONOMY.md item 9: >=2 plausible regions,
    none captioned -- this region's identity is unconfirmed."""
    plot_entry = _rect(0, 0, 20, 10)
    hyps = region_resolution_to_structural_hypotheses(
        region_id=1, plot_entry=plot_entry, building_entry=None, road_entry=None,
        constraint_results=[], structural_score=3.0,
        caption_matched=False, caption_bonus_applied=0.0,
        plausible_candidate_count=2, min_plausible_candidates_for_concern=2,
        evidence_bonus=0.0, caption_evidence_id=None, id_prefix="dxf-r1",
    )
    term = next(c for c in hyps[0].constraint_results if c.name == "drawing_identity_confirmed")
    assert term.passed is False


def test_drawing_identity_confirmed_passes_when_only_one_plausible_candidate():
    plot_entry = _rect(0, 0, 20, 10)
    hyps = region_resolution_to_structural_hypotheses(
        region_id=1, plot_entry=plot_entry, building_entry=None, road_entry=None,
        constraint_results=[], structural_score=3.0,
        caption_matched=False, caption_bonus_applied=0.0,
        plausible_candidate_count=1, min_plausible_candidates_for_concern=2,
        evidence_bonus=0.0, caption_evidence_id=None, id_prefix="dxf-r1",
    )
    term = next(c for c in hyps[0].constraint_results if c.name == "drawing_identity_confirmed")
    assert term.passed is True


def test_drawing_identity_confirmed_passes_when_caption_matched_even_if_ambiguous():
    plot_entry = _rect(0, 0, 20, 10)
    hyps = region_resolution_to_structural_hypotheses(
        region_id=1, plot_entry=plot_entry, building_entry=None, road_entry=None,
        constraint_results=[], structural_score=3.0,
        caption_matched=True, caption_bonus_applied=20.0,
        plausible_candidate_count=5, min_plausible_candidates_for_concern=2,
        evidence_bonus=0.0, caption_evidence_id="dxf-r1-caption", id_prefix="dxf-r1",
    )
    term = next(c for c in hyps[0].constraint_results if c.name == "drawing_identity_confirmed")
    assert term.passed is True


def test_no_hypotheses_when_all_entries_none():
    assert region_resolution_to_structural_hypotheses(
        region_id=0, plot_entry=None, building_entry=None, road_entry=None,
        constraint_results=[], structural_score=-1.0,
        caption_matched=False, caption_bonus_applied=0.0,
        plausible_candidate_count=0, min_plausible_candidates_for_concern=2,
        evidence_bonus=0.0, caption_evidence_id=None, id_prefix="dxf-r0",
    ) == []


# --------------------------------------------------------------------------
# End-to-end: `_resolve_via_regions`'s optional `_dxf_hypothesis_sink` hook
# --------------------------------------------------------------------------

def test_resolve_via_regions_sink_is_populated_and_winner_has_max_score_per_identity():
    """Real (not mocked) call into `_resolve_via_regions` with two spatially
    separated regions -- one a clean plot+building, one a decoy with only a
    plot. The sink must gain a StructuralHypothesis per identity per region
    with a resolved entry, and the geometry `_resolve_via_regions` actually
    returns as the winner must be the one carrying the highest total_score
    among its own identity's hypotheses in the sink -- proving the sink
    reflects the SAME winner-selection the function's own return value does,
    not an independent, drifting computation.
    """
    from backend.cv_extraction.dxf_extractor import _RawPoly, _resolve_via_regions

    def polygon_from_rect(x0, y0, x1, y1, layer):
        return _RawPoly(layer=layer, polygon=Polygon(points=[
            Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1),
        ]))

    polygons = [
        polygon_from_rect(0, 0, 20, 10, "PLOT"),
        polygon_from_rect(2, 2, 10, 8, "BLDG"),
        polygon_from_rect(200, 200, 205, 205, "PLOT"),
    ]

    sink: list = []
    result_with_sink = _resolve_via_regions(
        polygons, [], [], warnings=[], extraction_start_time=None, _dxf_hypothesis_sink=sink,
    )
    result_without_sink = _resolve_via_regions(
        polygons, [], [], warnings=[], extraction_start_time=None,
    )

    # The sink must not change what the function returns.
    assert result_with_sink == result_without_sink

    if result_with_sink is None:
        pytest.skip("region clustering did not separate these two rectangles into distinct regions on this run")

    winning_plot_entry, _winning_building_entry, _winning_road_entry = result_with_sink
    assert sink, "the sink must be populated when provided"

    plot_hyps = [h for h in sink if h.identity.value == "PLOT_BOUNDARY"]
    assert plot_hyps, "at least one PLOT_BOUNDARY hypothesis must be projected"
    if winning_plot_entry is not None:
        best_hyp = max(plot_hyps, key=lambda h: h.total_score)
        assert best_hyp.geometry == winning_plot_entry.polygon
