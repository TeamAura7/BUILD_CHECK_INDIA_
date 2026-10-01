"""
Tests for `backend.spatial_reasoning.pdf_dxf_reconciliation.reconcile_pdf_dxf`
(the post-hoc PDF<->DXF cross-check). Constructs `NormalizedPlan` fixtures
directly, same style as `tests/test_final_fusion.py`, since this module
operates purely on two already-produced `NormalizedPlan`s -- it never calls
either extractor.
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.normalized_plan import (
    BuildingSection,
    NormalizedPlan,
    PlotSection,
    RoadSection,
    SetbackSection,
)
from backend.spatial_reasoning.pdf_dxf_reconciliation import COMPARED_FIELDS, reconcile_pdf_dxf


def _vf(value, level=ConfidenceLevel.HIGH):
    if value is None:
        return ValueField[float].missing("test fixture")
    return ValueField[float](value=value, confidence=Confidence(level=level, score=0.9, reason="test fixture"))


def _plan(
    plan_id="p", *, plot_width=12.0, plot_depth=18.0, plot_area=216.0,
    building_width=8.0, building_depth=12.0, footprint_area=96.0,
    floor_count=2, road_width=9.0,
    front=1.0, rear=1.0, left=1.0, right=1.0,
    coverage=44.0, far=0.5,
    level=ConfidenceLevel.HIGH,
) -> NormalizedPlan:
    return NormalizedPlan(
        plan_id=plan_id, source_document_id=plan_id,
        plot=PlotSection(width=_vf(plot_width, level), depth=_vf(plot_depth, level), area=_vf(plot_area, level)),
        building=BuildingSection(
            width=_vf(building_width, level), depth=_vf(building_depth, level),
            footprint_area=_vf(footprint_area, level),
            floor_count=(ValueField[int](value=floor_count, confidence=Confidence(level=level)) if floor_count is not None else None),
        ),
        road=RoadSection(width=_vf(road_width, level)),
        setbacks=SetbackSection(front=_vf(front, level), rear=_vf(rear, level), left=_vf(left, level), right=_vf(right, level)),
        coverage=_vf(coverage, level), far=_vf(far, level),
    )


def test_agreed_field_boosts_confidence_to_high():
    pdf_plan = _plan(plot_width=12.19, level=ConfidenceLevel.MEDIUM)
    dxf_plan = _plan(plot_width=12.20)  # within tolerance (0.15 abs)
    reconciled, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert reconciled.plot.width.value == 12.19
    assert reconciled.plot.width.confidence.level == ConfidenceLevel.HIGH
    assert reconciled.plot.width.conflict is None

    row = next(f for f in report.fields if f.field == "plot.width")
    assert row.status == "AGREED"
    assert row.shipped_value == 12.19


def test_conflict_field_ships_pdf_value_unchanged_but_flagged():
    pdf_plan = _plan(plot_width=12.19, level=ConfidenceLevel.HIGH)
    dxf_plan = _plan(plot_width=4.11)  # far beyond tolerance
    reconciled, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert reconciled.plot.width.value == 12.19  # PDF's own value, unchanged
    assert reconciled.plot.width.confidence.level == ConfidenceLevel.HIGH  # unchanged, NOT CONFLICTING
    assert reconciled.plot.width.conflict is not None
    assert "12.19" in reconciled.plot.width.conflict.description
    assert "4.11" in reconciled.plot.width.conflict.description

    row = next(f for f in report.fields if f.field == "plot.width")
    assert row.status == "CONFLICT"
    assert row.shipped_value == 12.19
    assert row.pdf_value == 12.19
    assert row.dxf_value == 4.11


def test_conflict_never_sets_confidence_level_to_conflicting():
    """Regression pin: `backend/compliance/engine.py` short-circuits ANY
    field with `confidence.level == CONFLICTING` to `CONFLICTING_EVIDENCE`
    regardless of `.value` -- setting that level here would silently
    discard the very "ship PDF's value" decision this module exists to
    make. Checked across every compared field, not just one."""
    pdf_plan = _plan()
    dxf_plan = _plan(plot_width=1.0, building_width=1.0, road_width=1.0, coverage=1.0, far=99.0)
    reconciled, _report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    for field_path in COMPARED_FIELDS:
        obj = reconciled
        for part in field_path.split("."):
            obj = getattr(obj, part)
        if obj is None:
            continue
        assert obj.confidence.level != ConfidenceLevel.CONFLICTING, field_path


def test_pdf_only_field_ships_unchanged_no_conflict():
    pdf_plan = _plan(road_width=9.2)
    dxf_plan = _plan()
    dxf_plan.road.width = ValueField[float].missing("DXF found no road")
    reconciled, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert reconciled.road.width.value == 9.2
    assert reconciled.road.width.conflict is None
    row = next(f for f in report.fields if f.field == "road.width")
    assert row.status == "PDF_ONLY"
    assert row.shipped_value == 9.2


def test_dxf_only_field_is_reported_but_not_backfilled():
    pdf_plan = _plan()
    pdf_plan.road.width = ValueField[float].missing("PDF found no road")
    dxf_plan = _plan(road_width=9.2)
    reconciled, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert reconciled.road.width.value is None  # NOT backfilled from DXF
    row = next(f for f in report.fields if f.field == "road.width")
    assert row.status == "DXF_ONLY"
    assert row.shipped_value is None
    assert row.dxf_value == 9.2


def test_both_missing_field_stays_missing():
    pdf_plan = _plan()
    pdf_plan.road.width = ValueField[float].missing("PDF found no road")
    dxf_plan = _plan()
    dxf_plan.road.width = ValueField[float].missing("DXF found no road")
    reconciled, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert reconciled.road.width.value is None
    row = next(f for f in report.fields if f.field == "road.width")
    assert row.status == "BOTH_MISSING"


def test_pdf_only_fields_like_building_use_are_never_compared():
    assert "building_use" not in COMPARED_FIELDS
    assert "development_area" not in COMPARED_FIELDS
    assert "building_height_estimated" not in COMPARED_FIELDS
    assert "floor_areas" not in COMPARED_FIELDS


def test_report_summary_counts_match_field_statuses():
    pdf_plan = _plan()
    dxf_plan = _plan(plot_width=1.0)  # only this one field conflicts
    _reconciled, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert sum(report.summary.values()) == len(COMPARED_FIELDS)
    assert report.summary["CONFLICT"] == 1
    assert report.has_conflicts is True
    status_counts = {"AGREED": 0, "CONFLICT": 0, "PDF_ONLY": 0, "DXF_ONLY": 0, "BOTH_MISSING": 0}
    for f in report.fields:
        status_counts[f.status] += 1
    assert status_counts == report.summary


def test_plan_conflicts_list_contains_every_conflict_field():
    pdf_plan = _plan()
    dxf_plan = _plan(plot_width=1.0, building_width=1.0)
    reconciled, _report = reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert len(reconciled.conflicts) == 2
    described_fields = {c.description for c in reconciled.conflicts}
    assert any("plot.width" in d for d in described_fields)
    assert any("building.width" in d for d in described_fields)


def test_neither_input_plan_is_mutated():
    pdf_plan = _plan(plot_width=12.19)
    dxf_plan = _plan(plot_width=4.11)
    reconcile_pdf_dxf(pdf_plan, dxf_plan)

    assert pdf_plan.plot.width.value == 12.19
    assert pdf_plan.plot.width.conflict is None
    assert dxf_plan.plot.width.value == 4.11
