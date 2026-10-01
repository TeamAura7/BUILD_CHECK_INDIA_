"""
Integration test for `reconcile_pdf_dxf` against REAL extractors and real
bundled plans (PLAN5, PLAN6) -- not the synthetic `NormalizedPlan` fixtures
`tests/test_pdf_dxf_reconciliation.py` uses.

Live extraction (see this test's own assertions, corroborated against
`data/test_plans/PLAN5.expected.json` / `PLAN6.expected.json`) shows the
DXF pipeline reads BOTH of these real DXFs at roughly a quarter of the
PDF's own (ground-truth-corroborated) scale -- e.g. PLAN5 plot.width:
PDF=17.59m vs DXF=4.74m. That is a genuine, pre-existing DXF extraction
bug (unrelated to this reconciliation feature -- see
DXF_FAILURE_TAXONOMY.md), and it is exactly the scenario this cross-check
exists to catch: nearly every shared geometric field on both plans comes
back CONFLICT, not AGREED. This test pins that PDF's value ships
unchanged in every one of those cases, never DXF's and never an average,
and that no field's confidence level is ever laundered to CONFLICTING
(see `backend/compliance/engine.py`'s CONFLICTING short-circuit, the
load-bearing finding documented in
`backend/spatial_reasoning/pdf_dxf_reconciliation.py`'s own docstring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.schemas.enums import ConfidenceLevel
from backend.spatial_reasoning.pdf_dxf_reconciliation import COMPARED_FIELDS, reconcile_pdf_dxf
from backend.spatial_reasoning.pipeline import build_normalized_plan

PLANS_DIR = Path(__file__).parent.parent / "data" / "test_plans"


def _reconcile(stem: str):
    pdf_path = PLANS_DIR / f"{stem}.pdf"
    dxf_path = PLANS_DIR / f"{stem}.dxf"
    if not (pdf_path.exists() and dxf_path.exists()):
        pytest.skip(f"{stem}.pdf/{stem}.dxf fixtures not present")

    pdf_extraction = PDFHybridExtractor(enable_vision=False).extract(pdf_path, stem)
    pdf_plan = build_normalized_plan(pdf_extraction, plan_id=f"plan-{stem}-pdf")
    dxf_extraction = DXFHybridExtractor().extract(dxf_path, stem)
    dxf_plan = build_normalized_plan(dxf_extraction, plan_id=f"plan-{stem}-dxf")

    return reconcile_pdf_dxf(pdf_plan, dxf_plan)


def _get_field(plan, path: str):
    obj = plan
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("stem", ["PLAN5", "PLAN6"])
def test_no_compared_field_ever_ships_conflicting_confidence_level(stem):
    """Regression pin for the compliance-engine finding: a CONFLICT here
    must never set `confidence.level = CONFLICTING`, or the compliance
    engine would refuse to use the very value this module just shipped."""
    reconciled, _report = _reconcile(stem)
    for field_path in COMPARED_FIELDS:
        vf = _get_field(reconciled, field_path)
        if vf is None:
            continue
        assert vf.confidence.level != ConfidenceLevel.CONFLICTING, field_path


@pytest.mark.parametrize("stem", ["PLAN5", "PLAN6"])
def test_conflict_fields_ship_pdfs_own_value_not_dxfs(stem):
    """On both real plans, the DXF pipeline reads a substantially
    different scale than the PDF -- most compared fields conflict. Every
    one of them must ship PDF's own value, unchanged."""
    reconciled, report = _reconcile(stem)
    conflict_rows = [f for f in report.fields if f.status == "CONFLICT"]
    assert conflict_rows, f"{stem}: expected at least one CONFLICT field to exercise the PDF-wins path"

    for row in conflict_rows:
        assert row.shipped_value == row.pdf_value
        assert row.shipped_value != row.dxf_value
        vf = _get_field(reconciled, row.field)
        assert vf.value == pytest.approx(row.pdf_value)
        assert vf.conflict is not None
        conflict_descriptions = {c.description for c in reconciled.conflicts}
        assert any(row.field in d for d in conflict_descriptions)


@pytest.mark.parametrize("stem", ["PLAN5", "PLAN6"])
def test_shipped_plan_matches_pdf_only_extraction_on_every_compared_field(stem):
    """The reconciled plan must never silently diverge from what the PDF
    pipeline alone would have produced for any compared field -- DXF can
    only flag (via `.conflict`) or boost confidence (on AGREED), never
    change the shipped number."""
    pdf_path = PLANS_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        pytest.skip(f"{stem}.pdf fixture not present")
    pdf_extraction = PDFHybridExtractor(enable_vision=False).extract(pdf_path, stem)
    pdf_only_plan = build_normalized_plan(pdf_extraction, plan_id=f"plan-{stem}-pdf-only")

    reconciled, _report = _reconcile(stem)
    for field_path in COMPARED_FIELDS:
        expected_vf = _get_field(pdf_only_plan, field_path)
        actual_vf = _get_field(reconciled, field_path)
        if expected_vf is None or expected_vf.value is None:
            assert actual_vf is None or actual_vf.value is None, field_path
        else:
            assert actual_vf is not None and actual_vf.value == pytest.approx(expected_vf.value), field_path


def test_plan5_reconciled_values_corroborate_the_ground_truthed_expected_json():
    """Cross-check the shipped (PDF-wins) values against PLAN5's own
    documented ground truth (`data/test_plans/PLAN5.expected.json`,
    `tests/test_real_plan_regression.py`'s PLAN5 fixture) -- confirms the
    reconciliation didn't corrupt PDF's already-correct reading.

    Uses the SAME relative 2% tolerance `test_real_plan_regression.py`'s own
    `_assert_close` uses for this exact plan (not a tighter absolute one):
    PLAN5's plot boundary is dash-dot drawn, and that file's own docstrings
    document ~7cm of slack in the reconstructed depth as expected, not a
    regression.
    """
    reconciled, _report = _reconcile("PLAN5")
    expected = {
        "plot.width": 17.59, "plot.depth": 9.14, "road.width": 10.0,
        "setbacks.front": 3.00, "setbacks.rear": 1.50,
    }
    for field_path, expected_value in expected.items():
        vf = _get_field(reconciled, field_path)
        assert vf.value == pytest.approx(expected_value, rel=0.02), field_path


def test_plan6_reconciled_values_corroborate_the_ground_truthed_expected_json():
    reconciled, _report = _reconcile("PLAN6")
    expected = {"plot.width": 10.00, "plot.depth": 13.10, "road.width": 7.3}
    for field_path, expected_value in expected.items():
        vf = _get_field(reconciled, field_path)
        assert vf.value == pytest.approx(expected_value, abs=0.05), field_path
