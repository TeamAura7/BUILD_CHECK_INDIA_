from __future__ import annotations

from pathlib import Path

from backend.cv_extraction.site_plan import extract_independent_cv


PLAN2 = Path(__file__).resolve().parents[1] / "data" / "test_plans" / "PLAN2.pdf"

EXPECTED = {
    "plot.width": 12.19,
    "plot.depth": 18.28,
    "building.width": 10.59,
    "building.depth": 16.48,
    "road.width": 9.20,
    "setbacks.front": 1.00,
    "setbacks.rear": 0.80,
    "setbacks.left": 0.80,
    "setbacks.right": 0.80,
}


def _map(result):
    return {m.field: m for m in result.measurements if m.value_m is not None}


def test_plan2_independent_cv_resolves_site_plan_measurements():
    assert PLAN2.exists(), "PLAN2 fixture must be packaged with Phase 3 validation ZIP"
    result = extract_independent_cv(PLAN2, "PLAN2")
    values = _map(result)
    missing = sorted(set(EXPECTED) - set(values))
    assert not missing, f"Independent CV missed required PLAN2 fields: {missing}"
    for field, expected in EXPECTED.items():
        actual = values[field].value_m
        assert abs(actual - expected) <= 0.03, f"{field}: CV={actual}, expected={expected}"


def test_plan2_cv_setbacks_keep_spatial_evidence_and_do_not_average_duplicates():
    result = extract_independent_cv(PLAN2, "PLAN2")
    values = _map(result)
    for field in ("setbacks.front", "setbacks.rear", "setbacks.left", "setbacks.right"):
        assert values[field].source in {"NATIVE_TEXT", "DERIVED"}
        assert values[field].evidence
    assert values["setbacks.front"].value_m == 1.0
    assert values["setbacks.rear"].value_m == 0.8
    assert values["setbacks.left"].value_m == 0.8
    assert values["setbacks.right"].value_m == 0.8



def test_plan2_independent_cv_extracts_area_statement_fields():
    from backend.cv_extraction.site_plan import extract_independent_cv
    from pathlib import Path
    result = extract_independent_cv(Path("data/test_plans/PLAN2.pdf"), "PLAN2")
    values = {}
    for m in result.measurements:
        if m.value is not None:
            values.setdefault(m.field, []).append(m.value)
    assert values["plot.area"] == [222.83]
    assert values["building.footprint_area"] == [174.52]
    assert values["coverage"] == [78.32]
    assert values["far.area"] == [386.55]
    assert values["far"] == [1.73]
    assert values["building.gross_built_up_area"] == [579.9]
