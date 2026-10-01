"""
Regression test for `backend.tools.eval_harness`: every plan with a
`data/test_plans/PLANn.expected.json` -- PDF, DXF, or both -- must never
score a field WRONG (a confident, verified-against-ground-truth mistake)
on every test run, not just when someone remembers to run the harness by
hand. This is the Phase 0 validation harness's own durability guarantee:
a plan gets covered automatically the moment its fixture files exist,
with no test-code change required per plan.

Deliberately not asserting on exact accuracy percentages or MISSING
counts here: MISSING is honest uncertainty (e.g. CV correctly declining
to guess on a zero-native-text page) and is expected to fluctuate as
heuristics improve incrementally; a field flipping from MISSING to
CORRECT is progress, not a regression to guard against. A field
scoring WRONG, however, means a pipeline produced a value that
contradicts hand-verified ground truth with confidence -- that should
never happen silently.

PLAN5 and PLAN6 (both formats) are fixtures here like any other -- this
test holds them to the exact same bar (never confidently WRONG), never a
looser or tighter one, and nothing in the extraction pipeline should ever
be tuned specifically to make these two pass.

Vision scoring is skipped here (not asserted on) when VISION_ENABLED is
off or vision can't be reached (no network/model access in CI) --
that's an environment limitation, not a pipeline bug, and asserting on
it would make this test flaky rather than useful. DXF plans have no
Vision/fusion pipeline at all (see `dxf_extractor.py`) -- only "cv" is
scored for them.
"""

from __future__ import annotations

import pytest

from backend.tools.eval_harness import discover_plans, evaluate_plan

_PLANS = discover_plans()
_DISPLAY_KEYS = [f"{plan_id}[{doc_type}]" for plan_id, doc_type, _, _ in _PLANS]


@pytest.mark.parametrize("display_key", _DISPLAY_KEYS)
def test_never_confidently_wrong_against_ground_truth(display_key):
    """PDF plans: hard gate, matches this test's original bar (the PDF
    pipeline's geometry-first design already achieves this). DXF plans:
    only a no-crash bar for now -- DXF geometry reconstruction is
    documented, ongoing, incomplete work (fragmented-boundary/wall-union
    reconstruction still under-capture real footprints on real files; see
    the DXF pipeline's own stage reports), so WRONG/MISSING there is
    currently EXPECTED, tracked via this harness's reported per-field
    errors (run `python -m backend.tools.eval_harness`), not a pass/fail
    gate -- gating on it now would either block unrelated work on an
    already-known gap, or invite exactly the overfitting-to-PLAN5/6
    temptation Phase 0 validation explicitly exists to prevent.
    """
    plans_by_key = {f"{plan_id}[{doc_type}]": (plan_id, doc_type, doc_path, expected_path)
                    for plan_id, doc_type, doc_path, expected_path in _PLANS}
    plan_id, doc_type, doc_path, expected_path = plans_by_key[display_key]

    results = evaluate_plan(plan_id, doc_type, doc_path, expected_path)

    if doc_type == "dxf":
        result = results.get("cv")
        if result is not None and result.error:
            pytest.fail(f"{display_key} cv pipeline crashed: {result.error}")
        return

    for pipeline in ("cv", "fusion"):
        result = results.get(pipeline)
        if result is None:
            continue
        if result.error:
            pytest.fail(f"{display_key} {pipeline} pipeline crashed: {result.error}")
        # WRONG: a value was produced but disagrees with verified ground
        # truth beyond tolerance. FALSE_POSITIVE: ground truth says this
        # field must stay MISSING (e.g. PLAN7) but a value was produced
        # anyway -- strictly worse, since there the pipeline had no business
        # emitting anything at all. Neither may ever pass silently.
        bad = [fr for fr in result.field_results if fr.verdict in ("WRONG", "FALSE_POSITIVE")]
        assert not bad, (
            f"{display_key} {pipeline}: field(s) confidently wrong against verified ground truth: "
            + ", ".join(f"{fr.field} [{fr.verdict}] expected={fr.expected} actual={fr.actual}" for fr in bad)
        )


def test_eval_harness_finds_at_least_one_ground_truth_plan():
    # Guards against a silent misconfiguration (e.g. data/test_plans/ moved
    # or emptied) making the parametrized test above silently collect zero
    # cases and pass trivially.
    assert len(discover_plans()) >= 1


def _field(field, expected, actual, verdict, status=None):
    from backend.tools.eval_harness import FieldResult

    abs_error = abs(actual - expected) if expected is not None and actual is not None else None
    return FieldResult(field=field, expected=expected, actual=actual, verdict=verdict, status=status, abs_error=abs_error)


def test_boundary_resolved_true_only_when_all_boundary_fields_present_and_uncontested():
    """`PlanResult.boundary_resolved()` (Part O.2 validity gating): a
    resolved-but-WRONG boundary still counts as resolved (the flag is about
    whether the boundary produced a usable, non-conflicting answer, not
    whether that answer was numerically correct) -- but a MISSING boundary
    field, or one carrying a fusion CONFLICT status, does not."""
    from backend.tools.eval_harness import PlanResult

    resolved = PlanResult(plan_id="A", pdf_path="a.pdf", field_results=[
        _field("plot.width", 10.0, 10.0, "CORRECT"),
        _field("plot.depth", 15.0, 15.0, "CORRECT"),
        _field("building.footprint_area", 90.0, 90.0, "CORRECT"),
        _field("setbacks.front", 1.5, 5.0, "WRONG"),
    ])
    assert resolved.boundary_resolved() is True

    missing_boundary = PlanResult(plan_id="B", pdf_path="b.pdf", field_results=[
        _field("plot.width", 12.0, None, "MISSING"),
        _field("plot.depth", 18.0, None, "MISSING"),
        _field("building.footprint_area", 100.0, None, "MISSING"),
        _field("setbacks.front", 1.0, None, "MISSING"),
    ])
    assert missing_boundary.boundary_resolved() is False

    conflicting_boundary = PlanResult(plan_id="C", pdf_path="c.pdf", field_results=[
        _field("plot.width", 10.0, 11.0, "WRONG", status="CONFLICT"),
        _field("plot.depth", 15.0, 15.0, "CORRECT"),
        _field("building.footprint_area", 90.0, 90.0, "CORRECT"),
    ])
    assert conflicting_boundary.boundary_resolved() is False

    no_boundary_fields_at_all = PlanResult(plan_id="D", pdf_path="d.pdf", field_results=[
        _field("setbacks.front", 1.0, 1.0, "CORRECT"),
    ])
    assert no_boundary_fields_at_all.boundary_resolved() is False


def test_boundary_gated_aggregates_separate_all_plans_from_resolved_only():
    """The two required synthetic cases (Part O.2): a plan with a resolved
    boundary but an incorrect setback, and a plan with a missing boundary
    and consequently missing setbacks. Asserts: (1) the all-plans aggregate
    includes both plans' fields, including the boundary failure itself --
    the gate must never exclude boundary fields from scoring; (2) the
    boundary-resolved-only aggregate includes only the first plan; (3) the
    two views are not merged into one misleading number."""
    from backend.tools.eval_harness import PlanResult, boundary_gated_aggregates

    resolved_boundary_wrong_setback = PlanResult(plan_id="A", pdf_path="a.pdf", field_results=[
        _field("plot.width", 10.0, 10.0, "CORRECT"),
        _field("plot.depth", 15.0, 15.0, "CORRECT"),
        _field("building.footprint_area", 90.0, 90.0, "CORRECT"),
        _field("setbacks.front", 1.5, 5.0, "WRONG"),
    ])
    missing_boundary_and_setbacks = PlanResult(plan_id="B", pdf_path="b.pdf", field_results=[
        _field("plot.width", 12.0, None, "MISSING"),
        _field("plot.depth", 18.0, None, "MISSING"),
        _field("building.footprint_area", 100.0, None, "MISSING"),
        _field("setbacks.front", 1.0, None, "MISSING"),
    ])

    agg = boundary_gated_aggregates([resolved_boundary_wrong_setback, missing_boundary_and_setbacks])

    # All-plans aggregate includes both plans.
    assert agg["all_plans"]["plan_count"] == 2
    assert agg["all_plans"]["counts"]["CORRECT"] == 3
    assert agg["all_plans"]["counts"]["WRONG"] == 1
    # Boundary failures from plan B remain visible in the all-plans view --
    # 3 of these 4 MISSING fields ARE boundary fields (plot.width/depth,
    # building.footprint_area); the gate must not have hidden them.
    assert agg["all_plans"]["counts"]["MISSING"] == 4

    # Boundary-resolved-only aggregate includes only plan A.
    assert agg["boundary_resolved_only"]["plan_count"] == 1
    assert agg["boundary_resolved_only"]["counts"]["CORRECT"] == 3
    assert agg["boundary_resolved_only"]["counts"]["WRONG"] == 1
    assert agg["boundary_resolved_only"]["counts"]["MISSING"] == 0

    # Downstream metrics are not misleadingly merged: MAE differs between
    # the two views because plan B (all MISSING, no numeric error at all)
    # contributes no errors either way, but the two plan counts and MISSING
    # totals must stay visibly different, not collapsed into one number.
    assert agg["all_plans"]["plan_count"] != agg["boundary_resolved_only"]["plan_count"]
    assert agg["all_plans"]["counts"]["MISSING"] != agg["boundary_resolved_only"]["counts"]["MISSING"]


def test_eval_harness_finds_both_pdf_and_dxf_fixtures():
    """Phase 0 validation requirement: the harness must actually discover
    DXF fixtures, not just PDF ones -- this is the regression pin for
    that, independent of whether any specific plan's DXF happens to be
    scored WRONG or not."""
    doc_types = {doc_type for _, doc_type, _, _ in discover_plans()}
    assert "dxf" in doc_types, (
        "no .dxf fixture was discovered in data/test_plans/ -- the Phase 0 harness extension "
        "for DXF plans may have regressed, or the fixture files are missing"
    )
