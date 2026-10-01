"""
Regression tests for the eval harness's MISSING vs. N/A distinction on DXF
plans: a field with no production code path other than native DXF
TEXT/MTEXT matching (see `_TEXT_ONLY_DXF_FIELDS`) is structurally
unreachable -- not merely unresolved -- on a document with zero native text
entities, and must be scored N/A rather than MISSING so it stops counting
against every future measurement on that document class.
"""

from __future__ import annotations

from backend.tools.eval_harness import _TEXT_ONLY_DXF_FIELDS, PlanResult, _score


def test_text_only_field_missing_becomes_na_when_na_fields_given():
    expected = {"coverage": 65.09, "plot.width": 10.0}
    actual = {"plot.width": 10.0}  # coverage never produced

    results = _score(expected, actual, na_fields=_TEXT_ONLY_DXF_FIELDS)

    by_field = {r.field: r for r in results}
    assert by_field["coverage"].verdict == "N/A"
    assert by_field["plot.width"].verdict == "CORRECT"


def test_text_only_field_stays_missing_without_na_fields():
    expected = {"coverage": 65.09}
    actual: dict = {}

    results = _score(expected, actual)

    assert results[0].verdict == "MISSING"


def test_road_width_is_not_in_the_text_only_set():
    # road.width has a real geometry-only fallback (`_pick_road_polygon`,
    # layer name / adjacency, no text involved) -- it must never be
    # blanket-marked N/A just because a document has no native text.
    assert "road.width" not in _TEXT_ONLY_DXF_FIELDS


def test_na_fields_excluded_from_accuracy_and_completeness():
    expected = {"coverage": 65.09, "plot.width": 10.0, "plot.depth": 12.0}
    actual = {"plot.width": 10.0, "plot.depth": 999.0}  # depth wrong, coverage unreachable

    results = _score(expected, actual, na_fields=_TEXT_ONLY_DXF_FIELDS)
    result = PlanResult(plan_id="TEST", pdf_path="test.dxf", field_results=results, doc_type="dxf")

    counts = result.counts()
    assert counts["N/A"] == 1
    assert counts["CORRECT"] == 1
    assert counts["WRONG"] == 1

    # Accuracy denominator is CORRECT+WRONG+MISSING+FALSE_POSITIVE = 2, not 3.
    assert result.accuracy() == 0.5

    # Completeness denominator excludes the N/A field too: only plot.width
    # and plot.depth were ever scoreable, and both got a value.
    assert result.completeness() == 1.0


def test_ground_truth_null_still_takes_priority_over_na_fields():
    # A field the ground truth itself says must stay unresolved (expected is
    # None) is scored on the FALSE_POSITIVE/CORRECT axis regardless of
    # whether it's also in na_fields -- na_fields only ever affects the
    # "expected has a value but none was produced" branch.
    expected = {"coverage": None}
    actual: dict = {}

    results = _score(expected, actual, na_fields=_TEXT_ONLY_DXF_FIELDS)

    assert results[0].verdict == "CORRECT"
