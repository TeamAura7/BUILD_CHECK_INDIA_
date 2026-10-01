"""
Tests for Architecture V2, Phase DXF-3: the feature-flagged wiring of
`structural_hypothesis_decision.decide_structural_hypothesis` into
`dxf_extractor._resolve_via_regions`, behind `Settings.
use_dxf_evidence_decision_engine` (default `False`).

Mirrors `evidence_decision_bridge.py`'s own verification discipline: the
flag-off path must be byte-identical to the pre-DXF-3 behavior (covered by
the full `test_dxf_extractor.py`/`test_real_plan_regression.py` suites,
run separately as this phase's hard regression gate); these tests cover
what's specific to the new wiring itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import get_settings
from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
from tests.fixtures.dxf_builders import multi_drawing_sheet_with_frame_dxf


def _with_flag(monkeypatch, value: bool):
    base_settings = get_settings()
    monkeypatch.setattr(
        "backend.config.get_settings",
        lambda: base_settings.model_copy(update={"use_dxf_evidence_decision_engine": value}),
    )


def test_flag_off_by_default():
    assert get_settings().use_dxf_evidence_decision_engine is False


def test_flag_true_abstains_rather_than_ship_a_confidently_wrong_whole_sheet_fallback(monkeypatch, tmp_path):
    """`multi_drawing_sheet_with_frame_dxf` is a genuinely ambiguous fixture
    on ITS OWN structural merit (its own item-9 test,
    `test_item_9_fires_on_an_ordinary_multi_drawing_sheet_even_when_
    correct` in `test_dxf_extractor.py`, confirms this -- two independently
    plausible regions, no caption anywhere). The new engine lands in
    CONFLICT here (confirmed directly: region scores 5.593 vs 5.317, a
    0.276 margin, both above the usability floor).

    An earlier version of this wiring fell back to whole-sheet resolution
    in exactly this case and shipped `plot.width=90.0` -- the SHEET
    BORDER/FRAME polygon itself, since whole-sheet frame-rejection turned
    out not to be as robust as the region-scoped path's. This test pins
    the fix: CONFLICT/ABSTAIN must ship nothing (honest MISSING) rather
    than either the old winner-take-all region OR an untested fallback
    path's own confidently-wrong answer.
    """
    path = multi_drawing_sheet_with_frame_dxf(tmp_path / "multi.dxf")

    _with_flag(monkeypatch, True)
    result_flag_on = DXFHybridExtractor().extract(path, "multi")

    cv = result_flag_on.independent_cv
    assert cv is not None
    values = {m.field: (m.value_m if m.value_m is not None else m.value) for m in cv.measurements}
    # Specifically must NOT be 90.0 (the fixture's own sheet_w) or any other
    # frame-derived value -- honestly absent, not confidently wrong.
    assert values.get("plot.width") is None
    assert values.get("building.width") is None


def test_flag_true_on_real_plan5_dxf_does_not_regress_the_caption_confirmed_region(monkeypatch):
    """PLAN5.dxf is the real, already-diagnosed fixture `DXF_FAILURE_
    TAXONOMY.md` item 0 documents: region3 is the true site plan, scores
    WORST structurally (-4.967) among 12 regions, and wins ONLY because a
    recognized 'SITE PLAN' caption applies a +20 bonus. This is the one
    real case where the existing ad-hoc mechanism is already confirmed
    correct -- flipping the new engine on must not regress it: the shipped
    plot dimensions must be identical to the flag-off (default) result.
    """
    dxf_path = Path(__file__).resolve().parent.parent / "data" / "test_plans" / "PLAN5.dxf"
    if not dxf_path.exists():
        pytest.skip("PLAN5.dxf fixture not present in this checkout")

    result_flag_off = DXFHybridExtractor().extract(dxf_path, "PLAN5")
    _with_flag(monkeypatch, True)
    result_flag_on = DXFHybridExtractor().extract(dxf_path, "PLAN5")

    def _widths(result):
        cv = result.independent_cv
        assert cv is not None
        return {m.field: (m.value_m if m.value_m is not None else m.value) for m in cv.measurements}

    widths_off = _widths(result_flag_off)
    widths_on = _widths(result_flag_on)
    assert widths_on.get("plot.width") == pytest.approx(widths_off.get("plot.width"), abs=1e-6)
    assert widths_on.get("plot.depth") == pytest.approx(widths_off.get("plot.depth"), abs=1e-6)
