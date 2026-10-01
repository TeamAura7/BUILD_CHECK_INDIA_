"""
Architecture V2, Phase 4/5 -- unit tests for
`backend.spatial_reasoning.hypothesis_scoring.score_plot_building_resolution`,
the format-agnostic extraction of
`backend.cv_extraction.dxf_extractor._score_region_resolution`'s scoring
arithmetic (see ARCHITECTURE_V2.md Deliverable C.2 item 4).

These pin the exact arithmetic in isolation, independent of DXF-specific
types -- `test_dxf_extractor.py`'s real-fixture tests are the integration-
level regression floor proving `_score_region_resolution` itself (which now
delegates here) still produces identical numbers end to end.
"""

from __future__ import annotations

import math

from backend.spatial_reasoning.hypothesis_scoring import score_plot_building_resolution

_BOUNDS = (20.0, 20000.0)
_MIN_BUILDING_FRACTION = 0.03


def _score(**overrides) -> tuple[list, float]:
    defaults = dict(
        plot_area=100.0,
        plausible_plot_area_bounds=_BOUNDS,
        plot_is_envelope_reconstruction=False,
        plot_aspect_ratio=1.5,
        building_present=False,
        building_plot_area_ratio=None,
        min_building_plot_area_fraction=_MIN_BUILDING_FRACTION,
        building_is_reconstruction_or_vision=False,
    )
    defaults.update(overrides)
    return score_plot_building_resolution(**defaults)


def test_no_plot_scores_minus_one_with_a_named_term():
    results, total = score_plot_building_resolution(
        plot_area=None,
        plausible_plot_area_bounds=_BOUNDS,
        plot_is_envelope_reconstruction=False,
        plot_aspect_ratio=1.0,
        building_present=False,
        building_plot_area_ratio=None,
        min_building_plot_area_fraction=_MIN_BUILDING_FRACTION,
        building_is_reconstruction_or_vision=False,
    )
    assert total == -1.0
    assert len(results) == 1
    assert results[0].name == "plot_presence"
    assert results[0].passed is False


def test_total_score_is_always_the_sum_of_named_terms():
    results, total = _score(building_present=True, building_plot_area_ratio=0.2)
    assert total == sum(c.score for c in results)


def test_plausible_plot_area_within_bounds_contributes_zero_not_penalty():
    results, total = _score(plot_area=100.0)
    area_term = next(c for c in results if c.name == "plot_area_plausibility")
    assert area_term.passed is True
    assert area_term.score == 0.0


def test_implausible_plot_area_is_penalized_ten_points():
    results, total = _score(plot_area=5.0)  # below the 20.0 lower bound
    area_term = next(c for c in results if c.name == "plot_area_plausibility")
    assert area_term.passed is False
    assert area_term.score == -10.0


def test_envelope_reconstruction_costs_half_a_point():
    plain_results, plain_total = _score(plot_is_envelope_reconstruction=False)
    envelope_results, envelope_total = _score(plot_is_envelope_reconstruction=True)
    assert plain_total - envelope_total == 0.5


def test_building_presence_adds_two_points_plus_bonuses():
    no_building_results, no_building_total = _score(building_present=False)
    with_building_results, with_building_total = _score(
        building_present=True, building_plot_area_ratio=0.2, building_is_reconstruction_or_vision=False,
    )
    # +2.0 presence, +1.0 plausible ratio, +1.0 genuinely-closed polygon = +4.0
    assert with_building_total - no_building_total == 4.0


def test_building_ratio_outside_plausible_range_does_not_get_the_bonus():
    results, _total = _score(building_present=True, building_plot_area_ratio=0.99)
    ratio_term = next(c for c in results if c.name == "building_plot_ratio_plausibility")
    assert ratio_term.passed is False
    assert ratio_term.score == 0.0


def test_reconstruction_or_vision_building_does_not_get_the_polygon_quality_bonus():
    results, _total = _score(
        building_present=True, building_plot_area_ratio=0.2, building_is_reconstruction_or_vision=True,
    )
    quality_term = next(c for c in results if c.name == "building_polygon_quality")
    assert quality_term.passed is False
    assert quality_term.score == 0.0


def test_aspect_ratio_at_or_below_six_gets_the_bonus():
    ok_results, ok_total = _score(plot_aspect_ratio=6.0)
    bad_results, bad_total = _score(plot_aspect_ratio=6.01)
    assert ok_total - bad_total == 0.5


def test_non_finite_aspect_ratio_does_not_get_the_bonus():
    results, _total = _score(plot_aspect_ratio=math.inf)
    aspect_term = next(c for c in results if c.name == "aspect_ratio_plausibility")
    assert aspect_term.passed is False


def test_area_tiebreak_saturates_at_one_point():
    _results, small_total = _score(plot_area=100.0)  # 100/500 = 0.2
    _results, large_total = _score(plot_area=5000.0)  # saturates at 1.0
    tiebreak_small = next(c for c in _score(plot_area=100.0)[0] if c.name == "area_tiebreak")
    tiebreak_large = next(c for c in _score(plot_area=5000.0)[0] if c.name == "area_tiebreak")
    assert tiebreak_small.score == 100.0 / 500.0
    assert tiebreak_large.score == 1.0


def test_matches_the_original_dxf_extractor_arithmetic_end_to_end():
    """
    Literal replication of `_score_region_resolution`'s own arithmetic for
    a representative case (plot area 300, envelope reconstruction, a
    building present with a plausible ratio and a genuinely closed
    polygon, aspect ratio 3.0) -- computed by hand from the original
    function's source to confirm the extraction is exact, not just
    "reasonable".

    score = 1.0 (base)
            + 0.0 (area within [20, 20000])
            - 0.5 (envelope reconstruction)
            + 2.0 (building present)
            + 1.0 (ratio 0.2 within [0.03, 0.92])
            + 1.0 (genuinely closed building polygon)
            + 0.5 (aspect ratio 3.0 <= 6.0)
            + min(1.0, 300/500) = 0.6 (area tiebreak)
            = 5.6
    """
    _results, total = _score(
        plot_area=300.0,
        plot_is_envelope_reconstruction=True,
        plot_aspect_ratio=3.0,
        building_present=True,
        building_plot_area_ratio=0.2,
        building_is_reconstruction_or_vision=False,
    )
    assert total == 5.6
