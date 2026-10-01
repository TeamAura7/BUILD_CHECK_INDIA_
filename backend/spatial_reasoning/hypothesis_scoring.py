"""
Format-agnostic structural-hypothesis scoring (Architecture V2, Phase 4/5).

See ARCHITECTURE_V2.md, Deliverable C.2 item 4 and Deliverable E's migration
table entry for `dxf_extractor.py`. This module is a faithful EXTRACTION of
`backend.cv_extraction.dxf_extractor._score_region_resolution`'s scoring
arithmetic -- the same numbers, in the same order, producing the same total
-- decomposed into named, auditable `ConstraintResult` terms instead of one
opaque float, and expressed in terms of plain structural facts (areas,
ratios, booleans) rather than DXF's own private `_RawPoly`/`DxfRegion`
types, so the identical scoring logic can eventually be reused by the PDF
path (`plot_resolution.py`) without a second, independently-drifting copy.

`dxf_extractor._score_region_resolution` now delegates to
`score_plot_building_resolution` below and sums its `ConstraintResult`
scores -- this is a behavior-preserving refactor (move, not rewrite): the
existing `test_dxf_extractor.py`/`test_real_plan_regression.py` suites are
the regression floor proving the extraction changed nothing observable.
"""

from __future__ import annotations

import math
from typing import Optional

from backend.schemas.hypothesis import ConstraintResult

# A region's resolution is worthless if it didn't find a plot at all --
# mirrors dxf_extractor._score_region_resolution's early return of -1.0.
_NO_PLOT_SCORE = -1.0


def score_plot_building_resolution(
    *,
    plot_area: Optional[float],
    plausible_plot_area_bounds: tuple[float, float],
    plot_is_envelope_reconstruction: bool,
    plot_aspect_ratio: float,
    building_present: bool,
    building_plot_area_ratio: Optional[float],
    min_building_plot_area_fraction: float,
    building_is_reconstruction_or_vision: bool,
) -> tuple[list[ConstraintResult], float]:
    """
    Score one candidate's resolved plot (+ optional building) pair.

    Every argument is a plain structural fact the caller has already
    computed from its own geometry -- this function makes no assumption
    about DXF vs. PDF, native vs. reconstructed geometry beyond what the
    caller tells it via `plot_is_envelope_reconstruction`/
    `building_is_reconstruction_or_vision`. `plausible_plot_area_bounds`
    and `min_building_plot_area_fraction` are passed in explicitly, not
    hardcoded here, so a caller's own plan-plausibility constants remain
    that caller's responsibility (no shared "the" plausible plot area).

    Returns `(constraint_results, total_score)` where
    `total_score == sum(c.score for c in constraint_results)`, always.
    """
    if plot_area is None:
        return (
            [
                ConstraintResult(
                    name="plot_presence", score=_NO_PLOT_SCORE, passed=False,
                    detail="no plot resolved for this candidate",
                )
            ],
            _NO_PLOT_SCORE,
        )

    results: list[ConstraintResult] = [
        ConstraintResult(
            name="base", score=1.0, passed=True,
            detail="base score for having a resolved plot",
        )
    ]

    lo, hi = plausible_plot_area_bounds
    area_plausible = lo <= plot_area <= hi
    results.append(
        ConstraintResult(
            name="plot_area_plausibility",
            score=0.0 if area_plausible else -10.0,
            passed=area_plausible,
            detail=(
                f"plot area {plot_area:.3f} within plausible bounds [{lo}, {hi}]"
                if area_plausible
                else f"plot area {plot_area:.3f} outside plausible bounds [{lo}, {hi}]"
            ),
        )
    )

    results.append(
        ConstraintResult(
            name="envelope_reconstruction_penalty",
            score=-0.5 if plot_is_envelope_reconstruction else 0.0,
            passed=not plot_is_envelope_reconstruction,
            detail=(
                "plot geometry is a fitted envelope around a scattered point cloud, "
                "not a real closed boundary"
                if plot_is_envelope_reconstruction
                else "plot geometry is a real closed boundary or reconstruction, not a fitted envelope"
            ),
        )
    )

    if building_present:
        results.append(
            ConstraintResult(name="building_presence", score=2.0, passed=True, detail="a building was also resolved")
        )
        ratio_plausible = (
            building_plot_area_ratio is not None
            and min_building_plot_area_fraction <= building_plot_area_ratio <= 0.92
        )
        results.append(
            ConstraintResult(
                name="building_plot_ratio_plausibility",
                score=1.0 if ratio_plausible else 0.0,
                passed=ratio_plausible,
                detail=(
                    f"building/plot area ratio {building_plot_area_ratio:.3f} within "
                    f"[{min_building_plot_area_fraction}, 0.92]"
                    if building_plot_area_ratio is not None
                    else "plot area was non-positive, ratio not computed"
                ),
            )
        )
        results.append(
            ConstraintResult(
                name="building_polygon_quality",
                score=0.0 if building_is_reconstruction_or_vision else 1.0,
                passed=not building_is_reconstruction_or_vision,
                detail=(
                    "building geometry is a reconstruction/vision-fallback guess, not a genuinely closed polygon"
                    if building_is_reconstruction_or_vision
                    else "building geometry is a genuinely closed polygon, not a reconstruction guess"
                ),
            )
        )

    aspect_ok = math.isfinite(plot_aspect_ratio) and plot_aspect_ratio <= 6.0
    results.append(
        ConstraintResult(
            name="aspect_ratio_plausibility",
            score=0.5 if aspect_ok else 0.0,
            passed=aspect_ok,
            detail=f"plot aspect ratio {plot_aspect_ratio:.2f}" + (" <= 6.0" if aspect_ok else " > 6.0 (or non-finite)"),
        )
    )

    area_tiebreak = min(1.0, plot_area / 500.0)
    results.append(
        ConstraintResult(
            name="area_tiebreak", score=area_tiebreak, passed=True,
            detail=f"weak preference for larger absolute plot area ({plot_area:.1f} m^2)",
        )
    )

    total = sum(c.score for c in results)
    return results, total


__all__ = ["score_plot_building_resolution"]
