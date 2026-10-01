"""
Physical consistency checks.

Run once the plot/building/setback ValueFields are resolved. Never
silently "fixes" impossible geometry — it flags it as a `Conflict` on
the `NormalizedPlan` so a human reviewer (or the future RuleEngine) sees
it, per ARCHITECTURE.md's "uncertainty is a first-class value" principle.
"""

from __future__ import annotations

from backend.config import get_settings
from backend.schemas.evidence import Conflict, ValueField
from backend.schemas.normalized_plan import BuildingSection, PlotSection, SetbackSection
from backend.schemas.units import CanonicalUnit, UnitValue
from backend.spatial_reasoning import geometry_utils as geo

# Below this rectangularity score (polygon_area / bbox_area), a plot is
# treated as irregular and the rectangular width/depth cross-check
# (building + left + right ~= plot width, etc.) is skipped — per
# phase3.md, that validation is only meaningful "for rectangular plots".
_RECTANGULARITY_THRESHOLD = 0.9


def _is_rectangular(plot: PlotSection) -> bool:
    if plot.geometry is None or plot.geometry.polygon is None:
        # No polygon evidence at all: don't assume rectangular, but don't
        # block the check either — fall back to trusting width/depth pair.
        return True
    return geo.rectangularity(plot.geometry.polygon) >= _RECTANGULARITY_THRESHOLD


def check_physical_consistency(
    plot: PlotSection, building: BuildingSection, setbacks: SetbackSection
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    tol = get_settings().geometry_tolerance_m
    rectangular = _is_rectangular(plot)

    def _both_known(a: ValueField[float], b: ValueField[float]) -> bool:
        return a.value is not None and b.value is not None

    if _both_known(plot.width, building.width) and building.width.value >= plot.width.value + tol:
        conflicts.append(
            Conflict(
                description=(
                    f"Impossible geometry: building.width ({building.width.value:.3f} m) >= "
                    f"plot.width ({plot.width.value:.3f} m)."
                ),
                conflicting_raw_values=[
                    UnitValue(magnitude=plot.width.value, unit=CanonicalUnit.METRE.value),
                    UnitValue(magnitude=building.width.value, unit=CanonicalUnit.METRE.value),
                ],
                conflicting_sources=["plot.width", "building.width"],
            )
        )

    if _both_known(plot.depth, building.depth) and building.depth.value >= plot.depth.value + tol:
        conflicts.append(
            Conflict(
                description=(
                    f"Impossible geometry: building.depth ({building.depth.value:.3f} m) >= "
                    f"plot.depth ({plot.depth.value:.3f} m)."
                ),
                conflicting_raw_values=[
                    UnitValue(magnitude=plot.depth.value, unit=CanonicalUnit.METRE.value),
                    UnitValue(magnitude=building.depth.value, unit=CanonicalUnit.METRE.value),
                ],
                conflicting_sources=["plot.depth", "building.depth"],
            )
        )

    for side in ("front", "rear", "left", "right"):
        field: ValueField[float] = getattr(setbacks, side)
        if field.value is not None and field.value < -tol:
            conflicts.append(
                Conflict(
                    description=f"Impossible geometry: setbacks.{side} is negative ({field.value:.3f} m) — "
                    "the building would extend outside the plot boundary on this side.",
                    conflicting_raw_values=[UnitValue(magnitude=field.value, unit=CanonicalUnit.METRE.value)],
                    conflicting_sources=[f"setbacks.{side}"],
                )
            )

    if (
        rectangular
        and building.width.value is not None
        and setbacks.left.value is not None
        and setbacks.right.value is not None
        and plot.width.value is not None
    ):
        predicted = building.width.value + setbacks.left.value + setbacks.right.value
        diff = abs(predicted - plot.width.value)
        allowed = max(tol, plot.width.value * 0.05)
        if diff > allowed:
            conflicts.append(
                Conflict(
                    description=(
                        f"Rectangular cross-check failed: building.width + left + right setback "
                        f"({predicted:.3f} m) does not match plot.width ({plot.width.value:.3f} m); "
                        f"difference {diff:.3f} m exceeds tolerance {allowed:.3f} m."
                    ),
                    conflicting_raw_values=[
                        UnitValue(magnitude=predicted, unit=CanonicalUnit.METRE.value),
                        UnitValue(magnitude=plot.width.value, unit=CanonicalUnit.METRE.value),
                    ],
                    conflicting_sources=["building.width+setbacks.left+setbacks.right", "plot.width"],
                )
            )

    if (
        rectangular
        and building.depth.value is not None
        and setbacks.front.value is not None
        and setbacks.rear.value is not None
        and plot.depth.value is not None
    ):
        predicted = building.depth.value + setbacks.front.value + setbacks.rear.value
        diff = abs(predicted - plot.depth.value)
        allowed = max(tol, plot.depth.value * 0.05)
        if diff > allowed:
            conflicts.append(
                Conflict(
                    description=(
                        f"Rectangular cross-check failed: building.depth + front + rear setback "
                        f"({predicted:.3f} m) does not match plot.depth ({plot.depth.value:.3f} m); "
                        f"difference {diff:.3f} m exceeds tolerance {allowed:.3f} m."
                    ),
                    conflicting_raw_values=[
                        UnitValue(magnitude=predicted, unit=CanonicalUnit.METRE.value),
                        UnitValue(magnitude=plot.depth.value, unit=CanonicalUnit.METRE.value),
                    ],
                    conflicting_sources=["building.depth+setbacks.front+setbacks.rear", "plot.depth"],
                )
            )

    return conflicts


__all__ = ["check_physical_consistency"]
