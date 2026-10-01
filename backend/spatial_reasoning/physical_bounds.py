"""Physical-impossibility gate on a finished NormalizedPlan.

Every extractor here can be wrong in ways that produce numbers no real plot
can have: a unit or scale failure gives a 1 m2 site, a mis-picked "building"
that is really the plot outline gives four setbacks of exactly 0, a text
reader that takes a digit from a row number gives 10 m2 of built-up area.
When such a value carries HIGH confidence the compliance check then reports
a confident violation or pass on a number that cannot be true. The project's
rule is that a wrong answer is worse than a missing one, so a value that is
physically impossible is withdrawn rather than passed on.

The bounds are impossibilities, not statistics: nothing here is fitted to a
data set, and each bound is set well outside anything a sanctioned plan
contains. A rule that fires means the drawing's geometry chain is broken,
and every field derived from the same chain is withdrawn with it. Each
withdrawal is recorded in `plan.metadata["physical_bounds"]` and in
`overall_confidence_note`, so nothing disappears silently.
"""
from __future__ import annotations

from typing import Optional

from backend.schemas.evidence import ValueField
from backend.schemas.normalized_plan import NormalizedPlan

PLOT_AREA_M2 = (12.0, 500_000.0)      # smallest habitable site .. 50 ha
SITE_SIDE_M = (1.5, 1_000.0)
BUILDING_SIDE_M = (1.0, 1_000.0)
FOOTPRINT_M2 = (4.0, 200_000.0)
COVERAGE_PCT = (0.0, 100.0)           # canonical unit is %, above 100 cannot be
FAR_MAX = 12.0
BUILDING_IS_PLOT_RATIO = 0.95         # footprint this close to the plot: same polygon
CLOSED_SETBACK_M = 0.3                # under a wall thickness: no gap at all
MIN_CLOSED_SIDES = 3                  # a building closed on 3+ sides has no setback to check

_GEOMETRY_CHAIN = ("plot.area", "plot.width", "plot.depth", "building.width", "building.depth",
                   "building.footprint_area", "setbacks.front", "setbacks.rear", "setbacks.left",
                   "setbacks.right", "coverage", "far")
_BUILDING_CHAIN = ("building.width", "building.depth", "building.footprint_area", "setbacks.front",
                   "setbacks.rear", "setbacks.left", "setbacks.right", "coverage", "far")
_COVERAGE_CHAIN = ("coverage", "far")


def _get(plan: NormalizedPlan, name: str) -> Optional[ValueField]:
    obj = plan
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _value(plan: NormalizedPlan, name: str) -> Optional[float]:
    vf = _get(plan, name)
    return None if vf is None or vf.value is None else float(vf.value)


def _outside(x: Optional[float], bounds: tuple[float, float]) -> bool:
    return x is not None and not (bounds[0] <= x <= bounds[1])


def _withdraw(plan: NormalizedPlan, names: tuple[str, ...], reason: str, log: list[dict]) -> None:
    for name in names:
        vf = _get(plan, name)
        if vf is None or vf.value is None:
            continue
        head, _, tail = name.partition(".")
        message = f"Withdrawn: {reason}"
        replacement = ValueField.missing(message)
        if tail:
            setattr(getattr(plan, head), tail, replacement)
        else:
            setattr(plan, head, replacement)
        log.append({"field": name, "value": vf.value, "was": vf.confidence.level.value, "reason": reason})


def enforce_physical_bounds(plan: NormalizedPlan) -> NormalizedPlan:
    log: list[dict] = []
    plot_area = _value(plan, "plot.area")
    footprint = _value(plan, "building.footprint_area")

    if _outside(plot_area, PLOT_AREA_M2):
        _withdraw(plan, _GEOMETRY_CHAIN,
                  f"plot area {plot_area:g} m2 is outside {PLOT_AREA_M2[0]:g}-{PLOT_AREA_M2[1]:g} m2, so the "
                  "drawing's units or plot polygon are wrong and everything measured from it is suspect", log)
    for name in ("plot.width", "plot.depth"):
        if _outside(_value(plan, name), SITE_SIDE_M):
            _withdraw(plan, (name,), f"{name} {_value(plan, name):g} m is outside {SITE_SIDE_M[0]:g}-{SITE_SIDE_M[1]:g} m", log)
    for name in ("building.width", "building.depth"):
        if _outside(_value(plan, name), BUILDING_SIDE_M):
            _withdraw(plan, (name,), f"{name} {_value(plan, name):g} m is outside {BUILDING_SIDE_M[0]:g}-{BUILDING_SIDE_M[1]:g} m", log)

    footprint = _value(plan, "building.footprint_area")
    plot_area = _value(plan, "plot.area")
    if _outside(footprint, FOOTPRINT_M2):
        _withdraw(plan, _BUILDING_CHAIN, f"building footprint {footprint:g} m2 is outside "
                  f"{FOOTPRINT_M2[0]:g}-{FOOTPRINT_M2[1]:g} m2", log)
    elif footprint is not None and plot_area and footprint >= BUILDING_IS_PLOT_RATIO * plot_area:
        _withdraw(plan, _GEOMETRY_CHAIN, f"building footprint {footprint:g} m2 is {BUILDING_IS_PLOT_RATIO:.0%} or more of the "
                  f"plot {plot_area:g} m2: they are the same polygon, so either could be the mis-identified one", log)

    setbacks = [_value(plan, f"setbacks.{s}") for s in ("front", "rear", "left", "right")]
    closed = sum(v is not None and v <= CLOSED_SETBACK_M for v in setbacks)
    if closed >= MIN_CLOSED_SIDES:
        _withdraw(plan, _BUILDING_CHAIN, f"{closed} of 4 setbacks are under {CLOSED_SETBACK_M:g} m (a wall thickness): the "
                  "building outline coincides with the plot boundary, not a building set back inside it", log)

    biggest_side = max([v for v in (_value(plan, "plot.width"), _value(plan, "plot.depth")) if v is not None], default=None)
    for side in ("front", "rear", "left", "right"):
        v = _value(plan, f"setbacks.{side}")
        if v is not None and (v < 0 or (biggest_side is not None and v > biggest_side)):
            _withdraw(plan, (f"setbacks.{side}",), f"setback {v:g} m is negative or longer than the plot itself", log)

    cov = _value(plan, "coverage")
    if _outside(cov, COVERAGE_PCT):
        _withdraw(plan, _COVERAGE_CHAIN, f"coverage {cov:g}% is outside 0-100%", log)
    far = _value(plan, "far")
    if far is not None and not (0 <= far <= FAR_MAX):
        _withdraw(plan, ("far",), f"FAR {far:g} is outside 0-{FAR_MAX:g}", log)

    if log:
        plan.metadata = {**plan.metadata, "physical_bounds": log}
        note = "Physical-bounds gate withdrew: " + "; ".join(sorted({e["field"] for e in log})) + "."
        plan.overall_confidence_note = f"{plan.overall_confidence_note} {note}".strip() if plan.overall_confidence_note else note
    return plan
