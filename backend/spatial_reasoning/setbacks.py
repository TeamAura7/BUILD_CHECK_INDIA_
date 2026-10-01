"""
Setback derivation — geometry-first, per phase3.md.
...
"""

from __future__ import annotations

from typing import Optional

from backend.config import get_settings
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Conflict, GeometryEvidence
from backend.schemas.evidence import ValueField
from backend.schemas.geometry import Polygon
from backend.schemas.normalized_plan import SetbackSection
from backend.schemas.units import CanonicalUnit, UnitValue
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.dimension_classification import ClassifiedDimension, DimensionSemanticType
from backend.spatial_reasoning.evidence_reconciliation import EvidenceCandidate, to_value_field
from backend.spatial_reasoning.front_side import FrontSideResolution

_SIDE_TO_SETBACK_TYPE = {
    "front": DimensionSemanticType.FRONT_SETBACK,
    "rear": DimensionSemanticType.REAR_SETBACK,
    "left": DimensionSemanticType.LEFT_SETBACK,
    "right": DimensionSemanticType.RIGHT_SETBACK,
}

# front/rear setbacks eat into plot.depth - building.depth;
# left/right setbacks eat into plot.width - building.width.
#
# This is NOT an assumption about absolute page-space horizontal/vertical
# orientation (a "wrong axis on a rotated plot" bug was considered here and
# deliberately rejected -- see git history) -- it is a direct consequence of
# how `plot.width`/`plot.depth`/`building.width`/`building.depth` are
# themselves DEFINED throughout this codebase. `front_side.py`'s own module
# docstring states the convention ("plot width = frontage ... plot depth =
# perpendicular dimension"), and `dimension_classification.py` classifies a
# dimension as PLOT_WIDTH/BUILDING_WIDTH precisely when it is parallel to the
# ALREADY-RESOLVED front edge (`frontage_deg`, itself derived from
# `frontage_edges[0]`'s own orientation) and PLOT_DEPTH/BUILDING_DEPTH when
# perpendicular to it -- never from absolute page-space horizontal/vertical
# alignment. So "width" already always means "the frontage-parallel extent"
# and "depth" already always means "the frontage-perpendicular extent",
# regardless of how the plot happens to be rotated on the page. Re-deriving
# this mapping from the front edge's own absolute angle (horizontal vs.
# vertical) would be wrong: it would swap "width"/"depth" specifically on a
# plot whose front edge is vertical in page space, even though `plot.width`
# there is still, by the classifier's own definition, the frontage length.
_SIDE_TO_AXIS_BUDGET = {
    "front": "depth",
    "rear": "depth",
    "left": "width",
    "right": "width",
}


def _axis_budget(
    axis: str,
    plot_width: Optional[ValueField[float]],
    plot_depth: Optional[ValueField[float]],
    building_width: Optional[ValueField[float]],
    building_depth: Optional[ValueField[float]],
) -> Optional[float]:
    """Total gap available on one axis (front+rear, or left+right), if both
    the plot and building extents on that axis are known."""
    plot_field = plot_depth if axis == "depth" else plot_width
    building_field = building_depth if axis == "depth" else building_width
    if plot_field is None or building_field is None:
        return None
    if plot_field.value is None or building_field.value is None:
        return None
    return plot_field.value - building_field.value


def axis_budget_cap(
    side: str,
    plot_width: Optional[ValueField[float]],
    plot_depth: Optional[ValueField[float]],
    building_width: Optional[ValueField[float]],
    building_depth: Optional[ValueField[float]],
    tol: Optional[float] = None,
) -> Optional[float]:
    """Public wrapper around the same front/rear-vs-depth,
    left/right-vs-width plausibility ceiling `compute_setbacks` enforces on
    its own geometry/dimension-label candidates below. Any OTHER source of a
    setback value for `side` (e.g. a vision-extracted or native-site-label
    reading assembled outside this module) must be checked against this same
    cap before it ships -- a value this project already knows cannot
    physically fit between the plot and building boundaries on this axis is
    just as wrong regardless of which extraction path produced it.

    Returns None when the budget itself is unknown (plot/building extent on
    this axis unresolved) -- there is nothing to check against in that case,
    not license to accept anything.
    """
    if tol is None:
        tol = get_settings().geometry_tolerance_m
    budget = _axis_budget(_SIDE_TO_AXIS_BUDGET[side], plot_width, plot_depth, building_width, building_depth)
    if budget is None:
        return None
    return budget + tol


def flag_if_setback_implausible(
    side: str,
    field: ValueField[float],
    field_name: str,
    plot_width: Optional[ValueField[float]],
    plot_depth: Optional[ValueField[float]],
    building_width: Optional[ValueField[float]],
    building_depth: Optional[ValueField[float]],
    tol: Optional[float] = None,
    *,
    frontage_relative_axes: bool = True,
) -> ValueField[float]:
    """Attach a `Conflict` to `field` when its value cannot physically fit in
    the `axis_budget_cap` ceiling for `side` -- WITHOUT ever touching
    `field.value` or `field.confidence`.

    This is deliberately non-destructive, unlike an earlier version of this
    check that replaced an implausible value with `ValueField.conflicting()`
    (value=None, confidence=CONFLICTING): that version caused real
    regressions -- it turned PLAN5's correct, ground-truthed
    setbacks.front=3.0m into None, purely because it was numerically
    inconsistent with that same PDF pipeline's own (otherwise HIGH-
    confidence) plot.depth/building.depth reading on the same plan.
    Punishing a good, individually-verified value for a real but SEPARATE
    inconsistency elsewhere in the plan's geometry is exactly the kind of
    silent, opaque "fix" ARCHITECTURE.md's uncertainty-first-class-value
    principle warns against, and
    `test_no_compared_field_ever_ships_conflicting_confidence_level`
    already encodes system-wide that a reconciliation-style disagreement
    must never flip a field's own `confidence.level` to CONFLICTING (that
    would make `backend/compliance/engine.py` refuse to use the value at
    all, regardless of whether `.value` itself is fine) -- see
    `pdf_dxf_reconciliation.py`'s own docstring for the precedent this
    follows instead: flag via `.conflict` only, ship the value unchanged.
    (This is the same real-world case `check_physical_consistency`'s own
    rectangular cross-check already flags in `plan.conflicts` -- PLAN5's
    plot.depth/building.depth pairing does not actually leave room for its
    own documented front+rear setbacks, a genuine inconsistency in the
    plan's OWN geometry, orthogonal to whether the setback reading itself
    is correct.)

    Only evaluates the budget when BOTH the plot and building dimension on
    this axis are resolved to at least MEDIUM confidence (excludes
    LOW/MISSING/CONFLICTING) -- i.e. only when the budget itself is
    resolved enough to be worth flagging against at all. Given the value
    itself is never altered, a false positive here only ever costs an
    extra, inspectable `.conflict` annotation -- never a wrong number.

    `frontage_relative_axes` says whether `plot_width`/`building_width` are
    already the frontage-PARALLEL extent (the classifier convention
    `_SIDE_TO_AXIS_BUDGET` relies on: front/rear eat into depth, left/right
    into width). Pass False for values that come from the sheet itself
    (independent CV / Vision / DXF), whose "width" is the drawing's horizontal
    extent: on a sheet whose road is on a SIDE, front/rear then run along the
    horizontal extent instead, and the fixed side-to-axis map wrongly flags
    correct values (confirmed on two real plans, a setback of 3.0 m on a
    2.07 m depth budget when its true axis had 4.5 m). Without knowing the
    orientation, a setback is only provably implausible if it fits on NEITHER
    axis, so the cap is the larger of the two axis budgets.
    """
    if field.value is None or field.confidence.level == ConfidenceLevel.CONFLICTING:
        return field
    _acceptable = (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)

    def _axis_resolved(plot_field, building_field) -> bool:
        return not (
            plot_field is None or building_field is None
            or plot_field.confidence.level not in _acceptable
            or building_field.confidence.level not in _acceptable
        )

    if frontage_relative_axes:
        plot_field = plot_depth if _SIDE_TO_AXIS_BUDGET[side] == "depth" else plot_width
        building_field = building_depth if _SIDE_TO_AXIS_BUDGET[side] == "depth" else building_width
        if not _axis_resolved(plot_field, building_field):
            return field
        cap = axis_budget_cap(side, plot_width, plot_depth, building_width, building_depth, tol=tol)
    else:
        if not (_axis_resolved(plot_width, building_width) and _axis_resolved(plot_depth, building_depth)):
            return field
        budgets = [
            _axis_budget(axis, plot_width, plot_depth, building_width, building_depth)
            for axis in ("width", "depth")
        ]
        if any(b is None for b in budgets):
            return field
        cap = max(budgets) + (get_settings().geometry_tolerance_m if tol is None else tol)
    if cap is None or field.value <= cap:
        return field
    return field.model_copy(update={
        "conflict": Conflict(
            description=(
                f"{field_name}={field.value:.3f} m exceeds what's physically possible given the "
                f"resolved plot/building dimensions on this side (max {cap:.3f} m) -- the building "
                "and this setback cannot both fit within the plot on this axis. Value shipped "
                "unchanged (flagged, not silently altered); see this field's own `.conflict` for detail."
            ),
            conflicting_raw_values=[UnitValue(magnitude=round(field.value, 4), unit=CanonicalUnit.METRE.value)],
            conflicting_sources=[field_name],
        ),
    })


def compute_setbacks(
    building_polygon_metric: Optional[Polygon],
    plot_edges_metric: FrontSideResolution,
    classified_dimensions: list[ClassifiedDimension],
    points_per_metre: float,
    plot_width: Optional[ValueField[float]] = None,
    plot_depth: Optional[ValueField[float]] = None,
    building_width: Optional[ValueField[float]] = None,
    building_depth: Optional[ValueField[float]] = None,
) -> SetbackSection:
    side_edge_groups = {
        "front": plot_edges_metric.front_edges,
        "rear": plot_edges_metric.rear_edges,
        "left": plot_edges_metric.left_edges,
        "right": plot_edges_metric.right_edges,
    }

    tol = get_settings().geometry_tolerance_m
    fields = {}
    rejected_notes: dict[str, list[str]] = {}

    for side, edges in side_edge_groups.items():
        candidates: list[EvidenceCandidate] = []
        rejected_notes[side] = []

        budget = _axis_budget(_SIDE_TO_AXIS_BUDGET[side], plot_width, plot_depth, building_width, building_depth)
        # A little slack: two setbacks split a budget unevenly (e.g. 0 / 0.47),
        # but neither one can ever exceed the *whole* budget for that axis.
        budget_cap = budget + tol if budget is not None else None

        if building_polygon_metric is not None and edges:
            dist = geo.polygon_to_edges_distance(building_polygon_metric, edges)
            if dist != float("inf"):
                if budget_cap is not None and dist > budget_cap:
                    rejected_notes[side].append(
                        f"geometry distance {dist:.3f} m exceeds the {_SIDE_TO_AXIS_BUDGET[side]}-axis "
                        f"budget ({budget_cap:.3f} m) — discarded as implausible"
                    )
                else:
                    candidates.append(
                        EvidenceCandidate(
                            value=round(dist, 4),
                            source=f"geometry (building-to-{side}-boundary distance)",
                            evidence=GeometryEvidence(
                                description=f"Metric-space distance from building footprint to the {side} plot edge."
                            ),
                        )
                    )

        setback_type = _SIDE_TO_SETBACK_TYPE[side]
        for cd in classified_dimensions:
            if cd.semantic_type is setback_type and cd.value_metres is not None:
                if budget_cap is not None and cd.value_metres > budget_cap:
                    rejected_notes[side].append(
                        f"dimension label ('{cd.dimension.label}'={cd.value_metres:.3f} m) exceeds the "
                        f"{_SIDE_TO_AXIS_BUDGET[side]}-axis budget ({budget_cap:.3f} m) — almost certainly a "
                        f"misclassified plot/building dimension rather than a real setback — discarded"
                    )
                    continue
                candidates.append(
                    EvidenceCandidate(
                        value=round(cd.value_metres, 4),
                        source=f"dimension label ('{cd.dimension.label}')",
                        weight=cd.confidence,
                    )
                )

        missing_reason = f"No geometry-derived distance or classified dimension found for the {side} setback."
        if rejected_notes[side]:
            missing_reason += " Rejected candidate(s): " + "; ".join(rejected_notes[side])

        fields[side] = to_value_field(
            candidates,
            field_label=f"setbacks.{side}",
            missing_reason=missing_reason,
        )

    return SetbackSection(front=fields["front"], rear=fields["rear"], left=fields["left"], right=fields["right"])


__all__ = ["compute_setbacks", "axis_budget_cap", "flag_if_setback_implausible"]