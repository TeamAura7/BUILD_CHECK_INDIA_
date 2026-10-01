"""
Area / coverage / FAR computation.

Keeps distinct area concepts separate rather than silently substituting
one for another (room/carpet/built-up/footprint areas). The
`NormalizedPlan` contract only has room for `building.footprint_area`
(the FAR-relevant plinth/footprint area used for coverage), so any other
labeled areas found in the source text (carpet area, built-up area,
etc.) are surfaced via `label_notes` for the caller to fold into
`overall_confidence_note` — never merged into `footprint_area` itself.

FAR here is always the *diagnostic* gross FAR defined in phase3.md:

    gross_FAR = gross_built_up_area / plot_area

never a regulatory/sanctioned FAR (that belongs to the RuleEngine, which
compares this diagnostic against a municipality's `RuntimeRuleDefinition`
threshold).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.schemas.enums import ConfidenceLevel, cap_confidence_level
from backend.schemas.evidence import Confidence, Conflict, GeometryEvidence, TextEvidence, ValueField
from backend.schemas.geometry import BoundingBox, Polygon
from backend.schemas.units import CanonicalUnit, UnitValue

# A building footprint can legitimately be very close to the plot area
# (near-100% coverage happens on real, tightly-built urban plots), but it
# can NEVER exceed it — footprint is physically contained within the
# plot. A small tolerance absorbs rounding/digitisation noise without
# masking a genuine resolution error (e.g. mismatched plot/building
# candidates from different drawing regions being paired together).
_MAX_PHYSICALLY_PLAUSIBLE_COVERAGE_RATIO = 1.02

_LABELED_AREA_RE = re.compile(
    r"\b(?P<label>carpet\s*area|built[\s-]?up\s*area|builtup\s*area|plinth\s*area|"
    r"floor\s*area)\b[^\d]{0,15}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>sq\.?\s?ft|sqft|sq\.?\s?m|sqm)?",
    re.I,
)


@dataclass
class LabeledArea:
    label: str
    value: float
    unit: Optional[str]
    source_text: str


def find_labeled_areas(text_evidence: list[TextEvidence]) -> list[LabeledArea]:
    """Detect explicitly labeled areas (carpet/built-up/plinth/floor) in plan text, kept separate."""
    found: list[LabeledArea] = []
    for t in text_evidence:
        for m in _LABELED_AREA_RE.finditer(t.raw_text or ""):
            found.append(
                LabeledArea(
                    label=re.sub(r"\s+", " ", m.group("label")).strip().lower(),
                    value=float(m.group("value")),
                    unit=(m.group("unit") or None),
                    source_text=t.raw_text,
                )
            )
    return found


# --- Printed "AREA STATEMENT" table (site area + per-floor gross/nett areas) ----
#
# Virtually every Indian municipal sanction-plan sheet (BBMP and equivalent)
# carries a mandatory tabular "AREA STATEMENT": a "Site Area" figure plus one
# row per floor (Stilt/Ground/First/.../Terrace) giving that floor's Gross,
# Deduction, and Nett area in Sq.mts/Sq.ft. This is the *authoritative*,
# human-verified figure for the site and for each floor's built extent —
# far more reliable than trying to re-derive plot/building area from CV
# rectangle detection on a page that typically also contains several other,
# unrelated drawings (floor plans, elevations, structural details) sharing
# the same sheet.
#
# The table's cells arrive as separate `TextEvidence` spans (one per label,
# one per number) rather than one block of prose, and — because the sheet's
# columns don't always extract in reading order — a label and its own number
# are not guaranteed to be text-adjacent. So this pairs each row label with
# the nearest same-row (similar y, to its right) numeric span by bounding-box
# position, the same proximity approach already used elsewhere in this
# package (e.g. `_nearby_room_label` in building_resolution.py) rather than
# a purely textual regex. This is driven entirely by the label words below
# and the sheet's own geometry — nothing here is specific to any one plan.
_SITE_AREA_LABEL_RE = re.compile(r"\bsite\s*area\b|\bplot\s*area\b", re.I)
_FLOOR_AREA_LABEL_RE = re.compile(
    r"\b(?P<floor>stilt|ground|first|second|third|fourth|fifth|sixth|terrace|basement)\s*floor\b",
    re.I,
)
_AREA_NUMBER_RE = re.compile(
    r"(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>sq\.?\s?m(?:ts?)?\.?|sqm|sq\.?\s?ft\.?|sqft)\b", re.I
)


def _row_number_candidates(
    label_bbox: BoundingBox, numeric_items: list[tuple[float, float, BoundingBox]]
) -> list[tuple[float, str]]:
    """Numbers (value, unit) whose span sits on the same table row as label_bbox, left-to-right."""
    row_tol = max(label_bbox.height, 8.0) * 1.5
    row_y = label_bbox.center.y
    same_row = [
        (val, unit, bbox)
        for (val, unit, bbox) in numeric_items
        if abs(bbox.center.y - row_y) <= row_tol and bbox.min_x >= label_bbox.min_x - row_tol
    ]
    same_row.sort(key=lambda item: item[2].min_x)
    return [(val, unit) for (val, unit, _bbox) in same_row]


def _area_value_to_sqm(value: float, unit: Optional[str]) -> float:
    from backend.schemas.units import AreaUnit, area_to_sqm

    if unit and re.search(r"ft", unit, re.I):
        return area_to_sqm(value, AreaUnit.SQ_FT)
    return value


def find_area_statement(text_evidence: list[TextEvidence]) -> dict[str, LabeledArea]:
    """
    Parse the sheet's own printed Area Statement table, keyed by
    "site_area" and "<floor>_floor" (e.g. "ground_floor", "stilt_floor").
    For a floor row with multiple numbers (Gross, Deduction, Nett), the
    left-most (Gross/first-printed) figure is kept, matching how these
    tables are conventionally laid out — the physically-constructed
    footprint before any FSI deduction. Returns {} if the sheet has no
    such table (e.g. it wasn't printed, or text extraction found nothing
    resembling it) — callers must treat that as "no override available",
    not as an error.
    """
    numeric_items: list[tuple[float, float, BoundingBox]] = []
    for t in text_evidence:
        if t.bounding_box is None:
            continue
        m = _AREA_NUMBER_RE.search(t.raw_text or "")
        if m:
            numeric_items.append((float(m.group("value")), m.group("unit"), t.bounding_box))

    result: dict[str, LabeledArea] = {}
    for t in text_evidence:
        if t.bounding_box is None:
            continue
        text = t.raw_text or ""
        if _SITE_AREA_LABEL_RE.search(text) and "site_area" not in result:
            row = _row_number_candidates(t.bounding_box, numeric_items)
            if row:
                value, unit = row[0]
                result["site_area"] = LabeledArea(
                    label="site area", value=_area_value_to_sqm(value, unit), unit="sq_m", source_text=text
                )
            continue
        m = _FLOOR_AREA_LABEL_RE.search(text)
        if m:
            key = f"{m.group('floor').lower()}_floor"
            if key in result:
                continue
            row = _row_number_candidates(t.bounding_box, numeric_items)
            if row:
                value, unit = row[0]
                result[key] = LabeledArea(
                    label=key.replace("_", " "), value=_area_value_to_sqm(value, unit), unit="sq_m", source_text=text
                )
    return result


def polygon_area_field(
    polygon_metric: Optional[Polygon],
    width_field: Optional[ValueField[float]],
    depth_field: Optional[ValueField[float]],
    rectangularity_score: float,
    label: str,
) -> ValueField[float]:
    """
    Rectangular plot: area = width * depth (once both are known, HIGH confidence).
    Irregular plot: area = polygon_area(polygon) via the shoelace formula.
    Both are preserved where available: if width/depth are known AND the
    shape is clearly rectangular, prefer width*depth (matches how the
    dimensions themselves were read); otherwise use the geometric polygon
    area directly.
    """
    geom_area = polygon_metric.area if polygon_metric is not None else None

    if (
        rectangularity_score >= 0.9
        and width_field is not None
        and depth_field is not None
        and width_field.value is not None
        and depth_field.value is not None
    ):
        value = width_field.value * depth_field.value
        note = f"Rectangular plot: {label} = width * depth ({width_field.value:.3f} x {depth_field.value:.3f})."
        if geom_area is not None and geom_area > 0:
            rel_diff = abs(value - geom_area) / geom_area
            note += f" Geometry-derived polygon area is {geom_area:.3f} (relative diff {rel_diff:.1%})."
        return ValueField[float](
            value=round(value, 4),
            normalized_value=UnitValue(magnitude=round(value, 4), unit=CanonicalUnit.SQUARE_METRE.value),
            confidence=Confidence(level=ConfidenceLevel.HIGH, reason=note),
            source=f"{label} (width x depth)",
        )

    if geom_area is not None and geom_area > 0:
        return ValueField[float](
            value=round(geom_area, 4),
            normalized_value=UnitValue(magnitude=round(geom_area, 4), unit=CanonicalUnit.SQUARE_METRE.value),
            confidence=Confidence(
                level=ConfidenceLevel.MEDIUM,
                reason=f"Irregular/unclear-rectangularity {label}; using polygon_area(shoelace) directly.",
            ),
            source=f"{label} (polygon geometry)",
        )

    return ValueField[float].missing(f"No geometry or width/depth pair available to compute {label}.")


def coverage_field(footprint_area: ValueField[float], plot_area: ValueField[float]) -> ValueField[float]:
    if footprint_area.value is None or plot_area.value is None or plot_area.value <= 0:
        return ValueField[float].missing(
            "Coverage requires both building.footprint_area and plot.area to be resolved."
        )

    ratio = footprint_area.value / plot_area.value
    if ratio > _MAX_PHYSICALLY_PLAUSIBLE_COVERAGE_RATIO:
        # FIX #9 (phase3.1): footprint > plot area is not "unusually high
        # coverage", it is physically impossible — almost always a sign
        # that the resolved plot and building candidates are from
        # mismatched/incorrect geometry (e.g. a detail-view rectangle
        # paired with a site-scale plot). Surface this as an explicit
        # conflict rather than a nonsense percentage the caller has to
        # notice is wrong on their own.
        return ValueField[float].conflicting(
            Conflict(
                description=(
                    f"Physically impossible coverage: building.footprint_area "
                    f"({footprint_area.value:.3f}) exceeds plot.area ({plot_area.value:.3f}) "
                    f"by a factor of {ratio:.2f}x. This indicates the resolved plot and building "
                    "candidates likely do not both refer to the same real-world footprint "
                    "(e.g. mismatched drawing regions/scale) rather than genuinely high site "
                    "coverage."
                ),
                conflicting_raw_values=[
                    UnitValue(magnitude=footprint_area.value, unit=CanonicalUnit.SQUARE_METRE.value),
                    UnitValue(magnitude=plot_area.value, unit=CanonicalUnit.SQUARE_METRE.value),
                ],
                conflicting_sources=["building.footprint_area", "plot.area"],
            )
        )

    pct = ratio * 100.0
    # The ratio can only be as trustworthy as its WEAKER input -- a binary
    # "HIGH only if both HIGH, else MEDIUM" (the previous rule here) still
    # ships MEDIUM when one input is genuinely LOW (e.g. a plot/building
    # candidate `pipeline._cap_field_confidence`/`final_fusion.
    # _reconcile_area_field` already correctly capped to LOW), silently
    # laundering that LOW back up to a confidence compliance.engine's own
    # LOW-only REQUIRES_REVIEW gate does not catch. Reuse `cap_confidence_
    # level` (the same fix already applied to `_reconcile_area_field`) so a
    # LOW input actually propagates.
    level = cap_confidence_level(footprint_area.confidence.level, plot_area.confidence.level)
    return ValueField[float](
        value=pct,  # full precision preserved internally, not rounded
        normalized_value=UnitValue(magnitude=pct, unit=CanonicalUnit.PERCENTAGE.value),
        confidence=Confidence(
            level=level,
            reason=f"coverage = footprint_area({footprint_area.value:.3f}) / plot_area({plot_area.value:.3f}) * 100",
        ),
        source="coverage = building.footprint_area / plot.area * 100",
    )


def far_field(
    footprint_area: ValueField[float],
    plot_area: ValueField[float],
    floor_count: Optional[ValueField[int]],
) -> ValueField[float]:
    if footprint_area.value is None or plot_area.value is None or plot_area.value <= 0:
        return ValueField[float].missing("FAR requires both building.footprint_area and plot.area to be resolved.")

    if footprint_area.value / plot_area.value > _MAX_PHYSICALLY_PLAUSIBLE_COVERAGE_RATIO:
        # Same hard physical boundary as coverage_field: FAR is built on
        # the same footprint/plot relationship, so a physically
        # impossible footprint invalidates FAR too rather than producing
        # an inflated-but-plausible-looking ratio.
        return ValueField[float].conflicting(
            Conflict(
                description=(
                    f"FAR is not computable: building.footprint_area ({footprint_area.value:.3f}) "
                    f"exceeds plot.area ({plot_area.value:.3f}), which is physically impossible — "
                    "see building.footprint_area/coverage conflict for detail."
                ),
                conflicting_raw_values=[
                    UnitValue(magnitude=footprint_area.value, unit=CanonicalUnit.SQUARE_METRE.value),
                    UnitValue(magnitude=plot_area.value, unit=CanonicalUnit.SQUARE_METRE.value),
                ],
                conflicting_sources=["building.footprint_area", "plot.area"],
            )
        )

    if floor_count is not None and floor_count.value is not None and floor_count.value > 0:
        gross_built_up = footprint_area.value * floor_count.value
        note = (
            f"Diagnostic gross FAR = gross_built_up_area({gross_built_up:.3f}, "
            f"footprint x {floor_count.value} floors) / plot_area({plot_area.value:.3f}). "
            "NOT a regulatory/sanctioned FAR — that comparison belongs to the RuleEngine."
        )
        # FAR here is footprint x floors / plot, so it is only as trustworthy
        # as the floor count it multiplies by. Without this cap a LOW floor
        # count (e.g. read from a single caption form) shipped a MEDIUM FAR
        # -- on a real 5-storey sheet, 3.06 at MEDIUM against a true value
        # nothing near it -- i.e. a weak input laundered into a confident
        # output.
        level = cap_confidence_level(ConfidenceLevel.MEDIUM, floor_count.confidence.level)
        if level != ConfidenceLevel.MEDIUM:
            note += f" (confidence capped: the floor count is itself {floor_count.confidence.level.value})"
    else:
        gross_built_up = footprint_area.value
        note = (
            f"Diagnostic gross FAR = gross_built_up_area({gross_built_up:.3f}, footprint area, "
            "floor count NOT resolved so a single storey is assumed) / "
            f"plot_area({plot_area.value:.3f}). Treat as a lower bound. NOT a regulatory/sanctioned FAR."
        )
        level = ConfidenceLevel.LOW

    # Same reasoning as `coverage_field`: the diagnostic-FAR cap above is a
    # ceiling (this number is never "regulatory-grade", so never HIGH), not
    # a floor -- a genuinely LOW-confidence footprint/plot area (e.g.
    # PLAN7's capped candidates) must still pull FAR down to LOW, not get
    # silently rounded up to this branch's MEDIUM/LOW default.
    level = cap_confidence_level(level, cap_confidence_level(footprint_area.confidence.level, plot_area.confidence.level))

    ratio = gross_built_up / plot_area.value
    return ValueField[float](
        value=ratio,
        normalized_value=UnitValue(magnitude=ratio, unit=CanonicalUnit.RATIO.value),
        confidence=Confidence(level=level, reason=note),
        source="far = gross_built_up_area / plot.area (diagnostic gross FAR)",
    )


__all__ = [
    "LabeledArea",
    "find_labeled_areas",
    "find_area_statement",
    "polygon_area_field",
    "coverage_field",
    "far_field",
]
