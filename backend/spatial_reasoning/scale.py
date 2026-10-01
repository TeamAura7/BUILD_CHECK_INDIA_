"""
Page-points -> metres scale estimation.

A `Dimension` (schemas/geometry.py) pairs a real-world magnitude+unit
(read from text) with an optional page-space `geometry` line (the
dimension line it was read off). Each such pair is one *scale sample*:

    points_per_metre_sample = dimension_line.length_pts / magnitude_in_metres

We collect every usable sample, per page, and reduce them with a robust
(median + MAD) estimator so a single mis-read digit or mis-associated
dimension line can't dominate the result. Confidence reflects both the
sample count and their spread. When no usable sample exists at all we
fall back to `settings.default_points_per_metre`, but at LOW confidence
and with an explicit note — never silently treated as authoritative.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.config import get_settings
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.geometry import Dimension

if TYPE_CHECKING:
    from backend.schemas.geometry import BoundingBox

# Robust outlier rejection threshold, in units of MAD (~3 is a common
# "obviously an outlier" cutoff for a robust z-score).
_MAD_OUTLIER_K = 3.5
_M_PER_FOOT = 0.3048
_MM_PER_M = 1000.0
_CM_PER_M = 100.0
_IN_PER_FT = 12.0

_LENGTH_UNITS_TO_METRES = {
    "m": lambda v: v,
    "mm": lambda v: v / _MM_PER_M,
    "cm": lambda v: v / _CM_PER_M,
    "ft": lambda v: v * _M_PER_FOOT,
    "in": lambda v: (v / _IN_PER_FT) * _M_PER_FOOT,
    # dimension_candidates.py stores ft_in dimensions as decimal feet already
    "ft_in": lambda v: v * _M_PER_FOOT,
}


def dimension_length_metres(dim: Dimension) -> float | None:
    """Convert a Dimension's magnitude to metres, or None if not a length unit."""
    conv = _LENGTH_UNITS_TO_METRES.get((dim.unit or "").strip().lower())
    if conv is None:
        return None
    try:
        value = conv(dim.magnitude)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


@dataclass
class ScaleSample:
    page: int
    points_per_metre: float
    dimension_label: str | None


@dataclass
class ScaleEstimate:
    points_per_metre: float
    confidence: ConfidenceLevel
    reason: str
    samples_used: int
    samples_rejected: int
    score: float | None = None
    outlier_samples: list[ScaleSample] = field(default_factory=list)


def collect_scale_samples(
    dimensions: list[Dimension], page: int | None = None, region_bbox: "BoundingBox | None" = None,
    region_padding_factor: float = 1.0,
) -> list[ScaleSample]:
    """
    `region_bbox`, when given, scopes samples to dimensions whose
    geometry line falls within `region_bbox` padded outward by
    `region_padding_factor` * the region's own diagonal (phase3.1 FIX
    #5/#6: "local" scale estimation, since `Dimension` carries no page
    attribute of its own — geometric proximity to the region actually
    being scaled is the only page-agnostic way to scope samples to it
    and avoid mixing a detail view's dimensions with a site-plan's).
    """
    samples: list[ScaleSample] = []
    padded = _pad_bbox(region_bbox, region_padding_factor) if region_bbox is not None else None
    for dim in dimensions:
        if dim.geometry is None:
            continue
        length_m = dimension_length_metres(dim)
        if length_m is None:
            continue
        line_len_pts = dim.geometry.length
        if line_len_pts <= 1e-6:
            continue
        if padded is not None and not _line_intersects_bbox(dim.geometry, padded):
            continue
        # Dimension has no page attribute of its own (it's page-agnostic
        # in schemas/geometry.py); callers pre-filter by page when needed.
        samples.append(
            ScaleSample(page=page if page is not None else -1, points_per_metre=line_len_pts / length_m, dimension_label=dim.label)
        )
    return samples


def _pad_bbox(bbox: "BoundingBox", padding_factor: float) -> "BoundingBox":
    from backend.schemas.geometry import BoundingBox

    diag = ((bbox.width) ** 2 + (bbox.height) ** 2) ** 0.5
    pad = diag * max(0.0, padding_factor)
    return BoundingBox(
        min_x=bbox.min_x - pad, min_y=bbox.min_y - pad, max_x=bbox.max_x + pad, max_y=bbox.max_y + pad
    )


def _line_intersects_bbox(line, bbox: "BoundingBox") -> bool:
    # Cheap, sufficient check: either endpoint falls inside the padded
    # region, or the line's midpoint does (covers lines that cross the
    # region without either endpoint being inside it).
    points = [line.start, line.end]
    mx, my = (line.start.x + line.end.x) / 2.0, (line.start.y + line.end.y) / 2.0
    for p in points:
        if bbox.min_x <= p.x <= bbox.max_x and bbox.min_y <= p.y <= bbox.max_y:
            return True
    return bbox.min_x <= mx <= bbox.max_x and bbox.min_y <= my <= bbox.max_y


def _robust_filter(samples: list[ScaleSample]) -> tuple[list[ScaleSample], list[ScaleSample]]:
    if len(samples) < 3:
        return samples, []
    values = [s.points_per_metre for s in samples]
    med = statistics.median(values)
    abs_devs = [abs(v - med) for v in values]
    mad = statistics.median(abs_devs)
    if mad < 1e-9:
        return samples, []
    kept, rejected = [], []
    for s in samples:
        robust_z = 0.6745 * abs(s.points_per_metre - med) / mad
        (kept if robust_z <= _MAD_OUTLIER_K else rejected).append(s)
    if not kept:  # never reject everything
        return samples, []
    return kept, rejected


def estimate_scale(
    dimensions: list[Dimension],
    page: int | None = None,
    region_bbox: "BoundingBox | None" = None,
    region_padding_factor: float = 1.5,
) -> ScaleEstimate:
    """
    Estimate points-per-metre from labelled dimensions with associated
    geometry lines. Falls back to the configured default at LOW
    confidence when no usable sample exists.

    When `region_bbox` is given (phase3.1 FIX #5/#6), scale samples are
    first scoped to that region (e.g. the resolved plot's own bounding
    box) so a detail-view dimension elsewhere on the sheet can't be
    averaged in with the site-scale dimensions actually relevant to it.
    If the local search finds too few samples to say anything (zero),
    this falls back to the whole-document sample set, but note says so
    explicitly rather than silently mixing scopes.
    """
    settings = get_settings()
    local_samples = (
        collect_scale_samples(dimensions, page=page, region_bbox=region_bbox, region_padding_factor=region_padding_factor)
        if region_bbox is not None
        else []
    )
    used_local = region_bbox is not None and len(local_samples) > 0
    samples = local_samples if used_local else collect_scale_samples(dimensions, page=page)
    kept, rejected = _robust_filter(samples)

    if not kept:
        return ScaleEstimate(
            points_per_metre=settings.default_points_per_metre,
            confidence=ConfidenceLevel.LOW,
            reason="No dimension carried both a resolvable length unit and an associated "
            "geometry line; falling back to the configured default page->metre scale.",
            samples_used=0,
            samples_rejected=len(rejected),
            outlier_samples=rejected,
        )

    values = [s.points_per_metre for s in kept]
    estimate = statistics.median(values)
    if len(values) >= 2:
        spread = statistics.pstdev(values) / estimate if estimate else 1.0
    else:
        spread = 0.0

    if len(kept) >= 3 and spread < 0.05:
        level, reason = ConfidenceLevel.HIGH, (
            f"{len(kept)} consistent scale samples (relative spread {spread:.1%})."
        )
    elif len(kept) >= 2 and spread < 0.15:
        level, reason = ConfidenceLevel.MEDIUM, (
            f"{len(kept)} scale samples with moderate spread ({spread:.1%})."
        )
    elif len(kept) == 1:
        level, reason = ConfidenceLevel.LOW, "Only a single scale sample available."
    else:
        level, reason = ConfidenceLevel.LOW, (
            f"{len(kept)} scale samples but high spread ({spread:.1%}); treat with caution."
        )

    if used_local:
        reason = f"[local region-scoped] {reason}"
    elif region_bbox is not None:
        reason = f"[no local samples found; fell back to whole-document scale samples] {reason}"

    if rejected:
        reason += f" {len(rejected)} sample(s) rejected as outliers (robust MAD filter)."

    return ScaleEstimate(
        points_per_metre=estimate,
        confidence=level,
        reason=reason,
        samples_used=len(kept),
        samples_rejected=len(rejected),
        score=spread,
        outlier_samples=rejected,
    )


__all__ = [
    "ScaleSample",
    "ScaleEstimate",
    "dimension_length_metres",
    "collect_scale_samples",
    "estimate_scale",
]
