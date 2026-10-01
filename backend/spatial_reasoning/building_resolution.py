"""
Building footprint detection.

Filters Phase 2's building candidates to exclude things that are NOT the
building footprint (rooms, furniture, internal walls, columns, the
compound wall, the road, parking, annotations, dimension lines), then
scores and ranks what's left. Supports multiple building blocks: every
candidate that survives filtering and isn't nested inside another
surviving candidate is kept, and the aggregate footprint is their union
area (approximated as the sum of individual areas, since Phase 2 does
not currently emit overlapping building candidates for the same block).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from backend.schemas.candidates import BuildingCandidate, PlotCandidate
from backend.schemas.extraction import ExtractionResult
from backend.spatial_reasoning import geometry_utils as geo

_ROOM_LABEL_RE = re.compile(
    r"\b(bed\s*room|bedroom|kitchen|hall|toilet|bath(room)?|w\.?c\.?|living|dining|"
    r"store|balcony|passage|verandah|veranda|puja|study|column|col\.)\b",
    re.I,
)
_COMPOUND_WALL_RE = re.compile(r"\bcompound\s*wall\b", re.I)

# A candidate whose bbox occupies more than this fraction of the plot
# bbox is almost certainly the compound-wall outline / plot boundary
# itself, duplicated as a "building" candidate — not a real footprint.
_MAX_PLOT_FRACTION = 0.92
# A candidate smaller than this fraction of the plot bbox is treated as
# furniture/room/annotation noise rather than a structural footprint.
_MIN_PLOT_FRACTION = 0.01
# Aspect ratio beyond which a "building" is almost certainly a
# dimension-line/annotation artifact rather than a real footprint.
_MAX_ASPECT_RATIO = 8.0


@dataclass
class FilteredBuilding:
    candidate: BuildingCandidate
    reject_reason: Optional[str] = None
    kept: bool = True


def _nearby_room_label(bbox, text_evidence, page) -> bool:
    tol = max(bbox.width, bbox.height) * 0.15
    for t in text_evidence:
        if page is not None and t.page != page:
            continue
        if t.bounding_box is None:
            continue
        if not _ROOM_LABEL_RE.search(t.raw_text or ""):
            continue
        gap_x = max(bbox.min_x - t.bounding_box.max_x, t.bounding_box.min_x - bbox.max_x, 0.0)
        gap_y = max(bbox.min_y - t.bounding_box.max_y, t.bounding_box.min_y - bbox.max_y, 0.0)
        if (gap_x**2 + gap_y**2) ** 0.5 <= tol:
            return True
    return False


def filter_and_rank_buildings(
    extraction: ExtractionResult,
    plot: Optional[PlotCandidate],
) -> list[FilteredBuilding]:
    plot_bbox = plot.geometry.bounding_box if (plot and plot.geometry) else None
    plot_polygon = plot.geometry.polygon if (plot and plot.geometry) else None
    plot_page = plot.geometry.source_page if (plot and plot.geometry) else None
    # Prefer the plot's true polygon area over its bounding-box area: for a
    # rotated or non-rectangular (e.g. trapezoidal/L-shaped) plot, the
    # bounding box can substantially overstate the true plot area, which
    # silently loosens/tightens the fraction thresholds below in a way
    # unrelated to the actual shape (audit finding:
    # BUILDCHECK_FORENSIC_AUDIT.md Sec 4.5). Falls back to the bounding-box
    # area only when no polygon is available.
    if plot_polygon is not None:
        plot_area = plot_polygon.area
    else:
        plot_area = plot_bbox.width * plot_bbox.height if plot_bbox else None

    results: list[FilteredBuilding] = []
    kept_bboxes = []
    for cand in extraction.building_candidates:
        if cand.geometry is None or cand.geometry.bounding_box is None:
            results.append(FilteredBuilding(candidate=cand, kept=False, reject_reason="No geometry."))
            continue
        bbox = cand.geometry.bounding_box
        page = cand.geometry.source_page

        if plot_page is not None and page != plot_page:
            results.append(
                FilteredBuilding(candidate=cand, kept=False, reject_reason="On a different page than the resolved plot.")
            )
            continue

        ar = geo.aspect_ratio(bbox)
        if ar > _MAX_ASPECT_RATIO:
            results.append(
                FilteredBuilding(
                    candidate=cand,
                    kept=False,
                    reject_reason=f"Aspect ratio {ar:.1f} is too extreme for a footprint — likely a "
                    "dimension line or annotation artifact.",
                )
            )
            continue

        if plot_area:
            frac = (bbox.width * bbox.height) / plot_area
            if frac >= _MAX_PLOT_FRACTION:
                results.append(
                    FilteredBuilding(
                        candidate=cand,
                        kept=False,
                        reject_reason="Nearly the same footprint as the plot boundary — likely the "
                        "compound wall or the plot outline itself, not a building.",
                    )
                )
                continue
            if frac < _MIN_PLOT_FRACTION:
                results.append(
                    FilteredBuilding(
                        candidate=cand,
                        kept=False,
                        reject_reason="Too small relative to the plot to be a structural footprint — "
                        "likely furniture, a column, or an annotation glyph.",
                    )
                )
                continue

        if _nearby_room_label(bbox, extraction.text_evidence, page):
            results.append(
                FilteredBuilding(
                    candidate=cand,
                    kept=False,
                    reject_reason="Adjacent text names an internal room/space — this is an internal "
                    "partition, not the building footprint.",
                )
            )
            continue

        nested_in_kept = False
        for kb in kept_bboxes:
            if (
                bbox.min_x >= kb.min_x - 1
                and bbox.min_y >= kb.min_y - 1
                and bbox.max_x <= kb.max_x + 1
                and bbox.max_y <= kb.max_y + 1
                and (bbox.width * bbox.height) < (kb.width * kb.height) * 0.98
            ):
                nested_in_kept = True
                break
        if nested_in_kept:
            results.append(
                FilteredBuilding(
                    candidate=cand,
                    kept=False,
                    reject_reason="Nested inside an already-accepted building block's bounding box — "
                    "likely an internal wall/room within it, not a separate block.",
                )
            )
            continue

        kept_bboxes.append(bbox)
        results.append(FilteredBuilding(candidate=cand, kept=True))

    return results


def surviving_candidates(filtered: list[FilteredBuilding]) -> list[BuildingCandidate]:
    return [f.candidate for f in filtered if f.kept]


__all__ = ["FilteredBuilding", "filter_and_rank_buildings", "surviving_candidates"]
