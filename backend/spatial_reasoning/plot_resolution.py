"""
Plot candidate scoring/selection.

Deliberately NOT "largest rectangle = plot". Every candidate on the
winning page is scored across multiple weighted features (area rank,
perimeter, rectangularity, aspect ratio, position, boundary-annotation
completeness, plot/site label proximity, road adjacency, nesting), and
the highest-scoring candidate wins. Confidence reflects the margin
between the winner and the runner-up, not just "a candidate existed".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from backend.schemas.candidates import PlotCandidate
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import TextEvidence
from backend.schemas.extraction import ExtractionResult
from backend.spatial_reasoning import geometry_utils as geo
from backend.vision_extraction.spatial import region_score, semantic_dimension_score, semantic_regions

import re as _re

_LABEL_RE_PLOT = _re.compile(r"\b(plot|site|property)\b", _re.I)
_ROAD_RE = _re.compile(r"\broad\b", _re.I)

_WEIGHTS = {
    "area_rank": 0.20,
    "rectangularity": 0.15,
    "aspect_ratio": 0.10,
    "boundary_annotation": 0.20,
    "label_match": 0.20,
    "road_adjacency": 0.10,
    "not_nested": 0.05,
}


@dataclass
class ScoredPlotCandidate:
    candidate: PlotCandidate
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


def _aspect_score(bbox) -> float:
    ar = geo.aspect_ratio(bbox)
    if not math.isfinite(ar):
        return 0.0
    # Most real plots are well under a 6:1 aspect ratio; score decays past that.
    return max(0.0, 1.0 - max(0.0, ar - 1.0) / 6.0)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _label_match_score(bbox, text_evidence: list[TextEvidence], page: Optional[int], tol_basis: float) -> float:
    """Reward candidates whose bounding box is near text mentioning plot/site/property."""
    hits = 0
    tol = tol_basis * 0.35
    for t in text_evidence:
        if page is not None and t.page != page:
            continue
        if not _LABEL_RE_PLOT.search(t.raw_text or ""):
            continue
        if t.bounding_box is None:
            continue
        if _bbox_gap(bbox, t.bounding_box) <= tol:
            hits += 1
    return min(1.0, hits * 0.5)


def _boundary_annotation_score(bbox, dimensions, page: Optional[int], tol_basis: float) -> float:
    """Fraction of the candidate's 4 notional sides that have a nearby dimension."""
    if not dimensions:
        return 0.0
    tol = tol_basis * 0.08
    sides_hit = set()
    for dim in dimensions:
        if dim.geometry is None:
            continue
        mid = geo.edge_midpoint(dim.geometry)
        if abs(mid.x - bbox.min_x) <= tol:
            sides_hit.add("left")
        if abs(mid.x - bbox.max_x) <= tol:
            sides_hit.add("right")
        if abs(mid.y - bbox.min_y) <= tol:
            sides_hit.add("top")
        if abs(mid.y - bbox.max_y) <= tol:
            sides_hit.add("bottom")
    return len(sides_hit) / 4.0


def _road_adjacency_score(bbox, road_bboxes, tol_basis: float) -> float:
    if not road_bboxes:
        return 0.0
    best = min(_bbox_gap(bbox, rb) for rb in road_bboxes)
    tol = tol_basis * 0.15
    if best <= 0:
        return 1.0
    return max(0.0, 1.0 - best / max(tol, 1e-6))


def _bbox_gap(a, b) -> float:
    dx = max(a.min_x - b.max_x, b.min_x - a.max_x, 0.0)
    dy = max(a.min_y - b.max_y, b.min_y - a.max_y, 0.0)
    return math.hypot(dx, dy)


def _vision_plot_score(extraction: ExtractionResult, bbox, page: Optional[int]) -> float:
    if page is None or not extraction.vision_pages:
        return 0.0
    site_regions = semantic_regions(extraction, page, {"SITE_PLAN"})
    region = region_score(bbox, site_regions)
    dimensions = semantic_dimension_score(
        extraction, page, bbox, {"PLOT_WIDTH", "PLOT_DEPTH"}
    )
    return min(1.0, 0.45 * region + 0.55 * dimensions)


# A genuine plot at most nests its OWN building/walls/fixtures -- a
# handful of candidates. A sheet-spanning container (the drawing-sheet
# border/frame, or a title-block panel) instead fully contains MOST of
# the independent sub-drawings on a multi-view sheet, since it spans the
# whole page. `is_page_frame_like` (candidate_geometry.py) already
# rejects a frame that touches the page edge closely, but a real printed
# sheet border commonly sits with a genuine margin (tens of points)
# inset from the true page edge, which that edge-touch check misses --
# confirmed on the real PLAN1.pdf fixture (a composite sheet with a site
# plan + 4 floor plans + section + details), whose own border sits ~45-
# 48pt inset on every side and was never caught by that check, then won
# plot-candidate scoring outright by dwarfing every genuine sub-drawing.
# Requiring BOTH an absolute floor (>=4 nested siblings) and a fraction
# (>=half of all other same-page candidates) keeps this from ever firing
# on an ordinary plot+building pair (nesting exactly one sibling is the
# normal, expected case, not a frame).
_MIN_CONTAINED_SIBLINGS_FOR_SHEET_CONTAINER = 4
_MIN_CONTAINED_FRACTION_FOR_SHEET_CONTAINER = 0.5


def _looks_like_sheet_spanning_container(cand, others_on_page: list) -> bool:
    if not others_on_page:
        return False
    bbox = cand.geometry.bounding_box or cand.geometry.polygon.bounding_box
    contained = 0
    for other in others_on_page:
        other_bbox = other.geometry.bounding_box or other.geometry.polygon.bounding_box
        if (
            bbox.min_x <= other_bbox.min_x and bbox.min_y <= other_bbox.min_y
            and bbox.max_x >= other_bbox.max_x and bbox.max_y >= other_bbox.max_y
        ):
            contained += 1
    return (
        contained >= _MIN_CONTAINED_SIBLINGS_FOR_SHEET_CONTAINER
        and contained / len(others_on_page) >= _MIN_CONTAINED_FRACTION_FOR_SHEET_CONTAINER
    )


def score_plot_candidates(
    extraction: ExtractionResult,
) -> tuple[Optional[ScoredPlotCandidate], list[ScoredPlotCandidate], str]:
    """
    Returns (winner, all_scored, page_selection_note). Candidates are
    grouped by source page (a plan may have multiple sheets); the page
    with the highest-scoring candidate is selected.
    """
    candidates = [c for c in extraction.plot_candidates if c.geometry and c.geometry.polygon]
    if not candidates:
        return None, [], "No plot candidates with polygon geometry were extracted."

    by_page_all: dict[Optional[int], list] = {}
    for c in candidates:
        by_page_all.setdefault(c.geometry.source_page, []).append(c)
    sheet_container_ids = {
        id(c)
        for page_candidates in by_page_all.values()
        for c in page_candidates
        if _looks_like_sheet_spanning_container(c, [o for o in page_candidates if o is not c])
    }
    sheet_containers_excluded = len(sheet_container_ids)
    candidates = [c for c in candidates if id(c) not in sheet_container_ids]
    if not candidates:
        return None, [], (
            f"All {sheet_containers_excluded} candidate(s) looked like a sheet-spanning "
            "border/container (each nested most of the others) rather than a real plot boundary."
        )

    road_bboxes_by_page: dict[int, list] = {}
    for r in extraction.road_candidates:
        if r.geometry and r.geometry.bounding_box is not None:
            road_bboxes_by_page.setdefault(r.geometry.source_page or -1, []).append(r.geometry.bounding_box)

    areas = [c.geometry.polygon.area for c in candidates]
    max_area = max(areas) if areas else 1.0

    # Proximity-based scores (label_match, boundary_annotation, road_adjacency)
    # search within a tolerance meant to reflect "how big is a drawing at
    # about this scale" -- using each candidate's OWN bbox size for that
    # basis breaks down for a page-spanning sheet-border/frame candidate:
    # its self-referential tolerance balloons to the size of the whole
    # sheet, letting it "find" labels/dimensions anywhere on the page that
    # actually belong to a real, much smaller sub-drawing elsewhere on a
    # multi-view sheet, and win purely on size. Using the PAGE's median
    # candidate extent as the tolerance basis instead keeps normal-sized
    # candidates' behavior essentially unchanged (median ~= their own size
    # on a typical single-drawing sheet) while clamping an outlier-sized
    # frame candidate down to a page-typical search radius.
    extents_by_page: dict[Optional[int], list[float]] = {}
    for cand in candidates:
        bbox = cand.geometry.bounding_box or cand.geometry.polygon.bounding_box
        extents_by_page.setdefault(cand.geometry.source_page, []).append(max(bbox.width, bbox.height))
    tol_basis_by_page = {page: _median(extents) for page, extents in extents_by_page.items()}

    scored: list[ScoredPlotCandidate] = []
    for cand, area in zip(candidates, areas):
        bbox = cand.geometry.bounding_box or cand.geometry.polygon.bounding_box
        page = cand.geometry.source_page
        tol_basis = tol_basis_by_page.get(page, max(bbox.width, bbox.height))
        breakdown = {
            "area_rank": area / max_area if max_area else 0.0,
            "rectangularity": geo.rectangularity(cand.geometry.polygon),
            "aspect_ratio": _aspect_score(bbox),
            "boundary_annotation": _boundary_annotation_score(bbox, extraction.dimensions, page, tol_basis),
            "label_match": _label_match_score(bbox, extraction.text_evidence, page, tol_basis),
            "road_adjacency": _road_adjacency_score(bbox, road_bboxes_by_page.get(page, []), tol_basis),
            "not_nested": 1.0,  # penalized below if nested inside another candidate
        }
        for other in candidates:
            if other is cand or other.geometry is None or other.geometry.bounding_box is None:
                continue
            if other.geometry.source_page != page:
                continue
            other_bbox = other.geometry.bounding_box
            if (
                other_bbox.width * other_bbox.height > bbox.width * bbox.height * 1.02
                and bbox.min_x >= other_bbox.min_x
                and bbox.min_y >= other_bbox.min_y
                and bbox.max_x <= other_bbox.max_x
                and bbox.max_y <= other_bbox.max_y
            ):
                breakdown["not_nested"] = 0.0
                break

        total = sum(_WEIGHTS[k] * v for k, v in breakdown.items())
        if extraction.vision_pages:
            vision_score = _vision_plot_score(extraction, bbox, page)
            total += 0.20 * vision_score
            breakdown["vision_semantics"] = vision_score
        scored.append(ScoredPlotCandidate(candidate=cand, score=total, breakdown=breakdown))

    scored.sort(key=lambda s: s.score, reverse=True)
    winner = scored[0]
    note = f"Selected from {len(scored)} candidate(s) across {len({c.candidate.geometry.source_page for c in scored})} page(s)."
    if sheet_containers_excluded:
        note += (
            f" ({sheet_containers_excluded} sheet-spanning container candidate(s) excluded -- "
            "each nested most of the others on its page.)"
        )
    return winner, scored, note


# A winner's own ABSOLUTE score must clear this floor before margin-over-
# runner-up is even considered -- margin alone is not a safe proxy for
# plausibility, because on a document with NO real site plan (e.g. a
# photograph with no vector geometry), every candidate can be equally poor,
# which can produce a comfortably LARGE margin between two BAD candidates
# just as easily as a small one, while a genuine plan's two best candidates
# (the real plot and, say, a real building outline or an adjacent genuine
# rectangle) can legitimately score close to each other without either
# being wrong. Confirmed directly: measuring winner scores across the 5
# real PDF plans in this project's own regression corpus, the 4 plans with
# a real, resolvable site plan (PLAN2/4/5/6) scored 0.74-0.89, while PLAN7
# (a photograph of a blueprint with no vector geometry at all -- ground
# truth requires every geometric field to stay unresolved) scored 0.59 for
# its best (spurious) candidate -- yet the OLD margin-only logic rated
# PLAN7's winner MEDIUM (a comfortable margin over an equally spurious
# runner-up) while rating PLAN2/5/6's genuinely correct winners LOW (a
# narrow margin over a legitimate, similarly-plausible runner-up) --
# exactly backwards. n=5 is a small corpus; re-measure as it grows, per
# this project's own established practice for this kind of floor (see
# DXF_FAILURE_TAXONOMY.md's "Provisional thresholds" section).
_MIN_USABLE_PLOT_SCORE = 0.65


def plot_confidence(winner: ScoredPlotCandidate, all_scored: list[ScoredPlotCandidate]) -> tuple[ConfidenceLevel, str]:
    if winner.score < _MIN_USABLE_PLOT_SCORE:
        basis = "the only plot candidate found" if len(all_scored) == 1 else "the winning candidate"
        return ConfidenceLevel.LOW, (
            f"{basis} scored {winner.score:.2f}, below the {_MIN_USABLE_PLOT_SCORE} usability floor -- "
            "low absolute plot-likeness, regardless of its margin over any runner-up."
        )
    if len(all_scored) == 1:
        return ConfidenceLevel.MEDIUM, "Only one plot candidate was found; accepted on its own merits."
    runner_up = all_scored[1]
    margin = winner.score - runner_up.score
    if margin >= 0.15:
        return ConfidenceLevel.HIGH, f"Clear winner (score {winner.score:.2f}) with a {margin:.2f} margin over the runner-up."
    if margin >= 0.05:
        return ConfidenceLevel.MEDIUM, f"Winner (score {winner.score:.2f}) beats the runner-up by {margin:.2f}."
    return ConfidenceLevel.LOW, f"Winner only narrowly ({margin:.2f}) beats the runner-up ({runner_up.score:.2f}); ambiguous."


__all__ = ["ScoredPlotCandidate", "score_plot_candidates", "plot_confidence"]
