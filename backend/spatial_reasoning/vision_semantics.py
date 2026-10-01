"""Semantic dimension/area evidence helpers.

The vision model is used here as a semantic interpreter, not as a measuring
instrument. Values are admitted into the normalized plan only when they are
supported by the page's native dimension/area evidence (or when the page has
no native text at all). Geometry remains a validation signal.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import BoundingBox, Dimension
from backend.schemas.vision import VisionPageResult
from backend.spatial_reasoning.scale import dimension_length_metres


_DECIMAL_SMALL = re.compile(r"^\.?\d+(?:\.\d+)?$")


def vision_values(extraction: ExtractionResult, page: int, semantic_type: str, min_confidence: float = 0.75) -> list[float]:
    """Return unique VLM values for one semantic type on one page."""
    values: list[float] = []
    for vp in extraction.vision_pages:
        if vp.page_number != page + 1:
            continue
        for d in vp.dimensions:
            if d.type.upper() != semantic_type.upper() or d.value is None or d.confidence < min_confidence:
                continue
            value = float(d.value)
            if d.unit and d.unit.lower() in {"mm", "cm", "ft", "in"}:
                # Core architectural dimensions in this project are normally metres.
                # Ignore non-metre semantic readings here; native unit conversion can
                # still handle them elsewhere.
                continue
            if not any(math.isclose(value, x, rel_tol=0.0, abs_tol=max(0.02, value * 0.01)) for x in values):
                values.append(value)
    return values


def _native_matches(dimensions: list[Dimension], page: int, value: float) -> list[Dimension]:
    out = []
    for d in dimensions:
        if d.page != page:
            continue
        m = dimension_length_metres(d)
        if m is None:
            continue
        if math.isclose(m, value, rel_tol=0.0, abs_tol=max(0.08, value * 0.02)):
            out.append(d)
    return out


def preferred_vision_value(
    extraction: ExtractionResult,
    page: int,
    semantic_type: str,
    min_confidence: float = 0.75,
) -> Optional[float]:
    """Select a VLM semantic value only when it is independently grounded.

    On vector PDFs, a VLM value must also occur in the native dimension layer.
    On scanned pages with no native dimensions, the VLM confidence threshold is
    the fallback evidence gate.
    """
    values = vision_values(extraction, page, semantic_type, min_confidence)
    if not values:
        return None
    grounded = [v for v in values if _native_matches(extraction.dimensions, page, v)]
    if grounded:
        return grounded[0]
    has_native = any(d.page == page for d in extraction.dimensions)
    return None if has_native else values[0]


def preferred_vision_area(
    extraction: ExtractionResult,
    page: int,
    semantic_type: str,
    min_confidence: float = 0.75,
) -> Optional[float]:
    for vp in extraction.vision_pages:
        if vp.page_number != page + 1:
            continue
        for a in vp.areas:
            if a.type.upper() != semantic_type.upper() or a.value is None or a.confidence < min_confidence:
                continue
            if (a.unit or "m2").lower() not in {"m2", "sq_m", "sqm", "sq.m", "sq.mt"}:
                continue
            value = float(a.value)
            # Ground an explicit area against the page text when available.
            if any(str_token_matches(value, t.raw_text) for t in extraction.text_evidence if t.page == page):
                return value
            # A native area statement is still strong evidence even if the exact
            # text was grouped into a large PDF text span.
            if not any(t.page == page for t in extraction.text_evidence):
                return value
    return None


def str_token_matches(value: float, text: str) -> bool:
    tokens = {f"{value:.2f}", f"{value:.1f}", f"{value:g}"}
    return any(token in (text or "") for token in tokens)


def floor_count_from_vision(extraction: ExtractionResult, page: int, min_confidence: float = 0.7) -> Optional[int]:
    """Count distinct floor-plan semantic regions on the page."""
    types = set()
    for vp in extraction.vision_pages:
        if vp.page_number != page + 1:
            continue
        for r in vp.regions:
            typ = r.type.upper()
            if r.confidence >= min_confidence and typ.endswith("_FLOOR_PLAN"):
                types.add(typ)
    return len(types) if types else None


def site_small_dimension_candidates(extraction: ExtractionResult, page: int, max_value_m: float = 2.0) -> list[Dimension]:
    """Return small decimal dimensions likely to be setback annotations.

    The leading-dot form (.46/.47) is common in CAD drawings. This helper only
    supplies candidates; side assignment remains spatial and must not be based
    on a hard-coded value.
    """
    result: list[Dimension] = []
    for d in extraction.dimensions:
        if d.page != page or d.geometry is None:
            continue
        value = dimension_length_metres(d)
        if value is None or not (0 < value <= max_value_m):
            continue
        label = (d.label or "").strip()
        # Avoid door/window schedules such as 0.91 x 2.10 and room labels.
        if "x" in label.lower() or "×" in label:
            continue
        if not _DECIMAL_SMALL.match(label):
            continue
        result.append(d)
    return result


def vision_setbacks(extraction: ExtractionResult, page: int, min_confidence: float = 0.75) -> dict[str, float]:
    """Return explicitly labeled VLM setback values when present."""
    result: dict[str, float] = {}
    for vp in extraction.vision_pages:
        if vp.page_number != page + 1:
            continue
        for d in vp.dimensions:
            if d.value is None or d.confidence < min_confidence:
                continue
            typ = d.type.upper()
            mapping = {
                "FRONT_SETBACK": "front",
                "REAR_SETBACK": "rear",
                "LEFT_SETBACK": "left",
                "RIGHT_SETBACK": "right",
            }
            side = mapping.get(typ)
            if side:
                result[side] = float(d.value)
    return result


__all__ = [
    "vision_values",
    "preferred_vision_value",
    "preferred_vision_area",
    "floor_count_from_vision",
    "site_small_dimension_candidates",
    "vision_setbacks",
]


def _line_orientation(line) -> str:
    return "horizontal" if abs(line.end.x - line.start.x) >= abs(line.end.y - line.start.y) else "vertical"


def _bbox_from_lines(lines) -> Optional[BoundingBox]:
    if not lines:
        return None
    xs = [p.x for line in lines for p in (line.start, line.end)]
    ys = [p.y for line in lines for p in (line.start, line.end)]
    return BoundingBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _rect_score(h1, h2, v1, v2, target_ratio: float) -> float:
    hs = sorted([h1, h2], key=lambda l: (l.start.y + l.end.y) / 2)
    vs = sorted([v1, v2], key=lambda l: (l.start.x + l.end.x) / 2)
    bbox = _bbox_from_lines([*hs, *vs])
    if bbox is None or bbox.width <= 0 or bbox.height <= 0:
        return -1e9

    # Dimension lines are often offset slightly from the actual boundary.
    # Reward a pair when the horizontal line spans are close to the vertical
    # line x-coordinates and vice versa.
    vx_lo = min(min(v.start.x, v.end.x) for v in vs)
    vx_hi = max(max(v.start.x, v.end.x) for v in vs)
    hy_lo = min(min(h.start.y, h.end.y) for h in hs)
    hy_hi = max(max(h.start.y, h.end.y) for h in hs)
    gaps = []
    for h in hs:
        gaps.append(max(vx_lo - max(h.start.x, h.end.x), min(h.start.x, h.end.x) - vx_hi, 0.0))
    for v in vs:
        gaps.append(max(hy_lo - max(v.start.y, v.end.y), min(v.start.y, v.end.y) - hy_hi, 0.0))
    proximity = max(0.0, 1.0 - (sum(gaps) / 4.0) / max(bbox.width, bbox.height, 1.0) * 8.0)
    ratio = bbox.width / bbox.height
    ratio_error = abs(math.log(max(ratio, 1e-6) / max(target_ratio, 1e-6)))
    ratio_score = math.exp(-ratio_error * 2.0)
    return 0.55 * ratio_score + 0.45 * proximity


def _semantic_text_positions(extraction: ExtractionResult, page: int, value: float, horizontal: bool) -> list[tuple[float, float, Dimension]]:
    out = []
    for d in extraction.dimensions:
        if d.page != page or d.text_bounding_box is None:
            continue
        m = dimension_length_metres(d)
        if m is None or not math.isclose(m, value, rel_tol=0.0, abs_tol=max(0.08, value * 0.02)):
            continue
        b = d.text_bounding_box
        is_horizontal = b.width >= b.height
        if is_horizontal != horizontal:
            continue
        out.append((b.center.x, b.center.y, d))
    return out


def semantic_frame_from_dimensions(
    extraction: ExtractionResult,
    page: int,
    width_value: float,
    depth_value: float,
    containing_bbox: Optional[BoundingBox] = None,
) -> Optional[BoundingBox]:
    """Build a page-space frame from two semantic dimension values.

    This is useful when OpenCV fails to close a plot/building polygon. The
    source remains the native dimension lines; the VLM only tells us which
    values are semantically relevant. No page/template coordinates are used.
    """
    # First use the printed dimension-text positions. Native CAD dimension-line
    # association can occasionally attach a label to an unrelated long line;
    # the text position itself is stable and is the safer anchor for selecting
    # the correct drawing region.
    hpos = _semantic_text_positions(extraction, page, width_value, horizontal=True)
    vpos = _semantic_text_positions(extraction, page, depth_value, horizontal=False)
    if len(hpos) >= 2 and len(vpos) >= 2:
        candidates = []
        for hi in range(len(hpos)):
            for hj in range(hi + 1, len(hpos)):
                for vi in range(len(vpos)):
                    for vj in range(vi + 1, len(vpos)):
                        xs = [hpos[hi][0], hpos[hj][0], vpos[vi][0], vpos[vj][0]]
                        ys = [hpos[hi][1], hpos[hj][1], vpos[vi][1], vpos[vj][1]]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        w = max_x - min_x
                        h = max_y - min_y
                        if w <= 0 or h <= 0:
                            continue
                        center = BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y).center
                        if containing_bbox is not None and not (
                            containing_bbox.min_x <= center.x <= containing_bbox.max_x
                            and containing_bbox.min_y <= center.y <= containing_bbox.max_y
                        ):
                            continue
                        ratio_score = math.exp(-abs(math.log((w / h) / max(width_value / max(depth_value, 1e-9), 1e-9))) * 2.0)
                        # Opposite dimension labels should be on opposite sides;
                        # this rejects duplicate labels from repeated floor plans.
                        ysep = abs(hpos[hi][1] - hpos[hj][1])
                        xsep = abs(vpos[vi][0] - vpos[vj][0])
                        if ysep < h * 0.35 or xsep < w * 0.35:
                            continue
                        score = ratio_score
                        candidates.append((score, BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)))
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]

    hs = []
    vs = []
    for d in extraction.dimensions:
        if d.page != page or d.geometry is None:
            continue
        m = dimension_length_metres(d)
        if m is None:
            continue
        if math.isclose(m, width_value, rel_tol=0.0, abs_tol=max(0.08, width_value * 0.02)) and _line_orientation(d.geometry) == "horizontal":
            hs.append(d.geometry)
        if math.isclose(m, depth_value, rel_tol=0.0, abs_tol=max(0.08, depth_value * 0.02)) and _line_orientation(d.geometry) == "vertical":
            vs.append(d.geometry)

    if len(hs) < 2 or len(vs) < 2:
        return None

    target_ratio = width_value / max(depth_value, 1e-9)
    best = None
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            for k in range(len(vs)):
                for l in range(k + 1, len(vs)):
                    bbox = _bbox_from_lines([hs[i], hs[j], vs[k], vs[l]])
                    if bbox is None:
                        continue
                    if containing_bbox is not None:
                        # For a building frame, require its center to lie inside
                        # the already resolved site frame.
                        c = bbox.center
                        if not (
                            containing_bbox.min_x <= c.x <= containing_bbox.max_x
                            and containing_bbox.min_y <= c.y <= containing_bbox.max_y
                        ):
                            continue
                        # Do not let a candidate straddle far outside the parent.
                        if bbox.width > containing_bbox.width * 1.05 or bbox.height > containing_bbox.height * 1.05:
                            continue
                    score = _rect_score(hs[i], hs[j], vs[k], vs[l], target_ratio)
                    # Prefer the tightest plausible rectangle when scores are close.
                    tie = -(bbox.width * bbox.height) * 1e-9
                    key = score + tie
                    if best is None or key > best[0]:
                        best = (key, bbox)
    return best[1] if best else None


__all__.append("semantic_frame_from_dimensions")


def setback_values_from_native_site(
    extraction: ExtractionResult,
    page: int,
    plot_bbox: Optional[BoundingBox],
    road_bbox: Optional[BoundingBox],
    points_per_metre: float,
) -> dict[str, float]:
    """Resolve explicit small setback labels spatially from a semantic site frame."""
    if plot_bbox is None or points_per_metre <= 0:
        return {}
    candidates = site_small_dimension_candidates(extraction, page)
    if not candidates:
        return {}

    top = bottom = left = right = None
    cx, cy = plot_bbox.center.x, plot_bbox.center.y
    # Only accept labels inside/very close to the site frame. This excludes
    # door/window schedules and floor-plan dimensions elsewhere on the sheet.
    pad = max(plot_bbox.width, plot_bbox.height) * 0.05
    for d in candidates:
        b = d.geometry
        text_box = d.text_bounding_box
        if text_box is not None:
            mid = (text_box.center.x, text_box.center.y)
        else:
            mid = ((b.start.x + b.end.x) / 2.0, (b.start.y + b.end.y) / 2.0)
        if not (plot_bbox.min_x - pad <= mid[0] <= plot_bbox.max_x + pad and plot_bbox.min_y - pad <= mid[1] <= plot_bbox.max_y + pad):
            continue
        value = dimension_length_metres(d)
        if value is None:
            continue
        orient = "horizontal" if text_box is not None and text_box.width >= text_box.height else _line_orientation(b)
        if orient == "horizontal":
            # Horizontal dimension annotation means a left/right gap.
            if mid[0] >= cx:
                right = value if right is None else min(right, value)
            else:
                left = value if left is None else min(left, value)
        else:
            # Vertical dimension annotation means a top/bottom gap.
            if mid[1] <= cy:
                top = value if top is None else min(top, value)
            else:
                bottom = value if bottom is None else min(bottom, value)

    # Determine which plot edge is front from the road text/geometry.
    front_edge = None
    if road_bbox is not None:
        distances = {
            "left": abs(road_bbox.center.x - plot_bbox.min_x),
            "right": abs(road_bbox.center.x - plot_bbox.max_x),
            "top": abs(road_bbox.center.y - plot_bbox.min_y),
            "bottom": abs(road_bbox.center.y - plot_bbox.max_y),
        }
        front_edge = min(distances, key=distances.get)

    # If no road evidence exists, do not invent a front/rear orientation.
    if front_edge is None:
        return {}

    # The side directly facing the road is front; the opposite is rear.
    if front_edge == "right":
        result = {"front": right, "rear": left, "right": top, "left": bottom}
    elif front_edge == "left":
        result = {"front": left, "rear": right, "right": bottom, "left": top}
    elif front_edge == "top":
        result = {"front": top, "rear": bottom, "right": left, "left": right}
    else:
        result = {"front": bottom, "rear": top, "right": right, "left": left}

    # Missing is a real state. A 0m setback is meaningful when the geometry
    # shows the building edge coincident with the plot edge, but we do not
    # infer zero merely because a printed annotation was absent.
    return {k: v for k, v in result.items() if v is not None}


__all__.append("setback_values_from_native_site")
