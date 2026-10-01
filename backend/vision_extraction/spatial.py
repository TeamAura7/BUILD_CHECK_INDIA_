from __future__ import annotations

import math
from typing import Iterable, Optional

from backend.config import get_settings
from backend.schemas.geometry import BoundingBox
from backend.schemas.vision import VisionDimension, VisionPageResult, VisionRegion


def vision_bbox_to_page_points(
    bbox: list[float] | None,
    page_width_pts: float = 0.0,
    page_height_pts: float = 0.0,
) -> Optional[BoundingBox]:
    """
    Convert a vision-model bbox into PDF page-point space (the same
    rotated-display frame as CV geometry / `page.get_drawings()`).

    THE BUG THIS FIXES: despite the prompt asking for "image pixel
    coordinates", the Qwen-family models used by every backend in this
    project (SmolVLM2, local Qwen2.5-VL, and the hosted `api` backend's
    default Groq qwen model) actually emit bboxes on an internal
    normalized 0-1000 grid, regardless of the real rendered image size --
    confirmed empirically (every real bbox value stays under ~1000 even
    on a rendered image thousands of pixels wide). The previous
    implementation always assumed raw pixels at `vision_render_dpi` and
    divided by a DPI-derived scale; on a large-format architectural sheet
    that shrinks every vision region/dimension bbox down into a tiny
    sliver near the page origin, so it almost never overlaps real CV
    geometry -- silently zeroing out the vision contribution to plot
    resolution and dimension classification (`region_score`,
    `semantic_dimension_score`, `_vision_semantic_hint` all consume this
    function). `vision_extraction/base.py`'s grounding check had this fix
    applied to its own private copy already; this is the single shared
    implementation both now use.

    Since different backends/providers could in principle follow either
    convention, this auto-detects rather than hardcoding one: if every
    coordinate fits within the normalized grid (<=1024, a small margin
    over 1000) AND the actual rendered page is known to be bigger than
    that in pixel terms, treat it as normalized-0-1000. Otherwise, fall
    back to raw-pixel-at-render-DPI. If the real page size isn't known
    (page_width_pts/page_height_pts <= 0, e.g. a cached vision_result.json
    predating that field), auto-detection can't run and this falls back
    to the old DPI-scale assumption as a best effort.
    """
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    settings = get_settings()
    dpi = settings.vision_render_dpi

    if page_width_pts > 0 and page_height_pts > 0:
        rendered_w_px = page_width_pts / 72.0 * dpi
        rendered_h_px = page_height_pts / 72.0 * dpi
        looks_normalized = (
            max(x1, y1, x2, y2) <= 1024.0
            and (rendered_w_px > 1024.0 or rendered_h_px > 1024.0)
        )
        if looks_normalized:
            return BoundingBox(
                min_x=min(x1, x2) / 1000.0 * page_width_pts,
                min_y=min(y1, y2) / 1000.0 * page_height_pts,
                max_x=max(x1, x2) / 1000.0 * page_width_pts,
                max_y=max(y1, y2) / 1000.0 * page_height_pts,
            )
        # Raw pixels on the actually-rendered image -> convert via that
        # image's real size, not an assumed DPI (robust to any mismatch
        # between the configured DPI and how the image was really made).
        return BoundingBox(
            min_x=min(x1, x2) / rendered_w_px * page_width_pts,
            min_y=min(y1, y2) / rendered_h_px * page_height_pts,
            max_x=max(x1, x2) / rendered_w_px * page_width_pts,
            max_y=max(y1, y2) / rendered_h_px * page_height_pts,
        )

    # Unknown page size: can't auto-detect. Best-effort legacy behavior.
    scale = 72.0 / dpi
    return BoundingBox(
        min_x=min(x1, x2) * scale,
        min_y=min(y1, y2) * scale,
        max_x=max(x1, x2) * scale,
        max_y=max(y1, y2) * scale,
    )


def bbox_gap(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(a.min_x - b.max_x, b.min_x - a.max_x, 0.0)
    dy = max(a.min_y - b.max_y, b.min_y - a.max_y, 0.0)
    return math.hypot(dx, dy)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.min_x, b.min_x), max(a.min_y, b.min_y)
    ix2, iy2 = min(a.max_x, b.max_x), min(a.max_y, b.max_y)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def page_vision(extraction, page_index: int) -> list[VisionPageResult]:
    # Vision page_number is intentionally 1-based for human readability.
    return [p for p in extraction.vision_pages if p.page_number == page_index + 1]


def semantic_regions(extraction, page_index: int, region_types: Iterable[str]) -> list[BoundingBox]:
    wanted = {x.upper() for x in region_types}
    out: list[BoundingBox] = []
    for page in page_vision(extraction, page_index):
        for region in page.regions:
            if region.type.upper() in wanted:
                box = vision_bbox_to_page_points(region.bbox, page.page_width_pts, page.page_height_pts)
                if box is not None:
                    out.append(box)
    return out


def semantic_dimension_score(extraction, page_index: int, candidate_bbox: BoundingBox, types: Iterable[str]) -> float:
    wanted = {x.upper() for x in types}
    best = 0.0
    scale = max(candidate_bbox.width, candidate_bbox.height, 1.0)
    for page in page_vision(extraction, page_index):
        for dim in page.dimensions:
            if dim.type.upper() not in wanted or dim.confidence <= 0:
                continue
            box = vision_bbox_to_page_points(dim.bbox, page.page_width_pts, page.page_height_pts)
            if box is None:
                continue
            gap = bbox_gap(candidate_bbox, box)
            proximity = max(0.0, 1.0 - gap / (scale * 0.35))
            best = max(best, dim.confidence * max(proximity, 0.25 if gap <= scale else 0.0))
    return min(1.0, best)


def region_score(candidate_bbox: BoundingBox, regions: list[BoundingBox]) -> float:
    if not regions:
        return 0.0
    best = 0.0
    for region in regions:
        iou = bbox_iou(candidate_bbox, region)
        center_inside = (
            region.min_x <= candidate_bbox.center.x <= region.max_x
            and region.min_y <= candidate_bbox.center.y <= region.max_y
        )
        gap = bbox_gap(candidate_bbox, region)
        proximity = max(0.0, 1.0 - gap / max(region.width, region.height, 1.0))
        score = max(iou, 0.75 if center_inside else 0.0, proximity * 0.5)
        best = max(best, score)
    return min(1.0, best)
