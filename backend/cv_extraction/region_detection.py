"""
Deterministic, caption-anchored semantic region detection.

This exists to ground Vision's own region guesses in something that does
not depend on a VLM reading a downscaled full-page image correctly. On a
real, complex sheet (rotated, multi-view, dashed boundaries) a VLM's first-
pass region bbox for e.g. `SITE_PLAN` can land somewhere else on the page
entirely -- confirmed on `data/test_plans/PLAN5.pdf`, where Vision's own
first pass placed `SITE_PLAN` at the top-left of the page while the
sheet's actual site plan sits at the bottom-right (independently confirmed
by `backend.cv_extraction.site_plan.extract_independent_cv`'s
`site_plan_bbox_pts`, which is anchored to the printed "SITE PLAN" caption
and corroborated by the sheet's own area statement). A focused Vision pass
cropped from the wrong bbox reads whatever pixels it's given with full
confidence and no way to tell, from its own output, that the crop itself
was wrong.

The technique here is the same one `site_plan.py` already uses and has
proven on the real plans in `data/test_plans/`: find the region's own
printed caption (by regex, rotation-tolerant since a caption's bounding box
is taller-than-wide when the sheet is drawn sideways) and take a window
around it sized relative to the page. This module generalizes that from
`SITE_PLAN` only to every semantic region type Vision is asked to identify,
so a caller can cross-check or override a Vision-claimed region bbox before
using it to crop, and so a region-scoped OpenCV pass has a geometry-only
way to know where to look.

This module never decides what a number IN a region means -- it only
answers "where on the page is the AREA_STATEMENT table / the SITE_PLAN
drawing / the ELEVATION view", the same separation of concerns as the rest
of `backend/cv_extraction`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from backend.cv_extraction.raw_types import RawTextItem
from backend.schemas.geometry import BoundingBox
from backend.schemas.regions import DetectedRegion, RegionType

# Caption patterns, ordered by which region type they identify. Deliberately
# not anchored (`^`) for the same reason `site_plan.py`'s generalized area
# labels aren't: OCR on a rotated/scanned sheet can merge a caption into a
# longer run of neighbouring text, so the caption must be findable as a
# substring, not just as a whole line.
_CAPTION_PATTERNS: dict[RegionType, re.Pattern] = {
    "SITE_PLAN": re.compile(r"\bSITE\s+PLAN\b", re.I),
    "AREA_STATEMENT": re.compile(r"\bAREA\s+STATEMENT\b", re.I),
    "GROUND_FLOOR_PLAN": re.compile(r"\bGROUND\s+FLOOR\s+PLAN\b", re.I),
    "FIRST_FLOOR_PLAN": re.compile(r"\bFIRST\s+FLOOR\s+PLAN\b", re.I),
    "SECOND_FLOOR_PLAN": re.compile(r"\bSECOND\s+FLOOR\s+PLAN\b", re.I),
    "THIRD_FLOOR_PLAN": re.compile(r"\bTHIRD\s+FLOOR\s+PLAN\b", re.I),
    "OTHER_FLOOR_PLAN": re.compile(
        r"\b(?:TERRACE|STILT|BASEMENT|FOURTH|FIFTH|SIXTH|TYPICAL|MEZZANINE)\s+FLOOR\s+PLAN\b", re.I
    ),
    "ELEVATION": re.compile(r"\bELEVATION\b", re.I),
    # "SECTION A-A" / "SECTION X-X" / "SECTION-1" / "CROSS SECTION" -- but
    # not an unrelated legal clause reference ("as per section 5 of the
    # Act"), which the caption-length guard in `_find_caption_anchor`
    # filters out (a clause reference sits inside a long sentence; a
    # drawing caption is a short standalone label).
    "SECTION": re.compile(r"\bSECTION\b", re.I),
}

# A caption label is a short standalone line, not a word embedded in a
# sentence ("as per section 5 of the Karnataka..."). This threshold is
# generous (real captions like "SITE PLAN SCALE 1:200" or "SECOND FLOOR
# PLAN (PROPOSED)" are well under it) while still excluding prose.
_MAX_CAPTION_LENGTH = 45

# Same search-window sizing already proven on real sheets by
# `site_plan.py._site_region` for the SITE_PLAN caption specifically --
# reused unchanged as the generic default for every region type. A caption
# is printed alongside its own drawing/table, so the window extends further
# in the direction the drawing actually is; which direction that is swaps
# when the caption itself reads sideways (sheet drawn rotated), detected
# from the caption's own aspect ratio rather than any page-rotation
# metadata (which can read 0 even when the content is visually rotated).
_HALF_WIDTH_FRACTION = 0.18
_ABOVE_FRACTION = 0.30
_BELOW_FRACTION = 0.06


def _find_caption_anchor(text_items: list[RawTextItem], pattern: re.Pattern) -> Optional[RawTextItem]:
    matches = [
        t for t in text_items
        if len((t.text or "").strip()) <= _MAX_CAPTION_LENGTH and pattern.search(t.text or "")
    ]
    if not matches:
        return None
    # Prefer the clearest/longest caption match, same tie-break as
    # `site_plan.py._find_site_anchor`.
    return max(matches, key=lambda t: (len(t.text), -t.bounding_box.min_y))


def _region_window(anchor: RawTextItem, page_width: float, page_height: float) -> BoundingBox:
    """A window around a caption, sized relative to the sheet and oriented to match it.

    See `site_plan.py._site_region`'s docstring for the full reasoning
    (PLAN5's vertical "SITE PLAN SCALE 1:200" caption is the concrete case
    this rotation handling exists for). Duplicated here in generalized form
    rather than imported, since `site_plan.py`'s version is a private
    implementation detail of its own SITE_PLAN-specific resolver.
    """
    box = anchor.bounding_box
    c = box.center
    caption_is_rotated = box.height > box.width

    if caption_is_rotated:
        along = page_height * _HALF_WIDTH_FRACTION
        across = page_width * _ABOVE_FRACTION
        return BoundingBox(
            min_x=max(0.0, c.x - across),
            min_y=max(0.0, c.y - along),
            max_x=min(page_width, c.x + across),
            max_y=min(page_height, c.y + along),
        )

    half_width = page_width * _HALF_WIDTH_FRACTION
    return BoundingBox(
        min_x=max(0.0, c.x - half_width),
        min_y=max(0.0, c.y - page_height * _ABOVE_FRACTION),
        max_x=min(page_width, c.x + half_width),
        max_y=min(page_height, c.y + page_height * _BELOW_FRACTION),
    )


def detect_regions_from_text_items(
    text_items: list[RawTextItem],
    page_width: float,
    page_height: float,
    page_number: int = 0,
    region_types: Optional[list[RegionType]] = None,
) -> list[DetectedRegion]:
    """Locate every requested region type by its printed caption.

    Pure function: does no PDF I/O itself. Returns one `DetectedRegion` per
    region type that has a findable caption; a type with no caption match
    on this page is simply absent from the result (never guessed).
    """
    if page_width <= 0 or page_height <= 0:
        return []
    wanted = region_types or list(_CAPTION_PATTERNS.keys())
    found: list[DetectedRegion] = []
    for region_type in wanted:
        pattern = _CAPTION_PATTERNS.get(region_type)
        if pattern is None:
            continue
        anchor = _find_caption_anchor(text_items, pattern)
        if anchor is None:
            continue
        window = _region_window(anchor, page_width, page_height)
        found.append(
            DetectedRegion(
                type=region_type,
                bbox_pts=window,
                confidence=0.9,
                evidence=(anchor.text or "").strip(),
                source="caption_anchor",
                page=page_number,
            )
        )
    return found


def detect_regions(
    document_path: Path,
    page_number: int,
    region_types: Optional[list[RegionType]] = None,
) -> list[DetectedRegion]:
    """Locate regions on one page of a PDF, extracting text itself (native, OCR fallback).

    Same native-text-then-OCR-fallback acquisition pattern already used by
    `backend.cv_extraction.site_plan.extract_independent_cv` -- kept
    self-contained here (rather than requiring a caller to have already run
    that resolver) so this works from any entry point, including the
    Vision-only CLI/validation paths that never touch `extract_independent_cv`.
    """
    from backend.cv_extraction import ocr_fallback, pdf_native

    doc = pdf_native.open_document(document_path)
    try:
        if page_number >= doc.page_count:
            return []
        page = doc.load_page(page_number)
        meta = pdf_native.extract_page_metadata(page, page_number)
        text_items = pdf_native.extract_text_items(page, page_number)
        if not pdf_native.has_sufficient_native_text(text_items):
            try:
                image = ocr_fallback.rasterize_page(page, dpi=200.0)
                text_items = ocr_fallback.ocr_page_any_orientation(image, page_number, dpi=200.0)
            except Exception:
                pass
        return detect_regions_from_text_items(
            text_items, meta.width_pts, meta.height_pts, page_number, region_types
        )
    finally:
        doc.close()


def best_match(regions: list[DetectedRegion], region_type: RegionType) -> Optional[DetectedRegion]:
    candidates = [r for r in regions if r.type == region_type]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.confidence)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two page-space bounding boxes, for cross-checking a Vision bbox."""
    ix0, iy0 = max(a.min_x, b.min_x), max(a.min_y, b.min_y)
    ix1, iy1 = min(a.max_x, b.max_x), min(a.max_y, b.max_y)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(a.width, 0.0) * max(a.height, 0.0)
    area_b = max(b.width, 0.0) * max(b.height, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


__all__ = [
    "detect_regions",
    "detect_regions_from_text_items",
    "best_match",
    "iou",
]
