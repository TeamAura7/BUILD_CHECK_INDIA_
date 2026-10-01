"""
Vectorized-text recovery and dimension-evidence association for DXF sheets
that have NO native TEXT/MTEXT/DIMENSION entities at all -- confirmed
directly on a real production file (19,622 POLYLINE entities, all layer
'0', zero text entities of any kind): every printed dimension, label and
digit on the sheet was vectorized into bare polyline strokes with no
semantic tagging whatsoever.

Geometry-only reconstruction (region clustering, envelope fitting) cannot
recover WHAT NUMBER was actually printed on the sheet -- it can only guess
a boundary's shape from scattered fragments. This module recovers the
sheet's own printed numbers by rendering the vector geometry to a raster
image and running OCR on it (the same, already-hardened OCR stack this
project uses for scanned PDF pages), then associates each recovered
numeric annotation with the nearest plausible line-like geometry using the
EXACT SAME evidence-quality scoring already proven for native PDF
dimension text (`dimension_candidates.detect_dimension_candidates`,
including its hard-won "prose vs. dimension" plausibility gate that keeps
PID numbers, ward numbers and FAR clause numbers out of scale estimation).

This is deliberately OCR-on-a-render, not a from-scratch vector-glyph
classifier: rendering strokes to pixels and reading them is exactly what
tesseract already does reliably for scanned pages, and it produces the
same evidence shape (raw text + bbox + confidence) that the rest of the
pipeline already knows how to validate, associate and reject -- consistent
with the project's own rule that recovered text is EVIDENCE, never
authoritative on its own.
"""
from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from backend.cv_extraction.dimension_candidates import detect_dimension_candidates
from backend.cv_extraction.dxf_render import RasterTransform, render_polylines_to_image
from backend.cv_extraction.raw_types import DimensionCandidate, RawLine, RawTextItem, SourceKind
from backend.schemas.geometry import BoundingBox, Line, Point

Point2 = tuple[float, float]
Segment2 = tuple[Point2, Point2]

# Metres-per-unit for every unit `dimension_candidates.detect_dimension_
# candidates` can normalize a recognized annotation to. "ft_in" is already
# expressed in decimal feet by that module's feet-inches parsing.
_METRES_PER_UNIT = {
    "mm": 0.001, "cm": 0.01, "m": 1.0,
    "ft": 0.3048, "ft_in": 0.3048, "in": 0.0254,
}

# A recovered annotation is only associated with geometry within this
# fraction of the rendered region's own diagonal -- proximity relative to
# the drawing's own scale, never an absolute drawing-unit distance (which
# would be meaningless before the unit/scale is even known).
_PROXIMITY_FRACTION_OF_DIAGONAL = 0.06


@dataclass
class RecoveredDimension:
    """One OCR-recovered numeric annotation, associated with the nearest
    plausible line-like geometry it might be measuring."""

    value: float
    unit_hint: Optional[str]
    raw_text: str
    confidence: float
    world_bbox: tuple[float, float, float, float]
    associated_segment_index: Optional[int]
    associated_segment_length: Optional[float]

    @property
    def value_metres(self) -> Optional[float]:
        if self.unit_hint is None:
            return None
        factor = _METRES_PER_UNIT.get(self.unit_hint)
        return None if factor is None else self.value * factor


def _import_tesseract():
    import pytesseract

    try:
        from backend.config import get_settings

        cmd = get_settings().tesseract_cmd
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
    except Exception:
        pass
    return pytesseract


def _ocr_raw_text_items(
    polygon_points: Sequence[Sequence[Point2]],
    open_chain_points: Sequence[Sequence[Point2]],
    render_target_px: int,
) -> list[RawTextItem]:
    """Render the given (unscaled, raw drawing-unit) geometry to a raster
    image and OCR it ONCE, returning every recognized raw text item (bbox +
    confidence) before any dimension-shape filtering. Shared by
    `recover_dimension_evidence` and `recover_dimension_and_caption_evidence`
    so more than one kind of evidence (a measurement, a drawing's own
    printed caption/title) can be read off the SAME OCR pass instead of
    paying for a second render+OCR call.

    Returns an empty list (never raises) when OCR is unavailable or finds
    nothing.
    """
    if not polygon_points and not open_chain_points:
        return []
    try:
        pytesseract = _import_tesseract()
        from PIL import Image
    except Exception:
        return []

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "region.png"
        transform = render_polylines_to_image(
            polygon_points, open_chain_points, image_path, target_max_px=render_target_px
        )
        if transform is None:
            return []
        try:
            with Image.open(image_path) as image:
                image.load()
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            return []

    text_items: list[RawTextItem] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w <= 0 or h <= 0:
            continue
        min_x, min_y, max_x, max_y = transform.pixel_bbox_to_world_bbox((x, y, x + w, y + h))
        text_items.append(RawTextItem(
            text=text,
            bounding_box=BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y),
            page=0,
            source=SourceKind.OCR,
            ocr_confidence=max(0.0, min(1.0, conf / 100.0)),
        ))
    return text_items


def _dimensions_from_text_items(
    text_items: Sequence[RawTextItem],
    edge_segments: Sequence[Segment2],
    polygon_points: Sequence[Sequence[Point2]],
    open_chain_points: Sequence[Sequence[Point2]],
) -> list[RecoveredDimension]:
    if not text_items:
        return []

    raw_lines = [
        RawLine(line=Line(start=Point(x=s[0], y=s[1]), end=Point(x=e[0], y=e[1])), page=0, source=SourceKind.OPENCV_RASTER)
        for s, e in edge_segments
    ]

    all_pts = [p for chain in polygon_points for p in chain] + [p for chain in open_chain_points for p in chain]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if all_pts else 0.0
    proximity = max(diagonal * _PROXIMITY_FRACTION_OF_DIAGONAL, 1e-6)

    candidates = detect_dimension_candidates(text_items, raw_lines, proximity_pts=proximity)

    recovered: list[RecoveredDimension] = []
    for c in candidates:
        seg_idx = c.nearby_geometry_ids[0] if c.nearby_geometry_ids else None
        seg_len = raw_lines[seg_idx].line.length if seg_idx is not None else None
        recovered.append(RecoveredDimension(
            value=c.numeric_value,
            unit_hint=c.unit_hint,
            raw_text=c.raw_text,
            confidence=c.confidence,
            world_bbox=(c.bounding_box.min_x, c.bounding_box.min_y, c.bounding_box.max_x, c.bounding_box.max_y),
            associated_segment_index=seg_idx,
            associated_segment_length=seg_len,
        ))
    return recovered


def recover_dimension_evidence(
    polygon_points: Sequence[Sequence[Point2]],
    open_chain_points: Sequence[Sequence[Point2]],
    edge_segments: Sequence[Segment2],
    render_target_px: int = 2400,
) -> list[RecoveredDimension]:
    """
    Render the given (unscaled, raw drawing-unit) geometry to a raster
    image, OCR it, and return every recognized numeric annotation
    associated with its nearest plausible edge from `edge_segments`.

    `edge_segments` should be the region's own candidate-relevant edges
    (e.g. a candidate plot/building polygon's boundary segments, or the
    region's open wall/line-work) -- association is scored against exactly
    these, not arbitrary geometry, so a recovered value can only ever be
    read as measuring one of the spans the caller actually cares about.

    Returns an empty list (never raises) when OCR is unavailable or finds
    nothing -- this is best-effort evidence recovery, not a required step.
    """
    text_items = _ocr_raw_text_items(polygon_points, open_chain_points, render_target_px)
    return _dimensions_from_text_items(text_items, edge_segments, polygon_points, open_chain_points)


@dataclass
class RecoveredCaption:
    """One OCR-recovered piece of text that matched a known drawing-type
    caption pattern (e.g. "SITE PLAN") -- evidence of what KIND of drawing
    a region is, not a measurement."""

    text: str
    world_bbox: tuple[float, float, float, float]
    confidence: float


# How many text-item-heights apart two recognized words may be and still
# count as the SAME caption. A multi-word drawing caption ("SITE PLAN",
# "KEY MAP") is very often printed rotated/stacked one word per line rather
# than as a single horizontal run -- confirmed directly: tesseract read a
# real sheet's "SITE PLAN" caption as two entirely separate text items,
# "SITE" and "PLAN", vertically stacked with touching bounding boxes, not
# one joined string. Scaling the proximity check to each PAIR's own text
# size (rather than a fixed world-unit distance, or a fraction of the whole
# sheet's diagonal) keeps this correct regardless of the drawing's own
# scale or how large the sheet is.
_CAPTION_WORD_PROXIMITY_IN_TEXT_HEIGHTS = 4.0


def _bbox_diagonal(bbox: BoundingBox) -> float:
    return math.hypot(bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y)


def _bbox_center_distance(a: BoundingBox, b: BoundingBox) -> float:
    return math.hypot(
        (a.min_x + a.max_x) / 2.0 - (b.min_x + b.max_x) / 2.0,
        (a.min_y + a.max_y) / 2.0 - (b.min_y + b.max_y) / 2.0,
    )


def _union_bbox(boxes: Sequence[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        min_x=min(b.min_x for b in boxes), min_y=min(b.min_y for b in boxes),
        max_x=max(b.max_x for b in boxes), max_y=max(b.max_y for b in boxes),
    )


# A genuine drawing caption/title is (usually) a short, standalone label
# -- "SITE PLAN", "SITE PLAN SCALE 1:200" -- not a sentence. Without an
# isolation check, a keyword group like ("SITE", "PLAN") would equally
# match a throwaway PROSE mention elsewhere on the sheet ("REFER TO SITE
# PLAN FOR SETBACK DETAILS", a note inside an unrelated floor-plan
# region), wrongly handing that region the same large structural-score
# bonus meant to single out the actual site plan.
#
# TODO: calibrate against a wider ground-truth corpus -- and read this
# note before raising or lowering this number. An initial guess of 6 was
# tested directly against PLAN5's own real caption and FAILED: measured
# 11 other recognized text items within this same radius (site numbers,
# room labels, schedule-table entries -- a real, compact site-plan
# sub-drawing packs a lot of short labels into a small area, it is not
# sparse the way a "title standing alone" intuition assumes). 15 is set
# to clear that measured real case with some margin, NOT derived from any
# confirmed false-positive example -- none exists in the current 2-file
# corpus to measure against. This means the check, as it stands, only
# catches a sentence considerably denser than PLAN5's own busy caption
# area; it is closer to a loose sanity backstop than a validated
# discriminator, and its comment should be treated as more important than
# its number until a real prose-embedded-caption false positive is found
# and measured the same way this true positive was.
MAX_NEARBY_UNRELATED_WORDS_FOR_ISOLATED_CAPTION = 15

# How far around a candidate match to look for OTHER nearby text when
# judging isolation, scaled to the match's own size the same way proximity
# is scaled elsewhere in this module (never an absolute drawing-unit
# distance).
_CAPTION_ISOLATION_RADIUS_IN_TEXT_HEIGHTS = 3.0


def _is_isolated_caption_match(
    combo: Sequence[RawTextItem], all_text_items: Sequence[RawTextItem], own_scale: float,
) -> bool:
    """Refuse rather than guess: count OTHER recognized text items (not
    part of this match itself) within `_CAPTION_ISOLATION_RADIUS_IN_TEXT_
    HEIGHTS` of the match's own centroid. Too many suggests this sits
    inside a longer run of prose (a paragraph, a notes block) rather than
    standing alone as a title -- exactly the shape item 1's glyph-swarm
    check independently flags at the REGION level; this is the same idea
    applied locally, for a caption match that might sit in an otherwise
    normal (non-swarm) region."""
    combo_ids = {id(item) for item in combo}
    boxes = [c.bounding_box for c in combo]
    union = _union_bbox(boxes)
    cx, cy = (union.min_x + union.max_x) / 2.0, (union.min_y + union.max_y) / 2.0
    radius = max(own_scale, 1e-6) * _CAPTION_ISOLATION_RADIUS_IN_TEXT_HEIGHTS
    nearby_unrelated = 0
    for item in all_text_items:
        if id(item) in combo_ids:
            continue
        b = item.bounding_box
        icx, icy = (b.min_x + b.max_x) / 2.0, (b.min_y + b.max_y) / 2.0
        if math.hypot(icx - cx, icy - cy) <= radius:
            nearby_unrelated += 1
    return nearby_unrelated <= MAX_NEARBY_UNRELATED_WORDS_FOR_ISOLATED_CAPTION


def _find_caption_matches(
    text_items: Sequence[RawTextItem], keyword_groups: Sequence[Sequence[str]],
) -> list[RecoveredCaption]:
    """For each keyword group (e.g. `("SITE", "PLAN")`), find one item per
    keyword (case-insensitive whole-word match) such that every matched
    item lies within `_CAPTION_WORD_PROXIMITY_IN_TEXT_HEIGHTS` text-heights
    of every other -- covering both a caption OCR'd as one combined string
    (the same item matches more than one keyword, distance zero) and one
    OCR'd as separate stacked/adjacent words (the common case for a
    rotated multi-word title) -- AND is reasonably ISOLATED (see
    `_is_isolated_caption_match`), so a keyword pair embedded in an
    ordinary sentence elsewhere on the sheet isn't mistaken for a genuine
    drawing title. Returns at most one match per group.
    """
    import itertools

    results: list[RecoveredCaption] = []
    for keywords in keyword_groups:
        patterns = [re.compile(rf"\b{re.escape(kw)}\b", re.I) for kw in keywords]
        matches_per_keyword = [
            [item for item in text_items if pattern.search(item.text)] for pattern in patterns
        ]
        if any(not matches for matches in matches_per_keyword):
            continue  # at least one keyword in this group was never recognized anywhere on the sheet
        for combo in itertools.product(*matches_per_keyword):
            boxes = [c.bounding_box for c in combo]
            own_scale = max((_bbox_diagonal(b) for b in boxes), default=0.0)
            threshold = max(own_scale, 1e-6) * _CAPTION_WORD_PROXIMITY_IN_TEXT_HEIGHTS
            if not all(
                _bbox_center_distance(a, b) <= threshold
                for i, a in enumerate(boxes) for b in boxes[i + 1:]
            ):
                continue
            if not _is_isolated_caption_match(combo, text_items, own_scale):
                continue
            union = _union_bbox(boxes)
            results.append(RecoveredCaption(
                text=" ".join(dict.fromkeys(c.text for c in combo)),  # dedupe if the same item matched twice
                world_bbox=(union.min_x, union.min_y, union.max_x, union.max_y),
                confidence=min((c.ocr_confidence or 0.0) for c in combo),
            ))
            break  # one coherent match is enough for this group
    return results


def recover_dimension_and_caption_evidence(
    polygon_points: Sequence[Sequence[Point2]],
    open_chain_points: Sequence[Sequence[Point2]],
    edge_segments: Sequence[Segment2],
    caption_keyword_groups: Sequence[Sequence[str]],
    render_target_px: int = 2400,
) -> tuple[list[RecoveredDimension], list[RecoveredCaption]]:
    """Same OCR pass as `recover_dimension_evidence`, but also scans the
    recognized text for a drawing's own printed caption/title (e.g. "SITE
    PLAN") -- evidence of what KIND of drawing this is, not a measurement.
    Runs OCR exactly once and derives both results from it, at no extra
    render/OCR cost over calling `recover_dimension_evidence` alone.

    `caption_keyword_groups`: each inner sequence is a set of keywords that
    must ALL be found, on the same or nearby text items (see
    `_find_caption_matches`), for that group to count as a match -- e.g.
    `[("SITE", "PLAN"), ("KEY", "MAP")]`.
    """
    text_items = _ocr_raw_text_items(polygon_points, open_chain_points, render_target_px)
    dimensions = _dimensions_from_text_items(text_items, edge_segments, polygon_points, open_chain_points)
    captions = _find_caption_matches(text_items, caption_keyword_groups)
    return dimensions, captions


def scale_candidates_from_recovered_dimensions(
    recovered: Sequence[RecoveredDimension],
) -> list[tuple[float, RecoveredDimension]]:
    """
    Derive (scale_factor, evidence) pairs -- metres per raw drawing unit --
    from recovered annotations that have BOTH an explicit unit and a
    confident geometric association. Each pair is independent evidence for
    the sheet's true scale, measured from a printed number rather than
    guessed from a plausible-area heuristic.
    """
    out: list[tuple[float, RecoveredDimension]] = []
    for r in recovered:
        value_m = r.value_metres
        if value_m is None or not value_m > 0:
            continue
        if r.associated_segment_length is None or r.associated_segment_length <= 1e-9:
            continue
        out.append((value_m / r.associated_segment_length, r))
    return out


__all__ = [
    "RecoveredDimension",
    "RecoveredCaption",
    "recover_dimension_evidence",
    "recover_dimension_and_caption_evidence",
    "scale_candidates_from_recovered_dimensions",
]
