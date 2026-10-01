"""
Unit tests for `dxf_glyph_swarm.py` (DXF_FAILURE_TAXONOMY.md item 1).

These pin the SHAPE of the row-banding/classification mechanism using
synthetic bounding boxes -- not calibrated against any real file. The
module's own thresholds are explicit placeholders (see its `# TODO`
comments); these tests confirm the mechanism distinguishes the qualitative
patterns it's meant to (stacked text lines vs. a boundary-shaped scatter),
not that any specific real DXF is classified correctly.
"""
from __future__ import annotations

from backend.cv_extraction.dxf_glyph_swarm import (
    classify_glyph_swarm,
    compute_glyph_swarm_signal,
)
from backend.schemas.geometry import BoundingBox


def _glyph_bbox(x: float, y: float, w: float = 0.15, h: float = 0.2) -> BoundingBox:
    return BoundingBox(min_x=x, min_y=y, max_x=x + w, max_y=y + h)


def _text_paragraph_bboxes(lines: int = 25, glyphs_per_line: int = 25, line_height: float = 0.25) -> list[BoundingBox]:
    """Synthetic vectorized-text-paragraph shape: many lines, each with
    many same-height glyphs spread across a row, consistent row spacing --
    the shape actually observed on PLAN6's real notes/table swarms."""
    boxes = []
    for line_idx in range(lines):
        y = line_idx * line_height
        for glyph_idx in range(glyphs_per_line):
            boxes.append(_glyph_bbox(x=glyph_idx * 0.18, y=y))
    return boxes


def _dash_dot_boundary_bboxes(width: float = 15.0, depth: float = 10.0, dash_count_per_side: int = 40) -> list[BoundingBox]:
    """Synthetic dash-dot property-boundary shape: small fragments traced
    around a rectangle's PERIMETER, not stacked into horizontal lines --
    the shape a legitimate boundary region should NOT be excluded for."""
    boxes = []
    for i in range(dash_count_per_side):
        t = i / dash_count_per_side
        # bottom and top edges
        boxes.append(_glyph_bbox(x=t * width, y=0.0, w=0.1, h=0.05))
        boxes.append(_glyph_bbox(x=t * width, y=depth, w=0.1, h=0.05))
        # left and right edges
        boxes.append(_glyph_bbox(x=0.0, y=t * depth, w=0.05, h=0.1))
        boxes.append(_glyph_bbox(x=width, y=t * depth, w=0.05, h=0.1))
    return boxes


def test_text_paragraph_shape_is_classified_as_a_likely_swarm():
    signal = compute_glyph_swarm_signal(_text_paragraph_bboxes())
    is_swarm, reason = classify_glyph_swarm(signal)
    assert is_swarm, reason
    assert signal.row_band_count == 25


def _hatch_pattern_bboxes(rows: int = 70, ticks_per_row: int = 8, row_height: float = 0.1) -> list[BoundingBox]:
    """Synthetic diagonal-hatch-fill shape: many small fragments that
    happen to line up into rows purely from the hatch pattern's own
    periodicity, but with only a HANDFUL of ticks per row -- not a line of
    text. Root cause this pins: confirmed directly on PLAN5's real wall-
    SECTION-view masonry hatching (region0), which organized into 52
    row-bands with only a median of 7 polygons each -- a false positive an
    earlier, row-banding-only version of this classifier produced."""
    boxes = []
    for row_idx in range(rows):
        y = row_idx * row_height
        for tick_idx in range(ticks_per_row):
            boxes.append(_glyph_bbox(x=tick_idx * 0.3, y=y, w=0.05, h=0.05))
    return boxes


def test_hatch_pattern_shape_is_not_classified_as_a_swarm():
    """The median-polygons-per-band gate exists specifically for this:
    hatch fill (or wall/dimension-fragment reconstruction artifacts) can
    organize into just as many row-bands as real text, but with far fewer
    polygons in each one -- a real text line has many glyphs across it."""
    signal = compute_glyph_swarm_signal(_hatch_pattern_bboxes())
    is_swarm, reason = classify_glyph_swarm(signal)
    assert not is_swarm, reason
    assert "polygons per populated row-band" in reason


def test_dash_dot_boundary_shape_is_not_classified_as_a_swarm():
    """The whole point of this mechanism existing alongside item 2's
    agglomerative merge growth: it must never exclude a genuine fragmented
    boundary region just because it has many small polygons -- only a
    region actually organized into text-line rows."""
    signal = compute_glyph_swarm_signal(_dash_dot_boundary_bboxes())
    is_swarm, reason = classify_glyph_swarm(signal)
    assert not is_swarm, reason


def test_a_handful_of_polygons_refuses_to_classify_either_way():
    """Refuse rather than guess: too little data to say anything
    meaningful must never come back as a confident "yes, this is a
    swarm" -- only as an explicit low-count refusal."""
    signal = compute_glyph_swarm_signal([_glyph_bbox(x=i * 0.2, y=0.0) for i in range(5)])
    is_swarm, reason = classify_glyph_swarm(signal)
    assert not is_swarm
    assert "below" in reason and "floor" in reason


def test_a_short_one_line_caption_is_not_classified_as_a_swarm():
    """A single line (or two) of real text -- e.g. a genuine drawing
    caption sitting alone -- must not be excluded as a "paragraph swarm":
    this check is about multi-line blocks specifically (see
    MIN_ROW_BAND_COUNT), a one-line caption is exactly what item 0's
    caption-recognition mechanism is meant to find and use, not discard."""
    # Enough polygons to clear the count floor, but all on one line.
    boxes = [_glyph_bbox(x=i * 0.18, y=0.0) for i in range(600)]
    signal = compute_glyph_swarm_signal(boxes)
    is_swarm, reason = classify_glyph_swarm(signal)
    assert not is_swarm, reason


def test_height_variation_is_reported_but_does_not_block_classification():
    """Height-coefficient-of-variation is deliberately informational only,
    not a gate (see `_HEIGHT_CV_IS_INFORMATIONAL_ONLY`'s own comment):
    tested directly against PLAN6's own two real, confirmed text swarms
    (not just synthetic data), both measured well above an initially-
    guessed uniformity ceiling (0.78 and 0.63) despite being unambiguous
    vectorized text by every other signal -- real text routinely mixes a
    heading/label size with body text or table values within one block. A
    swarm shape with high height variation must still classify as a swarm
    (on row-banding alone), with the variation surfaced in the reason
    string for auditability, not silently hidden."""
    boxes = []
    for line_idx in range(25):
        y = line_idx * 0.3
        for glyph_idx in range(50):
            # Alternate a "heading-sized" and "body-sized" glyph within the
            # same line -- mimicking a real mixed-size text block, not a
            # single uniform font size. Enough glyphs per line that the
            # resulting per-line split (mixed heights shift each glyph's
            # own center-Y differently) still leaves each half comfortably
            # above the real-text band-density floor.
            h = 0.15 if glyph_idx % 2 == 0 else 0.4
            boxes.append(_glyph_bbox(x=glyph_idx * 0.2, y=y, h=h))
    signal = compute_glyph_swarm_signal(boxes)
    is_swarm, reason = classify_glyph_swarm(signal)
    assert is_swarm, reason
    assert "informational only" in reason
