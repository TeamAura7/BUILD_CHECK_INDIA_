"""
Text-glyph-swarm detection (DXF_FAILURE_TAXONOMY.md item 1).

Root cause this targets: a DXF with zero native TEXT/MTEXT entities (every
glyph exploded to closed polyline curves before export -- confirmed on real
bundled fixtures) can turn an ordinary paragraph of text, or a data table,
into a block of thousands of small, near-uniform, closed polygons. Density-
based region clustering has no notion of "this is text" -- it only sees
"many small closed shapes, tightly packed," which is also what a legitimate
dash-dot boundary or a hatch fill pattern looks like. A big enough glyph
swarm can dominate structural scoring by fragment count/envelope size alone,
confirmed directly: PLAN6's "General Conditions" notes paragraph (15,961
polygons) and its "AREA STATEMENT" table (1,933 polygons) both out-scored
the sheet's genuine dash-dot site-plan boundary.

The signal genuinely specific to TEXT (as opposed to a boundary or hatch)
is that printed text -- when not rotated -- is laid out in discrete
horizontal LINES: many glyphs sharing a baseline, at a fairly consistent
height, with a real gap to the next line. A dash-dot boundary traces a
polygon's perimeter (its fragments' centers trace a rectangle/quadrilateral
outline, not stacked horizontal rows); a hatch fill is either a dense
uniform grid or repeating diagonal strokes, not organized into text-line
bands with word/letter-like horizontal spread within each band.

This module computes that row-banding signal (and a few corroborating
ones) and exposes a classification decision. The classification THRESHOLDS
are explicitly marked as unvalidated placeholders (see the `# TODO`
comments below) -- calibrate them against a wider ground-truth corpus
before trusting them on a file meaningfully different from the two real
fixtures this was designed against. The intent is that this module's
INTERFACE and the signals it computes do not need to change when that
calibration happens -- only the cutoff constants do.

Known limitation, not attempted here: this only detects HORIZONTAL row
structure, so a genuine text block rotated 90 degrees (e.g. a vertical
caption or a title block's sideways paragraph) will not band correctly
along the Y axis and may score as "not a swarm" even when it is one. A
rotation-aware version would need to estimate the region's own dominant
text orientation first (e.g. via PCA of polygon centers) and band along
that axis instead -- real future work, not guessed at here.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from backend.schemas.geometry import BoundingBox


@dataclass
class GlyphSwarmSignal:
    """Per-region signals relevant to recognizing a text-glyph swarm. Not a
    verdict on its own -- combined by `classify_glyph_swarm` into a single
    decision, kept separate so each signal stays independently inspectable/
    testable and auditable in a warning message."""

    polygon_count: int
    median_polygon_height: float
    height_coefficient_of_variation: float  # low = glyphs share a consistent line-height (font size)
    row_band_count: int
    polygons_in_populated_bands_fraction: float  # fraction of ALL polygons belonging to a real "line", not a scattered outlier
    median_polygons_per_populated_band: float


def _row_bands(centers_y: Sequence[float], median_height: float, gap_factor: float) -> list[list[int]]:
    """Cluster 1-D Y-centers into bands (indices into `centers_y`, per
    band): consecutive values (sorted) are the same band while the gap
    between them stays under `gap_factor * median_height`; a bigger gap
    starts a new band. `gap_factor` is the ONE free parameter here -- how
    many "line heights" of vertical gap still counts as the same text line
    versus a new one -- deliberately passed in rather than hardcoded so the
    caller's calibration lives in one place (see `_ROW_BAND_GAP_FACTOR`
    below).
    """
    if not centers_y:
        return []
    order = sorted(range(len(centers_y)), key=lambda i: centers_y[i])
    threshold = max(median_height, 1e-9) * gap_factor
    bands: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        if centers_y[idx] - centers_y[bands[-1][-1]] <= threshold:
            bands[-1].append(idx)
        else:
            bands.append([idx])
    return bands


def compute_glyph_swarm_signal(
    polygon_bboxes: Sequence[BoundingBox], gap_factor: float = 0.6,
) -> GlyphSwarmSignal:
    """Compute the row-banding and size-consistency signals for one
    region's own polygon bounding boxes. Pure geometry -- no DXF-specific
    types, no thresholds baked in, so it stays testable with synthetic
    bboxes and reusable if this ever needs to run on a differently-shaped
    caller."""
    n = len(polygon_bboxes)
    if n == 0:
        return GlyphSwarmSignal(0, 0.0, 0.0, 0, 0.0, 0.0)

    heights = [b.height for b in polygon_bboxes]
    median_height = statistics.median(heights)
    height_cv = (statistics.pstdev(heights) / median_height) if median_height > 1e-9 else 0.0

    centers_y = [(b.min_y + b.max_y) / 2.0 for b in polygon_bboxes]
    bands = _row_bands(centers_y, median_height, gap_factor)

    # TODO: calibrate against a wider ground-truth corpus (see
    # DXF_FAILURE_TAXONOMY.md item 1) -- this is a placeholder for "how
    # many polygons in one band counts as a real text LINE, not a couple
    # of coincidentally-aligned unrelated fragments." 3 is chosen only as
    # a plausible floor (a real line of text is rarely just one or two
    # glyphs), not tuned to either real fixture.
    min_polygons_per_line = 3

    populated_bands = [band for band in bands if len(band) >= min_polygons_per_line]
    polygons_in_populated_bands = sum(len(band) for band in populated_bands)

    return GlyphSwarmSignal(
        polygon_count=n,
        median_polygon_height=median_height,
        height_coefficient_of_variation=height_cv,
        row_band_count=len(bands),
        polygons_in_populated_bands_fraction=polygons_in_populated_bands / n,
        median_polygons_per_populated_band=(
            statistics.median(len(band) for band in populated_bands) if populated_bands else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Classification. Every constant below is an unvalidated placeholder -- see
# each comment. Chosen only to be permissive enough not to exclude a
# legitimate boundary/hatch region and strict enough to catch the two
# extreme, already-confirmed swarms (15,961 and 1,933 polygons, organized
# into dozens of populated row-bands) -- NOT tuned against a wider corpus.
# Re-derive these once more real files are available (see
# DXF_FAILURE_TAXONOMY.md item 1's "how to use this" section), the same way
# `_SITE_PLAN_CAPTION_SCORE_BONUS` was re-derived from measured score
# ranges rather than picked by feel.
# ---------------------------------------------------------------------------

# TODO: calibrate. A region with fewer polygons than this is never worth
# examining as a potential swarm at all -- both confirmed real swarms have
# thousands; a small dash-dot fragment cluster commonly has dozens to a
# few hundred (see PLAN5's regions, all under 1000).
MIN_POLYGON_COUNT_FOR_SWARM_CONSIDERATION = 500

# TODO: calibrate. A genuine multi-line block of text should organize a
# large majority of its own polygons into real row-bands (not be mostly
# scattered singletons) -- both confirmed swarms exceed 90%.
MIN_POPULATED_BAND_FRACTION = 0.7

# TODO: calibrate. At least this many distinct text lines -- a one- or
# two-line label (e.g. a genuine drawing caption) should NOT be classified
# as a swarm by this alone; a swarm is a PARAGRAPH or a TABLE, many lines.
MIN_ROW_BAND_COUNT = 4

# TODO: calibrate -- this one matters a lot, see the comment below.
# Row-banding ALONE turned out not to be specific to text: tested directly
# against PLAN5's real fixture, its wall-SECTION-view masonry hatching
# (region0) and its floor-plan wall/dimension-fragment regions (region1,
# region2) ALL organize into dozens of row-bands too (52, 53, 58 bands,
# 96-98% populated) purely from hatch periodicity and reconstruction
# fragment alignment -- a false positive this module must not produce.
# The signal that actually separates them: a real text LINE has many
# glyphs across it (a line of words) -- PLAN6's two confirmed real swarms
# measured a median of 113 and 40 polygons per populated band, while
# PLAN5's three false positives measured only 7, 8.5, and 12.5 -- an
# order-of-magnitude gap. 20 is chosen as a conservative floor comfortably
# above the false-positive range and comfortably below the true-positive
# range measured so far, NOT a statistically fitted cutoff -- it reflects
# a gap seen in exactly two files' worth of examples and must be
# re-measured against a wider corpus before being trusted on a file where
# that gap might not be so clean (e.g. a sparser table with genuinely
# short rows, or a hatch pattern denser than either of these two).
MIN_MEDIAN_POLYGONS_PER_POPULATED_BAND = 20

# Height consistency is deliberately NOT a hard gate below (see
# `classify_glyph_swarm`'s docstring) -- kept only as an informational
# figure in the returned reason string. Tested directly against PLAN6's
# own two real, confirmed swarms (not just synthetic data): a "General
# Conditions" notes paragraph measured height CV=0.78, and an "AREA
# STATEMENT" table measured CV=0.63 -- both real vectorized-text blocks,
# both well above an initially-guessed 0.5 ceiling, because real text
# routinely mixes a heading/label size with body text or table values
# within one block. Requiring single-font-size uniformity across an
# entire multi-line block turned out to be the wrong shape of check, not
# just a wrong number -- row-banding organization (MIN_ROW_BAND_COUNT,
# MIN_POPULATED_BAND_FRACTION) is the far more reliable signal, and both
# real swarms clear it overwhelmingly (176 bands/99.98% populated; 44
# bands/100% populated). Recording this measured data point for whoever
# calibrates this next, rather than silently tuning a number to make two
# known files pass -- if a wider corpus later shows height-CV IS a useful
# discriminator after all (e.g. against a hatch pattern that also happens
# to band into rows), reintroduce it as a gate then, with real numbers.
_HEIGHT_CV_IS_INFORMATIONAL_ONLY = True


def classify_glyph_swarm(signal: GlyphSwarmSignal) -> tuple[bool, str]:
    """Returns (is_likely_swarm, reason) -- reason is always populated,
    including the negative case, so a decision (exclude or don't) is
    auditable rather than a silent boolean. Refuses to guess when the
    polygon count alone is too low to say anything meaningful either way
    (returns False with an explicit "not enough polygons to judge" reason,
    never a confident classification off too little data).

    Deliberately does NOT gate on height-coefficient-of-variation -- see
    `_HEIGHT_CV_IS_INFORMATIONAL_ONLY`'s own comment for why that started
    as a fourth hard requirement and was demoted to informational after
    testing against real data, not synthetic fixtures alone.
    """
    if signal.polygon_count < MIN_POLYGON_COUNT_FOR_SWARM_CONSIDERATION:
        return False, (
            f"only {signal.polygon_count} polygons -- below the "
            f"{MIN_POLYGON_COUNT_FOR_SWARM_CONSIDERATION}-polygon floor this check considers at all"
        )
    if signal.row_band_count < MIN_ROW_BAND_COUNT:
        return False, (
            f"{signal.row_band_count} row-band(s) found -- fewer than the {MIN_ROW_BAND_COUNT} a "
            "multi-line paragraph/table would produce (could be a short label, or content that "
            "doesn't organize into horizontal lines at all, e.g. a rotated caption -- see this "
            "module's own known-limitation note)"
        )
    if signal.polygons_in_populated_bands_fraction < MIN_POPULATED_BAND_FRACTION:
        return False, (
            f"only {signal.polygons_in_populated_bands_fraction:.0%} of polygons fall into a populated "
            f"row-band -- below the {MIN_POPULATED_BAND_FRACTION:.0%} floor expected of genuine text lines "
            "(most polygons here are scattered rather than lined up)"
        )
    if signal.median_polygons_per_populated_band < MIN_MEDIAN_POLYGONS_PER_POPULATED_BAND:
        return False, (
            f"only a median of {signal.median_polygons_per_populated_band:.0f} polygons per populated "
            f"row-band -- below the {MIN_MEDIAN_POLYGONS_PER_POPULATED_BAND:.0f} floor a real line of text "
            "(many glyphs across a line of words) would produce; this many small polygons organizing "
            "into rows is consistent with hatch-fill periodicity or wall/dimension-fragment reconstruction "
            "artifacts lining up by coincidence, not text"
        )
    return True, (
        f"{signal.polygon_count} polygons organized into {signal.row_band_count} row-bands, "
        f"{signal.polygons_in_populated_bands_fraction:.0%} of them in a populated band (median "
        f"{signal.median_polygons_per_populated_band:.0f} polygons/band; height CV="
        f"{signal.height_coefficient_of_variation:.2f}, informational only) -- consistent with "
        "vectorized text lines, not a boundary or hatch pattern."
    )
