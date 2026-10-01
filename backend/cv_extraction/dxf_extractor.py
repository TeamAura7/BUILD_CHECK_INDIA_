"""
DXFHybridExtractor -- deterministic vector-geometry `GeometryExtractor` for
DXF site plans.

Unlike `PDFHybridExtractor` (rasterize -> OpenCV line/contour detection ->
OCR -> scale calibration -> guess), a DXF is not a picture of a drawing --
it IS the drawing's data. Every boundary, wall and label is stored as exact
floating-point coordinates and strings, not pixels. This module never
rasterizes, never runs OpenCV/Hough detection, and never OCRs anything: it
reads `ezdxf` entities directly and turns them into the same
`IndependentCVResult` / `ExtractionResult` shapes the PDF pipeline produces,
so everything downstream (spatial_reasoning, compliance engine, frontend)
is completely unaware of which source format was used.

Design choice: this extractor deliberately returns EMPTY
plot_candidates/building_candidates/road_candidates. Those lists (and the
legacy `score_plot_candidates` resolver in spatial_reasoning/pipeline.py)
assume PDF-page-space geometry with a points-per-metre scale that has to be
estimated from printed dimension labels -- a PDF-specific problem that does
not exist for DXF, where coordinates are already in the drawing's real-world
units. Returning no candidates makes `build_normalized_plan` take the
existing "independent CV" fallback path (already wired for exactly this
situation -- see spatial_reasoning/pipeline.py), building the NormalizedPlan
directly from `independent_cv.measurements`, which this module populates
with values read straight from DXF vector geometry.

The resolver is conservative in the same spirit as the PDF site-plan
resolver (backend/cv_extraction/site_plan.py): if a field's evidence is
not strong enough, it is left out of `measurements` (-> MISSING downstream)
rather than guessed.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from backend.cv_extraction.dxf_glyph_swarm import classify_glyph_swarm, compute_glyph_swarm_signal
from backend.cv_extraction.interfaces import GeometryExtractor
from backend.schemas.enums import ConfidenceLevel, DocumentType, SourceType
from backend.schemas.evidence import TextEvidence
from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import BoundingBox, Dimension, Line, Point, Polygon
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.front_side import resolve_front_side
from backend.spatial_reasoning.road_access import collect_access_evidence
from backend.tools import bounded_execution
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

EXTRACTOR_NAME = "backend.cv_extraction.dxf_extractor.DXFHybridExtractor"
EXTRACTOR_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# DXF header $INSUNITS codes -> metres per drawing unit. Only the codes that
# plausibly show up on an architectural/site drawing are listed; anything
# else falls through to the sanity-check path below.
_INSUNITS_TO_METRES = {
    1: 0.0254,      # Inches
    2: 0.3048,      # Feet
    3: 1609.344,    # Miles
    4: 0.001,       # Millimeters
    5: 0.01,        # Centimeters
    6: 1.0,         # Meters
    7: 1000.0,      # Kilometers
    10: 0.9144,     # Yards
    14: 0.1,        # Decimeters
    15: 10.0,       # Decameters
    16: 100.0,      # Hectometers
}

# A plausible real-world plot area, used to sanity-check (and, when
# $INSUNITS is 0/missing, to guess) the drawing's unit scale -- same spirit
# as PDF's default-dimension-unit detection in pdf_extractor.py.
_PLAUSIBLE_PLOT_AREA_M2 = (20.0, 20000.0)

_CANDIDATE_UNIT_FACTORS = [
    ("metres (native)", 1.0),
    ("millimetres", 0.001),
    ("centimetres", 0.01),
    ("feet", 0.3048),
    ("inches", 0.0254),
]


# How much to trust the resolved scale itself, separate from how much any
# individual measurement's own source (exact vector geometry vs. a
# reconstructed/Vision-guessed polygon) is trusted. Every length/area
# measurement derived from DXF coordinates is scaled by this factor, so an
# uncertain unit resolution must cap those measurements' confidence too --
# a hardcoded 0.95/0.98 regardless of how the scale was determined would
# silently claim precision the extractor does not actually have (see
# Phase 6/16 of the DXF audit: "do not hardcode confidence when evidence is
# actually ambiguous").
_UNIT_CONFIDENCE_HEADER_VALIDATED = 0.98
_UNIT_CONFIDENCE_HEADER_ALONE = 0.90
_UNIT_CONFIDENCE_CANDIDATE_SANITY_CHECKED = 0.60
_UNIT_CONFIDENCE_HEADER_DESPITE_IMPLAUSIBLE = 0.50
_UNIT_CONFIDENCE_UNRESOLVED_DEFAULT = 0.20


def _resolve_unit_factor(
    insunits: Optional[int], largest_raw_area: Optional[float]
) -> tuple[float, str, float]:
    """Resolve drawing-unit -> metre factor, plus how confident that resolution is.

    Returns (factor, reason, unit_confidence). `unit_confidence` must never
    be used to silently produce a HIGH-confidence measurement when the scale
    itself was only guessed -- callers cap each length/area measurement's
    own confidence at this value.
    """
    header_factor = _INSUNITS_TO_METRES.get(insunits) if insunits else None
    lo, hi = _PLAUSIBLE_PLOT_AREA_M2

    if not largest_raw_area or largest_raw_area <= 0:
        if header_factor is not None:
            return (
                header_factor,
                f"$INSUNITS={insunits} -> {header_factor} m/drawing-unit (no closed polygon available to sanity-check against).",
                _UNIT_CONFIDENCE_HEADER_ALONE,
            )
        return (
            1.0,
            "No $INSUNITS header and no closed polygon to calibrate against; defaulting to 1 drawing unit = 1 metre. VERIFY MANUALLY.",
            _UNIT_CONFIDENCE_UNRESOLVED_DEFAULT,
        )

    if header_factor is not None:
        header_area = largest_raw_area * (header_factor ** 2)
        if lo <= header_area <= hi:
            return (
                header_factor,
                f"$INSUNITS={insunits} -> {header_factor} m/drawing-unit; largest closed polygon scales to a plausible plot area ({header_area:.1f} m2).",
                _UNIT_CONFIDENCE_HEADER_VALIDATED,
            )

    for label, factor in _CANDIDATE_UNIT_FACTORS:
        area_m2 = largest_raw_area * (factor ** 2)
        if lo <= area_m2 <= hi:
            reason = (
                f"$INSUNITS ({insunits}) missing or produced an implausible area; "
                f"sanity-checked against candidate scales -- treating drawing units as {label} "
                f"gives a plausible plot area ({area_m2:.1f} m2)."
            )
            return factor, reason, _UNIT_CONFIDENCE_CANDIDATE_SANITY_CHECKED

    if header_factor is not None:
        return (
            header_factor,
            f"$INSUNITS={insunits} -> {header_factor} m/drawing-unit (no candidate scale produced a "
            "plausible plot area; trusting the header value). VERIFY MANUALLY.",
            _UNIT_CONFIDENCE_HEADER_DESPITE_IMPLAUSIBLE,
        )
    return (
        1.0,
        "Could not resolve drawing units from $INSUNITS or from a plausible-plot-area sanity check; "
        "defaulting to 1 drawing unit = 1 metre. VERIFY MANUALLY.",
        _UNIT_CONFIDENCE_UNRESOLVED_DEFAULT,
    )


# ---------------------------------------------------------------------------
# Layer / text classification
# ---------------------------------------------------------------------------

_PLOT_LAYER_RE = re.compile(r"PLOT|BOUNDARY|SITE[-_ ]?BOUND|PROPERTY[-_ ]?LINE|PARCEL", re.I)
_BUILDING_LAYER_RE = re.compile(r"BUILDING|BLDG|FOOTPRINT|STRUCTUR|\bWALL", re.I)
_ROAD_LAYER_RE = re.compile(r"\bROAD\b|\bROW\b|RIGHT[-_ ]?OF[-_ ]?WAY|STREET", re.I)

_ROAD_VALUE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m|mt|mtr|metres?|meters?|ft|feet|foot)?\s*"
    r"(?:WIDE\s+)?R\s*O\s*A\s*D\b",
    re.I,
)
_ROAD_UNIT_TO_METRES = {"ft": 0.3048, "feet": 0.3048, "foot": 0.3048}

_SIDE_WORDS = {"front": ("FRONT",), "rear": ("REAR", "BACK"), "left": ("LEFT",), "right": ("RIGHT",)}

# Field -> regex matching "LABEL ... NUMBER" inside a single text/MTEXT
# string (DXF annotation is very often one block containing both the label
# and the value, unlike a PDF where they may be separately-positioned text
# runs). Order matters: more specific labels (net/coverage/etc.) are tried
# before the generic ones so they aren't shadowed.
_AREA_FIELD_PATTERNS: list[tuple[str, "re.Pattern[str]", str]] = [
    ("plot.net_area", re.compile(r"NET\s+(?:AREA\s+OF\s+)?PLOT[^0-9\-]{0,20}(?P<value>\d+(?:\.\d+)?)", re.I), "m2"),
    ("plot.area", re.compile(r"(?:AREA\s+OF\s+PLOT|PLOT\s+AREA|SITE\s+AREA)[^0-9\-]{0,20}(?P<value>\d+(?:\.\d+)?)", re.I), "m2"),
    (
        "building.footprint_area",
        re.compile(r"(?:PROPOSED\s+COVERAGE\s+AREA|GROUND\s+COVERAGE\s+AREA|GROUND\s+FLOOR\s+AREA|PLINTH\s+AREA)[^0-9\-]{0,20}(?P<value>\d+(?:\.\d+)?)", re.I),
        "m2",
    ),
    ("coverage", re.compile(r"COVERAGE[^0-9\-%]{0,20}(?P<value>\d+(?:\.\d+)?)\s*%", re.I), "%"),
    # Allows letters in the gap (e.g. "FAR Area (1.64)"), matching
    # `coverage`'s own style just above -- see `_select_labeled_area_match`'s
    # docstring for why the OLD, letter-forbidding gap here was itself a bug,
    # not just a source of ambiguity.
    ("far", re.compile(r"\bFAR\b[^0-9\-%]{0,20}(?P<value>\d+(?:\.\d+)?)", re.I), "ratio"),
    (
        "building.gross_built_up_area",
        re.compile(r"(?:TOTAL|GROSS)\s+BUILT[- ]?UP\s+AREA[^0-9\-]{0,20}(?P<value>\d+(?:\.\d+)?)", re.I),
        "m2",
    ),
]

# Area-table rows in AutoDCR-style templates cite other rows by number:
# "3.  BALANCE AREA OF PLOT (1-2) :", "13.  TOTAL BUILT UP AREA PROPOSED
# (10+11+12)". Those digits are row references, not measurements; the label
# patterns below otherwise read them as the value (plot.area = 1.0 m2,
# gross built-up = 10.0 m2, both asserted at 0.9 confidence). A parenthesised
# arithmetic expression is never a value, whereas a lone "(1.64)" can be.
_ROW_FORMULA_RE = re.compile(r"\(\s*\d+(?:\.\d+)?(?:\s*[-+*/x×]\s*\d+(?:\.\d+)?)+\s*\)")


_PERCENT_TAIL_RE = re.compile(r"\s*%")


def _without_row_formulas(text: str) -> str:
    return _ROW_FORMULA_RE.sub(" ", text)


# A single coverage/FAR worksheet routinely restates the SAME
# `_AREA_FIELD_PATTERNS` field several times with different qualifiers: a
# regulatory ceiling ("Permissible Coverage area"), the actual as-built
# figure ("Achieved Net coverage area"/"Proposed Coverage Area"), and a
# derived remainder ("Balance coverage area left") -- confirmed directly on
# a real fixture (PLAN8.dxf), where blind first-match-in-file-order
# selection shipped the PERMISSIBLE ceiling (70%) as "coverage" instead of
# the achieved 62.83%, and -- more seriously, because the OLD `far` regex
# above could not match "FAR Area" phrasing at all (see its own comment) --
# an unrelated "Residential FAR (100.00%)" sub-line as "far" instead of the
# sheet's own "Achieved Net FAR Area (1.64)" a few lines below it, a
# physically-impossible FAR ratio shipped at the same hardcoded 0.9
# confidence as a genuine reading. This project's coverage/far fields mean
# the building's own achieved figure, never a limit it's being checked
# against or a remainder calculation, so a qualifier naming the achieved
# figure is preferred when more than one reading exists on one sheet.
#
# This ONLY changes behavior when a genuine choice exists (more than one
# text run matches the same field's pattern) -- a sheet with a single,
# unqualified reading (the common case, and the only case any existing
# fixture/test exercises) is completely unaffected: `_select_labeled_area_
# match` falls back to the first (only) candidate exactly as before.
_PREFERRED_AREA_QUALIFIERS = re.compile(r"ACHIEVED|\bNET\b|PROPOSED", re.I)
_DEPRIORITIZED_AREA_QUALIFIERS = re.compile(
    r"PERMISSIBLE|ALLOWABLE|\bMAX\b|\bPERM\b|BALANCE|\bTDR\b|PREMIUM|\bTOTAL\b", re.I
)


def _select_labeled_area_match(candidates: list[tuple["_RawText", float]]) -> Optional[tuple["_RawText", float]]:
    """Pick which of several same-field label matches on one sheet to ship.

    `candidates`: (source text entity, parsed value) pairs, in file-
    encounter order, all already confirmed to match the SAME `_AREA_FIELD_
    PATTERNS` field. Never makes a field that previously always resolved to
    something now resolve to nothing: the worst case (every candidate
    carries a deprioritized qualifier, or none carries any qualifier at
    all) falls back to the first candidate, i.e. the OLD behavior exactly --
    this function can only IMPROVE which candidate wins, never remove one
    that would have shipped before.
    """
    if not candidates:
        return None
    preferred = [c for c in candidates if _PREFERRED_AREA_QUALIFIERS.search(c[0].text)]
    if preferred:
        return preferred[0]
    non_deprioritized = [c for c in candidates if not _DEPRIORITIZED_AREA_QUALIFIERS.search(c[0].text)]
    if non_deprioritized:
        return non_deprioritized[0]
    return candidates[0]


# A sheet routinely contains several genuinely different drawings at once
# (a site plan, a wall section, one or more floor plans, elevations, a
# schedule-of-openings table) -- confirmed directly on a real bundled DXF
# fixture: an explicitly-labeled "SITE PLAN SCALE 1:200" sub-drawing, whose
# own printed dimensions matched ground truth almost exactly, scored the
# WORST of twelve detected regions on generic structural heuristics alone
# (fragment count/envelope size), while an unrelated wall-section view and
# an unrelated floor plan scored highest and were what production actually
# resolved from. Structural/geometric scoring has no notion of "is this
# actually the site plan" -- it can only ask "which reconstructed envelope
# looks most plot-shaped," which the WRONG drawing can win by accident.
# Recognizing a region's own printed caption/title is a cheap, generic way
# to break that tie in favor of the drawing actually labeled as the one we
# want, on any file, not just the ones that motivated this.
#
# Expressed as keyword GROUPS, not a single joined-phrase regex: a rotated,
# multi-word title (very common in an architectural title block) is often
# OCR'd as separate stacked single-word items ("SITE", then "PLAN" on the
# next line down) rather than one recognized string -- confirmed directly
# on a real bundled fixture. `_find_caption_matches` requires every keyword
# in a group to be found on the same or a nearby text item, so this still
# only fires when all of a group's words are genuinely present together.
_SITE_PLAN_CAPTION_KEYWORD_GROUPS: list[tuple[str, ...]] = [
    ("SITE", "PLAN"),
    ("PLOT", "PLAN"),
    ("LOCATION", "PLAN"),
    ("KEY", "PLAN"),
]

# Derived from the observed range of this pipeline's own structural scores,
# not tuned to make any single file's result come out a particular way:
# measured directly across every region on BOTH real bundled fixtures (26
# regions total -- 12 on PLAN5, 14 on PLAN6), `_score_region_resolution`
# ranged from -6.995 (PLAN6 region1) to 5.102 (PLAN6 region2), a span of
# ~12.1. The bonus must exceed that full span for a captioned region to
# always outrank an uncaptioned one, regardless of which end of the range
# either one happens to sit at. 20.0 gives roughly 8 points (~1.65x the
# observed span) of margin beyond that minimum, so a new file's own
# somewhat wider spread doesn't silently erode the guarantee right at the
# boundary. Re-measure this range against a wider corpus (walk every
# region's own `_score_region_resolution` the way `_resolve_via_regions`
# does, before any caption bonus) if a future file's scores are found to
# exceed it, rather than assuming this margin holds indefinitely.
_SITE_PLAN_CAPTION_SCORE_BONUS = 20.0

# A region can be the genuinely right DRAWING (confirmed by its caption)
# while its own reconstructed geometry is still known-implausible -- a
# caption match is not evidence that the geometry is accurate. Confirmed
# directly: PLAN5's region3 is unambiguously the correct site plan, yet its
# own reconstructed plot area (16.6 m2) is below this module's own
# plausible-plot-area floor (`_PLAUSIBLE_PLOT_AREA_M2`), which is most of
# why its pre-caption score (-4.97) was the worst of all 12 regions on that
# sheet. Shipping those measurements at a normal reconstruction confidence
# (0.6-0.85) would misrepresent how much this specific number should be
# trusted. Deliberately well below `VISION_CONFIDENCE_LOW_THRESHOLD` (0.75,
# see `enums.py`) so this always buckets as LOW confidence downstream, never
# silently MEDIUM.
_CAPTION_OVERRIDE_CONFIDENCE_CAP = 0.4

# DXF_FAILURE_TAXONOMY.md item 9 -- see `_RawPoly.unconfirmed_drawing_
# identity`'s own docstring for the failure mode this targets. TODO:
# calibrate against a wider ground-truth corpus (see the taxonomy's
# consolidated "provisional thresholds" section) -- 2 is chosen only
# because it is the smallest number that means "more than one," not
# because it was measured against a range of real multi-drawing sheets;
# PLAN6 (where this fires) actually has 4 originally-plausible regions
# once its glyph swarms are excluded, so this threshold has real margin
# on the one file that motivated it, but that is one file, not a corpus.
_MIN_PLAUSIBLE_CANDIDATES_FOR_UNCONFIRMED_IDENTITY_CONCERN = 2


def _explicit_setback_value(texts: list["_RawText"], side: str) -> Optional[tuple[float, str]]:
    words = "|".join(_SIDE_WORDS[side])
    pattern = re.compile(rf"\b(?:{words})\s*SETBACK\b[^0-9\-]{{0,15}}(?P<value>\d+(?:\.\d+)?)", re.I)
    for t in texts:
        m = pattern.search(t.text)
        if m:
            try:
                return float(m.group("value")), t.text.strip()
            except (TypeError, ValueError):
                continue
    return None


def _find_road_text(texts: list["_RawText"]) -> Optional[tuple[float, str]]:
    for t in texts:
        m = _ROAD_VALUE_RE.search(t.text)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except (TypeError, ValueError):
            continue
        unit = (m.group("unit") or "").strip().lower()
        return value * _ROAD_UNIT_TO_METRES.get(unit, 1.0), t.text.strip()
    return None


# ---------------------------------------------------------------------------
# Internal raw-entity representation (drawing units, pre-scaling)
# ---------------------------------------------------------------------------


@dataclass
class _RawPoly:
    layer: str
    polygon: Polygon
    # True when this polygon's boundary includes an ARC/CIRCLE (or an
    # LWPOLYLINE bulge) that was flattened into straight segments rather
    # than being drawn with straight lines to begin with -- an approximation
    # of the true curved boundary, not a loss of precision on already-
    # straight geometry. Carried through so measurement confidence can
    # reflect it (see `_entry_source_confidence`) instead of silently
    # claiming the same exactness as a polygon with no curves at all.
    approximated_curve: bool = False
    # Set only when this polygon came from `_reconstruct_plot_envelope`:
    # "fragment_boundary" for the line-fitting reconstruction
    # (`geo.reconstruct_boundary_from_fragments`), "bounding_rectangle" for
    # the cruder minimum-area-rectangle fallback used when the former
    # can't find enough well-supported sides. None for anything else
    # (exact vector geometry, wall-loop reconstruction, Vision fallback).
    # Carried through so provenance text and confidence can honestly
    # reflect which method actually produced the polygon.
    envelope_method: Optional[str] = None
    # Set only when this polygon's region won its resolution SOLELY because
    # of a drawing-type-caption score override (`_SITE_PLAN_CAPTION_SCORE_
    # BONUS`) -- i.e. this region's own PRE-caption structural score was
    # below `_MIN_USABLE_REGION_SCORE` (would never have been considered
    # usable on structural merit alone; see `_score_region_resolution`'s own
    # plot-area-implausibility penalty, the most common cause). A caption
    # match is strong evidence this is the right DRAWING, but it is not
    # evidence that this region's own reconstructed geometry is accurate --
    # confirmed as a real, separate failure mode: PLAN5's region3 is
    # unambiguously the correct site plan, yet its own reconstructed plot
    # area (16.6 m2) is still below the plausible-plot-area floor its own
    # score reflects, and its shipped width/depth are ~3-4x too small.
    # Carried through so measurement confidence can be capped accordingly
    # instead of shipping a normal reconstruction confidence for a region
    # the pipeline's own structural check already flagged as implausible.
    caption_overrode_implausible_score: bool = False
    # DXF_FAILURE_TAXONOMY.md item 9. Set when this region won on ordinary
    # structural/evidence scoring on a sheet where MULTIPLE regions
    # independently looked like plausible drawings on their own merit
    # (`plausible_candidate_count >= _MIN_PLAUSIBLE_CANDIDATES_FOR_
    # UNCONFIRMED_IDENTITY_CONCERN` in `_resolve_via_regions`) and NOT ONE
    # of them was confirmed by a drawing-type caption anywhere on the
    # sheet. This is a materially different, and more dangerous, situation
    # than `caption_overrode_implausible_score` above: there, a caption
    # POSITIVELY identified a specific (if structurally poor-looking)
    # region as the site plan. Here, there is no positive identification
    # signal for ANY region at all, on a sheet complex enough that
    # misidentifying the drawing is a live possibility -- confirmed
    # directly: PLAN6[dxf], with its six text-glyph-swarm regions
    # correctly excluded (item 1), resolved from a merge of genuine
    # dash-dot fragments that never received a caption match, while the
    # ACTUAL site plan (region12, matching ground truth almost exactly)
    # was independently confirmed to exist and be excluded from winning
    # only by an OCR-legibility gap (item 0) -- and shipped at a normal
    # 0.6 reconstruction confidence with nothing anywhere to indicate that.
    # Shares `_CAPTION_OVERRIDE_CONFIDENCE_CAP`'s own magnitude (both
    # represent "don't trust this pipeline's identification of which
    # drawing this is," just via different evidence), tracked as its own
    # flag rather than reusing the same one so a future reader can always
    # tell which of the two reasons applied.
    unconfirmed_drawing_identity: bool = False

    def scaled(self, factor: float) -> "_RawPoly":
        return _RawPoly(
            self.layer,
            Polygon(points=[Point(x=p.x * factor, y=p.y * factor) for p in self.polygon.points]),
            self.approximated_curve,
            self.envelope_method,
            self.caption_overrode_implausible_score,
            self.unconfirmed_drawing_identity,
        )


@dataclass
class _RawText:
    text: str
    position: tuple[float, float]
    layer: str

    def scaled(self, factor: float) -> "_RawText":
        return _RawText(self.text, (self.position[0] * factor, self.position[1] * factor), self.layer)


@dataclass
class _RawDim:
    value: float
    position: tuple[float, float]
    raw_text: str
    layer: str
    # The two extension-line origin points (DXF group codes 13/14, ezdxf
    # `dxf.defpoint2`/`dxf.defpoint3`) for a LINEAR or ALIGNED DIMENSION
    # entity -- the exact real-world points this dimension measures BETWEEN,
    # straight from the dimension's own definition, not inferred from its
    # text position. None for dimension types (radius/diameter/angular) or
    # malformed entities where this isn't a straight two-point span. See
    # `_find_dimension_edge_matches` for what this enables: verifying a
    # dimension against a specific polygon edge by exact endpoint
    # coincidence, rather than proximity/text matching.
    defpoint2: Optional[tuple[float, float]] = None
    defpoint3: Optional[tuple[float, float]] = None

    def scaled(self, factor: float) -> "_RawDim":
        return _RawDim(
            self.value * factor, (self.position[0] * factor, self.position[1] * factor), self.raw_text, self.layer,
            (self.defpoint2[0] * factor, self.defpoint2[1] * factor) if self.defpoint2 is not None else None,
            (self.defpoint3[0] * factor, self.defpoint3[1] * factor) if self.defpoint3 is not None else None,
        )


def _make_polygon(points_xy: list[tuple[float, float]]) -> Optional[Polygon]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points_xy:
        if cleaned and math.hypot(x - cleaned[-1][0], y - cleaned[-1][1]) < 1e-9:
            continue
        cleaned.append((x, y))
    if len(cleaned) >= 2 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < 1e-9:
        cleaned.pop()
    if len(cleaned) < 3:
        return None
    poly = Polygon(points=[Point(x=x, y=y) for x, y in cleaned])
    if poly.area < 1e-6:
        return None
    return poly


# Max points a single flattened curve contributes, to bound cost on a
# pathological input (e.g. an enormous CIRCLE radius combined with a tiny
# relative sagitta). Ample for any real architectural boundary curve.
_MAX_CURVE_FLATTEN_POINTS = 400


def _flatten_curve_points(entity, radius: float) -> list[tuple[float, float]]:
    """Approximate an ARC/CIRCLE entity's curve as a bounded list of (x, y) points.

    `sagitta` (the max deviation between the true arc and the flattened
    chord) is sized relative to the entity's own radius rather than a fixed
    drawing-unit constant, since drawing units are not yet resolved to
    metres at this stage (raw collection happens before `_resolve_unit_factor`)
    -- a fixed absolute sagitta would be far too coarse for a drawing in
    metres and far too fine (huge point counts) for one in millimetres.
    """
    if radius <= 0:
        return []
    sagitta = max(radius * 0.02, 1e-9)
    try:
        pts = [(v.x, v.y) for v in entity.flattening(sagitta)]
    except Exception:
        return []
    if len(pts) > _MAX_CURVE_FLATTEN_POINTS:
        step = len(pts) // _MAX_CURVE_FLATTEN_POINTS + 1
        pts = pts[::step]
    return pts


def _flatten_lwpolyline_points(e) -> list[tuple[float, float]]:
    """Flatten an LWPOLYLINE's own vertices, respecting bulge (arc) segments.

    `e.get_points("xy")` alone silently ignores bulge and connects vertices
    with a straight line, which quietly turns a rounded plot corner or a
    curved boundary into a polygon with the wrong shape and the wrong area.
    `virtual_entities()` decomposes the polyline into its real LINE/ARC
    pieces in vertex order; each ARC piece is then flattened via
    `_flatten_curve_points` and the straight LINE pieces are kept as-is, so
    the result is the polyline's actual boundary rather than a silent
    straight-line approximation of it.
    """
    pts: list[tuple[float, float]] = []
    try:
        for sub in e.virtual_entities():
            sub_type = sub.dxftype()
            if sub_type == "LINE":
                start, end = sub.dxf.start, sub.dxf.end
                if not pts:
                    pts.append((start.x, start.y))
                pts.append((end.x, end.y))
            elif sub_type == "ARC":
                try:
                    radius = float(sub.dxf.radius)
                except Exception:
                    radius = 0.0
                arc_pts = _flatten_curve_points(sub, radius)
                if arc_pts:
                    if not pts:
                        pts.append(arc_pts[0])
                    pts.extend(arc_pts[1:])
    except Exception:
        return []
    return pts


def _point_bbox(pos: tuple[float, float], eps: float = 0.01) -> BoundingBox:
    x, y = pos
    return BoundingBox(min_x=x - eps, min_y=y - eps, max_x=x + eps, max_y=y + eps)


def _bbox_list(bbox: Optional[BoundingBox]) -> Optional[list[float]]:
    if bbox is None:
        return None
    return [round(bbox.min_x, 3), round(bbox.min_y, 3), round(bbox.max_x, 3), round(bbox.max_y, 3)]


def _bbox_nested(inner: BoundingBox, outer: BoundingBox, tol: float = 0.5) -> bool:
    return (
        inner.min_x >= outer.min_x - tol
        and inner.max_x <= outer.max_x + tol
        and inner.min_y >= outer.min_y - tol
        and inner.max_y <= outer.max_y + tol
    )


# Above this vertex count, a polygon is simplified to its own bounding-box
# rectangle for graph-construction purposes only (see `_polygons_to_site_graph`).
# Lossless for the common case: a simple 3-4 vertex axis-aligned quad (the
# shape of, e.g., one dash of a vectorized dash-dot boundary rendered as a
# tiny filled rectangle) already equals its own bounding box, so this only
# ever approximates the rarer, genuinely complex/many-vertex shapes -- which
# is exactly where `build_site_graph`'s per-pair cost (point-to-segment
# distance checks scale with each polygon's own vertex count) is
# concentrated. `site_graph.build_site_graph`'s own docstring documents it
# as O(n^2) "fine for a few hundred polygons"; that held in wall-clock
# terms only for simple shapes -- 300 polygons averaging ~14 vertices each
# (a real bundled DXF sample, produced by vectorizing a scanned sheet)
# measured at 92s before this simplification.
_GRAPH_SIMPLIFY_VERTEX_THRESHOLD = 8


def _graph_polygon(polygon: Polygon) -> Polygon:
    if len(polygon.points) <= _GRAPH_SIMPLIFY_VERTEX_THRESHOLD:
        return polygon
    b = polygon.bounding_box
    return Polygon(points=[
        Point(x=b.min_x, y=b.min_y), Point(x=b.max_x, y=b.min_y),
        Point(x=b.max_x, y=b.max_y), Point(x=b.min_x, y=b.max_y),
    ])


def _polygons_to_site_graph(polygons: list[_RawPoly]) -> tuple["SiteGraph", dict[int, _RawPoly]]:
    from backend.spatial_reasoning.site_graph import SiteGraphNode, build_site_graph

    # `by_id` always maps back to the ORIGINAL `_RawPoly` (exact polygon,
    # used for the final plot.area/building.footprint_area measurements
    # once a winner is chosen) -- only the graph's own nodes use the
    # simplified geometry, purely to make role-inference's internal
    # containment/adjacency comparisons fast.
    nodes = [
        SiteGraphNode(id=i, polygon=_graph_polygon(p.polygon), layer=p.layer)
        for i, p in enumerate(polygons)
    ]
    graph = build_site_graph(nodes)
    by_id = {i: p for i, p in enumerate(polygons)}
    return graph, by_id


def _build_role_inference_graph(polygons: list[_RawPoly]) -> tuple["SiteGraph", dict[int, _RawPoly]]:
    """
    Build the plot/building/road role-inference graph ONCE, capped to a
    bounded working set.

    `site_graph.build_site_graph` is O(n^2) in polygon count BY DESIGN (see
    its own docstring) -- fine for a normal architect-authored DXF's few
    hundred closed shapes. A DXF produced by vectorizing/tracing a scanned
    sheet can instead contain many thousands of tiny closed fragments (a
    real bundled sample has ~6,300: each dash of a dash-dot property line
    rendered as its own tiny filled quad, the same dashed-boundary pattern
    already documented for the PDF version of this exact sheet in
    `cv_extraction/site_plan.py`). Feeding that many polygons into an O(n^2)
    build -- and doing it THREE separate times, once each for plot/
    building/road, which is what `_pick_plot_polygon`/`_pick_building_polygon`/
    `_pick_road_polygon` used to do independently -- is tens of millions of
    pairwise geometry comparisons per call, which in practice made a DXF
    upload appear to hang indefinitely (no exception, no timeout, just an
    extremely long-running computation).

    The fix: build the graph exactly once, and cap it to the largest
    `settings.max_dxf_role_inference_polygons` closed shapes by area, plus
    (uncapped) any polygon whose layer name already matches the PLOT/
    BUILDING/ROAD layer-name hints -- the real plot/building/road are
    always among the largest shapes on a real sheet, or explicitly
    layer-tagged, so nothing structurally relevant is lost.
    """
    from backend.config import get_settings

    limit = get_settings().max_dxf_role_inference_polygons
    if len(polygons) <= limit:
        working_set = polygons
    else:
        by_area = sorted(polygons, key=lambda p: p.polygon.area, reverse=True)
        top_n = by_area[:limit]
        layer_hinted = [
            p for p in by_area[limit:]
            if _PLOT_LAYER_RE.search(p.layer or "") or _BUILDING_LAYER_RE.search(p.layer or "") or _ROAD_LAYER_RE.search(p.layer or "")
        ]
        working_set = top_n + layer_hinted
    return _polygons_to_site_graph(working_set)


def _pick_plot_polygon(
    polygons: list[_RawPoly], graph: "SiteGraph", by_id: dict[int, _RawPoly]
) -> Optional[_RawPoly]:
    """
    Delegates to site_graph.infer_plot_node -- see that module for why
    role assignment is a graph-structure query (containment fraction, not
    raw area) rather than "biggest polygon in the file". This preserves
    the _RawPoly in/out contract the rest of this file uses; the actual
    reasoning lives in site_graph.py so the same logic is available to
    PDF extraction and GNN training without re-implementing it a third
    and fourth time.
    """
    from backend.spatial_reasoning.site_graph import infer_plot_node

    matches = [p for p in polygons if _PLOT_LAYER_RE.search(p.layer or "")]
    if matches:
        return max(matches, key=lambda p: p.polygon.area)
    if not graph.nodes:
        return None
    winner = infer_plot_node(graph)
    return by_id[winner.id] if winner is not None else None


def _pick_building_polygon(
    graph: "SiteGraph", by_id: dict[int, _RawPoly], plot_entry: Optional[_RawPoly]
) -> Optional[_RawPoly]:
    from backend.spatial_reasoning.site_graph import infer_building_node

    if plot_entry is None:
        return None
    plot_id = next((i for i, p in by_id.items() if p is plot_entry), None)
    if plot_id is None:
        # The plot was resolved via the layer-name fast path and happens
        # not to be part of the (possibly capped) role-inference graph --
        # can't query containment against a node the graph doesn't have.
        return None
    layer_hint_ids = [
        i for i, p in by_id.items() if p is not plot_entry and _BUILDING_LAYER_RE.search(p.layer or "")
    ]
    layer_hint = [graph.node(i) for i in layer_hint_ids] or None
    winner = infer_building_node(graph, graph.node(plot_id), layer_hint)
    return by_id[winner.id] if winner is not None else None


# `_pick_road_polygon`'s geometric-adjacency path for a synthetic envelope
# plot runs once per candidate region -- keep its own candidate count small
# even though `by_id` is already capped, since a real multi-region sheet
# can call it many times over.
_MAX_CANDIDATES_FOR_ENVELOPE_ROAD_ADJACENCY = 80


def _pick_road_polygon(
    polygons: list[_RawPoly],
    graph: "SiteGraph",
    by_id: dict[int, _RawPoly],
    plot_entry: Optional[_RawPoly] = None,
) -> Optional[_RawPoly]:
    from backend.spatial_reasoning.road_access import infer_road_polygon_by_adjacency
    from backend.spatial_reasoning.site_graph import infer_road_node

    if plot_entry is not None:
        plot_id = next((i for i, p in by_id.items() if p is plot_entry), None)
        if plot_id is not None:
            winner = infer_road_node(graph, graph.node(plot_id))
            if winner is not None:
                return by_id[winner.id]
        elif len(by_id) <= _MAX_CANDIDATES_FOR_ENVELOPE_ROAD_ADJACENCY:
            # The plot was resolved as a synthetic envelope (see
            # `_reconstruct_plot_envelope`) or via the layer-name fast
            # path, so it was never one of the polygons `graph`/`by_id`
            # were built from -- `infer_road_node`'s own by-id lookup
            # always fails here. Geometric road-adjacency inference
            # (`infer_road_polygon_by_adjacency`) only needs the plot's
            # own polygon, not its graph membership, so run it directly
            # against every OTHER candidate polygon rather than skipping
            # straight to the layer-name-only fallback -- the same
            # reasoning `_pick_building_by_geometric_containment` already
            # applies on the building side of this exact gap. Scoped to
            # `by_id`'s own (already bounded, at most `max_dxf_role_
            # inference_polygons`) working set, and further capped here:
            # this runs once per CANDIDATE REGION (there can be many on a
            # real multi-drawing sheet), so even the 300-polygon graph cap
            # multiplied across regions measurably added to real DXF
            # extraction time in validation -- a sheet with this many
            # unlabeled candidates and no closed-loop plot is unlikely to
            # have its road reliably identifiable by shape alone anyway,
            # so skipping straight to the (cheap) layer-name fallback here
            # is the safer trade.
            others = [p.polygon for p in by_id.values() if p is not plot_entry]
            road_polygon = infer_road_polygon_by_adjacency(plot_entry.polygon, others)
            if road_polygon is not None:
                match = next((p for p in by_id.values() if p.polygon is road_polygon), None)
                if match is not None:
                    return match

    # Fall back to layer-name matching only when geometry found nothing
    # adjacent/road-shaped (e.g. the road wasn't drawn as a closed polygon
    # at all, only as open lines) -- an explicit "ROAD" layer is still
    # useful signal in that case.
    matches = [p for p in polygons if _ROAD_LAYER_RE.search(p.layer or "")]
    if not matches:
        return None
    return max(matches, key=lambda p: p.polygon.area)


@dataclass
class BorrowedRoadCandidate:
    """A road polygon resolved by a DIFFERENT region on the same sheet,
    offered to a region whose own isolated resolution found none of its
    own. See `find_nearby_road_candidate`'s own docstring."""

    road_entry: "_RawPoly"
    source_region_id: int
    center_distance: float


# TODO: calibrate against a wider ground-truth corpus (see
# DXF_FAILURE_TAXONOMY.md item 8) -- placeholder, not tuned to either real
# fixture. A road drawn as its own region (density-based clustering kept
# it separate from the plot/building content it fronts) should sit close
# to that plot, but "close" has to be judged relative to the PLOT's own
# size, not a fixed drawing-unit distance -- a large plot's road can
# legitimately sit farther away in raw units than a small plot's ever
# would, and a small plot merely near a large road region should not
# borrow it just because raw distance happens to be small.
_MAX_BORROWED_ROAD_DISTANCE_FACTOR = 3.0

# TODO: calibrate. When a second candidate's own distance is within this
# fraction of the closest one's, treat the choice as ambiguous and refuse
# to borrow either -- picking the wrong one would silently attribute an
# unrelated road's width/position to this plot, which is worse than
# leaving road.width MISSING.
_BORROWED_ROAD_AMBIGUITY_MARGIN = 0.25


def find_nearby_road_candidate(
    target_plot_bbox: BoundingBox, candidates: Sequence[tuple[int, "_RawPoly"]],
) -> Optional[BorrowedRoadCandidate]:
    """DXF_FAILURE_TAXONOMY.md item 8: a small, isolated site-plan region
    resolved on its own (e.g. via `_resolve_via_regions`'s per-region Pass
    1) very often finds no road polygon of its own -- not because no road
    is drawn on the sheet, but because density-based region clustering
    kept the road strip as its OWN separate region, split apart from the
    plot/building content exactly the way item 2's merge growth already
    handles for fragmented plot/building geometry, just not yet extended
    to roads.

    Root cause confirmed directly on two real fixtures (PLAN5's region3,
    PLAN6's region12): without a road edge in the region's own candidate-
    edge set, the sheet's own printed road-width label ("10m WIDE ROAD" /
    "7.30m Wide Road") -- the ONE piece of independently-confirmable scale
    evidence actually available on these sheets -- mis-associates with the
    nearest WRONG edge instead (a plot or building edge), producing
    incoherent, disagreeing implied scale factors rather than a usable
    correction, instead of the correct one the sheet's own text could
    otherwise confirm.

    This borrows another region's ALREADY-RESOLVED road_entry (no new
    geometry work -- reuses Pass 1's own per-region `_resolve_plot_
    building_road` output, `candidates` is just `[(region_id, road_entry),
    ...]` for every OTHER region that found one) whose own bbox center
    sits close to `target_plot_bbox`, relative to the target's own size.
    Returns the single closest candidate -- but refuses (returns None)
    rather than guessing when:
      - there is no candidate at all,
      - the closest one is farther than `_MAX_BORROWED_ROAD_DISTANCE_
        FACTOR` times the target plot's own diagonal (not "nearby" by any
        reasonable reading), or
      - a second candidate is comparably close (within `_BORROWED_ROAD_
        AMBIGUITY_MARGIN` of the closest) -- ambiguous, and guessing wrong
        here would silently corrupt evidence association with a road that
        isn't actually this plot's own frontage.
    """
    if not candidates:
        return None
    target_cx = (target_plot_bbox.min_x + target_plot_bbox.max_x) / 2.0
    target_cy = (target_plot_bbox.min_y + target_plot_bbox.max_y) / 2.0
    target_diagonal = math.hypot(
        target_plot_bbox.max_x - target_plot_bbox.min_x, target_plot_bbox.max_y - target_plot_bbox.min_y,
    )
    if target_diagonal <= 1e-9:
        return None

    scored = []
    for region_id, road_entry in candidates:
        rb = road_entry.polygon.bounding_box
        cx, cy = (rb.min_x + rb.max_x) / 2.0, (rb.min_y + rb.max_y) / 2.0
        scored.append((math.hypot(cx - target_cx, cy - target_cy), region_id, road_entry))
    scored.sort(key=lambda t: t[0])

    closest_dist, closest_region_id, closest_road = scored[0]
    if closest_dist > target_diagonal * _MAX_BORROWED_ROAD_DISTANCE_FACTOR:
        return None
    if len(scored) > 1 and scored[1][0] <= closest_dist * (1.0 + _BORROWED_ROAD_AMBIGUITY_MARGIN):
        return None
    return BorrowedRoadCandidate(road_entry=closest_road, source_region_id=closest_region_id, center_distance=closest_dist)


# --- Vision-render fallback --------------------------------------------------
#
# Only used when the deterministic closed-polygon search above finds no
# plausible building/road. See `_vision_fallback_regions`'s own docstring.

_VISION_FALLBACK_LAYER = "VISION_FALLBACK"
_RECONSTRUCTED_LAYER = "RECONSTRUCTED_FROM_LINEWORK"
_WALL_UNION_RECONSTRUCTED_LAYER = "RECONSTRUCTED_WALL_UNION"

# Matches dxf_reconstruction.py's own plausibility bounds -- kept as the
# single generic definition of "a footprint this size relative to its plot
# is architecturally implausible" used both for a genuinely closed
# building polygon (below) and for a reconstructed one (dxf_reconstruction.py).
_MIN_BUILDING_PLOT_AREA_FRACTION = 0.03


def _reject_implausible_building(
    entry: Optional["_RawPoly"], plot_entry: Optional["_RawPoly"], warnings: list[str]
) -> Optional["_RawPoly"]:
    """A building footprint under ~3% of its plot's area is implausible for
    any real residential/commercial structure -- far more likely that no
    genuine building outline exists as a single closed polygon/region here
    at all than that a real building is genuinely this small. A confident-
    looking but implausibly tiny footprint is worse than an honest MISSING,
    per this project's "wrong is worse than missing" principle. Standalone
    (not a closure) so both the global and the per-region resolution paths
    can share the exact same rule.
    """
    if (
        entry is not None
        and plot_entry is not None
        and plot_entry.polygon.area > 0
        and entry.polygon.area / plot_entry.polygon.area < _MIN_BUILDING_PLOT_AREA_FRACTION
    ):
        warnings.append(
            f"Rejected a candidate building-footprint polygon ({entry.polygon.area:.2f} m2, "
            f"layer '{entry.layer}') as implausibly small relative to the plot "
            f"({plot_entry.polygon.area:.2f} m2) -- likely a stray fragment rather than the "
            "actual building outline, which may not exist as a single closed polygon in this "
            "DXF at all. Treated as unresolved."
        )
        return None
    return entry


_ENVELOPE_RECONSTRUCTED_LAYER = "RECONSTRUCTED_ENVELOPE"
# A handful of stray closed shapes must never be promoted into a
# fabricated plot boundary -- this bar is deliberately high, matched to
# the "hundreds/thousands of dash fragments" failure mode this exists for
# (confirmed on a real production file: every closed polygon under 8 m2
# despite a true plot area of ~160 m2), not to any specific plan's count.
_MIN_FRAGMENTS_FOR_ENVELOPE_RECONSTRUCTION = 20
_MAX_ENVELOPE_ASPECT_RATIO = 8.0


def _reconstruct_plot_envelope(
    polygons: list["_RawPoly"],
    open_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
) -> Optional["_RawPoly"]:
    """Recover a plot-like boundary from a scattered field of many small
    closed fragments and/or open line-work, for when no single closed
    polygon is plot-sized at all.

    Real vectorized-trace DXFs commonly render a dashed/dash-dot property
    boundary as hundreds or thousands of individually tiny CLOSED quads
    (one per dash) rather than one continuous outline. Neither
    `_pick_plot_polygon` (needs ONE polygon containing most other shapes)
    nor `dxf_reconstruction.reconstruct_building_polygon` (only consumes
    OPEN line-work, never small closed shapes) can recover a boundary made
    of many small closed fragments -- this fills that gap.

    Every point belonging to this polygon/segment set (fragment vertices +
    open-segment endpoints) is treated as a sample lying along the true
    boundary. Two recovery methods are tried, in order of trust:

    1. `geo.reconstruct_boundary_from_fragments` -- groups fragments onto
       shared boundary sides by orientation and perpendicular distance to
       a common supporting line (never by proximity between fragments),
       so it correctly bridges large intentional dash gaps and a rotated
       boundary, and does not simply absorb an unrelated stray mark into
       the fit the way a single bounding rectangle over every point
       would. This is the general fix for fragmented boundaries -- see
       its own docstring for the full algorithm.
    2. If that finds fewer than 3 well-supported sides (e.g. the fragment
       field is too sparse or irregular for line-fitting to lock onto),
       fall back to the coarser minimum-area oriented bounding rectangle
       over every point (`geo.min_area_bounding_rectangle`) -- still
       rotation-correct, just less precise when unrelated points are
       mixed in with the real boundary.

    Only attempted when there is a genuinely large fragment population
    (`_MIN_FRAGMENTS_FOR_ENVELOPE_RECONSTRUCTION`), and the result must
    still pass an aspect-ratio plausibility check, so a handful of stray
    marks or a degenerate sliver never gets promoted into a fabricated
    boundary.
    """
    n_fragments = len(polygons) + len(open_segments)
    if n_fragments < _MIN_FRAGMENTS_FOR_ENVELOPE_RECONSTRUCTION:
        return None
    points: list[Point] = []
    for p in polygons:
        points.extend(p.polygon.points)
    for _layer, start, end in open_segments:
        points.append(Point(x=start[0], y=start[1]))
        points.append(Point(x=end[0], y=end[1]))
    if len(points) < 3:
        return None

    fitted = geo.reconstruct_boundary_from_fragments(points)
    if fitted is not None and fitted.area > 0:
        ar = geo.aspect_ratio(fitted.bounding_box)
        if math.isfinite(ar) and ar <= _MAX_ENVELOPE_ASPECT_RATIO:
            return _RawPoly(layer=_ENVELOPE_RECONSTRUCTED_LAYER, polygon=fitted, envelope_method="fragment_boundary")

    rect = geo.min_area_bounding_rectangle(points)
    if rect is None or rect.area <= 0:
        return None
    ar = geo.aspect_ratio(rect.bounding_box)
    if not math.isfinite(ar) or ar > _MAX_ENVELOPE_ASPECT_RATIO:
        return None
    return _RawPoly(layer=_ENVELOPE_RECONSTRUCTED_LAYER, polygon=rect, envelope_method="bounding_rectangle")


def _pick_building_by_geometric_containment(
    polygons: list["_RawPoly"], plot_entry: Optional["_RawPoly"]
) -> Optional["_RawPoly"]:
    """Find a plausible building candidate purely by geometry, when the
    graph-based `_pick_building_polygon` cannot be used at all.

    `_pick_building_polygon` looks up the plot as a NODE in the
    role-inference graph (`by_id`) to query which other nodes it
    geometrically contains -- but when the plot was resolved as a
    synthetic envelope (`_reconstruct_plot_envelope`, fitted around a
    scattered point cloud rather than read from one closed DXF polygon),
    it was never one of the polygons that graph was built from, so that
    lookup always fails and returns None. A real, genuinely closed
    building polygon can still exist among this region's own polygons in
    that case (e.g. the plot boundary is a dashed/dash-dot fragment field
    but the building itself is one honest closed outline) -- this checks
    containment directly against the plot's geometry instead of going
    through the graph, so that real polygon is not missed just because
    the plot itself wasn't found the "normal" way.
    """
    if plot_entry is None or plot_entry.polygon.area <= 0:
        return None
    from backend.spatial_reasoning.site_graph import MAX_BUILDING_ASPECT_RATIO

    # A real building typically covers a substantial fraction of its plot
    # (residential coverage norms in this project's own domain start well
    # above 10%) -- a much higher bar than the generic "implausibly tiny"
    # rejection threshold used elsewhere. This path searches every closed
    # polygon in the region, including whatever small dash/fragment noise
    # is scattered around (the same fragments the plot's own envelope was
    # built from); without a higher bar here specifically, the largest
    # fragment that happens to clear only the generic 3% floor can win by
    # default and be reported as "the building" even though it is really
    # just another dash-sized fragment, not a traced structure.
    min_ratio = 0.10

    candidates: list["_RawPoly"] = []
    for p in polygons:
        if p is plot_entry or p.polygon is plot_entry.polygon:
            continue
        ratio = p.polygon.area / plot_entry.polygon.area
        if not (min_ratio <= ratio <= 0.92):
            continue
        if geo.aspect_ratio(p.polygon.bounding_box) > MAX_BUILDING_ASPECT_RATIO:
            continue
        if not geo.point_in_polygon(p.polygon.bounding_box.center, plot_entry.polygon):
            continue
        candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.polygon.area)


def _resolve_plot_building_road(
    polygons: list["_RawPoly"],
    open_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    warnings: list[str],
    context_label: str = "",
    allow_envelope_reconstruction: bool = False,
) -> tuple[Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"]]:
    """Deterministic plot/building/road resolution over one polygon set.

    Shared by both the whole-sheet (legacy, single-drawing) path and the
    per-region path below -- role-inference graph, implausibility
    rejection, and fragmented-line-work reconstruction, all scoped to
    whatever `polygons`/`open_segments` the caller passes in. Never
    touches Vision -- that fallback stays a whole-sheet-only, final step
    in `_build_independent_cv`, run at most once regardless of how many
    regions were tried.

    `allow_envelope_reconstruction` gates `_reconstruct_plot_envelope`
    (see its docstring) -- deliberately opt-in and used ONLY by the
    per-region path, never by the whole-sheet fallback: bounding every
    fragment across the WHOLE sheet would just reconstruct the sheet's own
    frame again in a different form, whereas bounding fragments within one
    already-validated spatially-distinct region is a much stronger claim.
    """
    role_graph, role_by_id = _build_role_inference_graph(polygons)
    plot_entry = _pick_plot_polygon(polygons, role_graph, role_by_id)

    if plot_entry is None and allow_envelope_reconstruction:
        envelope = _reconstruct_plot_envelope(polygons, open_segments)
        if envelope is not None:
            prefix = f"[{context_label}] " if context_label else ""
            warnings.append(
                f"{prefix}No single closed polygon was plot-sized ({len(polygons)} closed fragment(s) "
                "present, all individually small -- consistent with a dashed/dash-dot boundary rendered "
                f"as many tiny quads); reconstructed the plot boundary as the minimum-area bounding "
                f"envelope of {len(polygons) + len(open_segments)} fragment/line-work point(s)."
            )
            plot_entry = envelope

    building_entry = _pick_building_polygon(role_graph, role_by_id, plot_entry)
    road_entry = _pick_road_polygon(polygons, role_graph, role_by_id, plot_entry)

    if building_entry is None and plot_entry is not None and plot_entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER:
        # The graph-based pick above always returns None here (see
        # `_pick_building_by_geometric_containment`'s docstring) -- fall
        # back to plain geometric containment so a genuinely closed
        # building polygon elsewhere in this region isn't missed just
        # because the plot itself resolved via envelope reconstruction.
        geometric_building = _pick_building_by_geometric_containment(polygons, plot_entry)
        if geometric_building is not None:
            prefix = f"[{context_label}] " if context_label else ""
            warnings.append(
                f"{prefix}Building resolved by direct geometric containment within the reconstructed "
                "plot envelope (the graph-based nesting check does not apply to a synthetic envelope, "
                "which is not itself a role-inference graph node)."
            )
            building_entry = geometric_building

    building_entry = _reject_implausible_building(building_entry, plot_entry, warnings)

    if building_entry is None:
        from backend.cv_extraction.dxf_reconstruction import reconstruct_building_polygon

        # An envelope-reconstructed plot is a TIGHT (minimum-area) fit
        # around a scattered point cloud, with no margin at all -- a real
        # building loop can easily have a centroid or edge that falls
        # microscopically outside it purely from the envelope's own
        # rotation/fit imprecision, which would otherwise make the
        # strict polygon-containment check inside `reconstruct_building_
        # polygon`'s scoring reject every real candidate. Since the
        # envelope is already an approximation, requiring exact
        # containment against it is inappropriate; score building
        # candidates on general shape plausibility only in this case, and
        # rely on `_reject_implausible_building`'s area-ratio check
        # (below) as the sanity net instead.
        plot_polygon = (
            None if plot_entry is not None and plot_entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER
            else (plot_entry.polygon if plot_entry is not None else None)
        )
        reconstruction = reconstruct_building_polygon(open_segments, plot_polygon)
        if reconstruction is not None:
            prefix = f"[{context_label}] " if context_label else ""
            warnings.append(
                f"{prefix}Building footprint had no closed polygon in the DXF data; " + reconstruction.reason
            )
            building_entry = _reject_implausible_building(
                _RawPoly(layer=_RECONSTRUCTED_LAYER, polygon=reconstruction.polygon),
                plot_entry, warnings,
            )

        # Cycle-based reconstruction requires the wall line-work to form a
        # single, ACTUAL CLOSED loop after endpoint snapping -- a real
        # door/window/opening gap, a T-junction from an interior partition
        # wall, or overlapping/duplicated line-work can all prevent that
        # loop from ever existing in the graph, even though the building's
        # own wall network is otherwise intact. When cycle detection found
        # nothing plausible, try the area-based alternative: buffer each
        # wall segment into a thin area by an estimated wall thickness,
        # bridge document-relative opening-sized gaps between dangling
        # wall endpoints, union everything, and take the union's outer
        # boundary -- see `dxf_wall_union`'s own module docstring for the
        # full rationale. This does not require the wall network to form
        # one single closed loop the way cycle detection does.
        if building_entry is None:
            from backend.cv_extraction.dxf_wall_union import reconstruct_building_via_wall_union

            wall_union_result = reconstruct_building_via_wall_union(open_segments, plot_polygon)
            if wall_union_result is not None:
                prefix = f"[{context_label}] " if context_label else ""
                warnings.append(
                    f"{prefix}Building footprint had no single closed wall loop in the DXF data "
                    "(a door/window gap, T-junction, or overlapping line-work can all prevent one from "
                    "existing even when the wall network is otherwise intact); " + wall_union_result.reason
                )
                building_entry = _reject_implausible_building(
                    _RawPoly(layer=_WALL_UNION_RECONSTRUCTED_LAYER, polygon=wall_union_result.polygon),
                    plot_entry, warnings,
                )

        # Cycle-based and wall-union reconstruction both still require SOME
        # amount of wall line-work with a discoverable structure -- when
        # that still leaves no plausible building and there is a real
        # population of wall-like open segments to work with, fall back to
        # the same minimum-area-envelope technique
        # used for the plot boundary, scoped to ONLY the open segments
        # (walls) rather than every point in the region -- this recovers
        # the building's overall footprint extent even when no single
        # exact closed outline could be traced.
        if building_entry is None and allow_envelope_reconstruction and plot_entry is not None:
            building_envelope = _reconstruct_plot_envelope([], open_segments)
            if building_envelope is not None:
                prefix = f"[{context_label}] " if context_label else ""
                warnings.append(
                    f"{prefix}No closed wall outline could be traced from the building's own line-work "
                    "(a real door/window gap in a wall centerline trace is larger than the endpoint-"
                    f"snapping tolerance used for closed-loop reconstruction); approximated the building "
                    f"footprint as the minimum-area bounding envelope of {len(open_segments)} wall/line-work "
                    "segment(s) instead."
                )
                building_entry = _reject_implausible_building(
                    _RawPoly(layer=_ENVELOPE_RECONSTRUCTED_LAYER, polygon=building_envelope.polygon),
                    plot_entry, warnings,
                )
    return plot_entry, building_entry, road_entry


# A region covering more than this fraction of the whole sheet's extent
# (while other regions exist too) is excluded from candidacy entirely --
# see `_resolve_via_regions`'s inline comment for the reasoning.
_REGION_MAX_SHEET_COVERAGE = 0.6

# A region's resolution is only used in place of the whole-sheet fallback
# when it clears this bar -- i.e. it actually found BOTH a plot and a
# building, not just a plot (a lone plot candidate with no building at all
# is no more useful than the legacy whole-sheet path, which gets the same
# opportunity to try Vision/other fallbacks afterward).
_MIN_USABLE_REGION_SCORE = 2.0


def _score_region_resolution_detailed(
    plot_entry: Optional["_RawPoly"], building_entry: Optional["_RawPoly"]
) -> tuple[list["ConstraintResult"], float]:
    """Same scoring as `_score_region_resolution`, but also returns the named
    `ConstraintResult` terms `score_plot_building_resolution` already
    computes internally -- previously discarded by `_score_region_resolution`
    (which kept only the summed total). Added for Architecture V2 Phase
    DXF-1 (`dxf_evidence_projection.py`), which needs these terms to build
    an auditable `StructuralHypothesis` instead of an opaque float; not a
    behavior change to the score itself, since `_score_region_resolution`
    below now simply returns this function's own `total`.
    """
    from backend.spatial_reasoning.hypothesis_scoring import score_plot_building_resolution

    if plot_entry is None:
        return score_plot_building_resolution(
            plot_area=None,
            plausible_plot_area_bounds=_PLAUSIBLE_PLOT_AREA_M2,
            plot_is_envelope_reconstruction=False,
            plot_aspect_ratio=1.0,
            building_present=False,
            building_plot_area_ratio=None,
            min_building_plot_area_fraction=_MIN_BUILDING_PLOT_AREA_FRACTION,
            building_is_reconstruction_or_vision=False,
        )

    try:
        plot_aspect_ratio = geo.aspect_ratio(plot_entry.polygon.bounding_box)
    except Exception:
        plot_aspect_ratio = 1.0

    plot_area = plot_entry.polygon.area
    building_present = building_entry is not None
    building_plot_area_ratio = (
        building_entry.polygon.area / plot_area if building_present and plot_area > 0 else None
    )
    building_is_reconstruction_or_vision = building_present and building_entry.layer in (
        _VISION_FALLBACK_LAYER, _RECONSTRUCTED_LAYER, _WALL_UNION_RECONSTRUCTED_LAYER, _ENVELOPE_RECONSTRUCTED_LAYER,
    )

    return score_plot_building_resolution(
        plot_area=plot_area,
        plausible_plot_area_bounds=_PLAUSIBLE_PLOT_AREA_M2,
        plot_is_envelope_reconstruction=plot_entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER,
        plot_aspect_ratio=plot_aspect_ratio,
        building_present=building_present,
        building_plot_area_ratio=building_plot_area_ratio,
        min_building_plot_area_fraction=_MIN_BUILDING_PLOT_AREA_FRACTION,
        building_is_reconstruction_or_vision=building_is_reconstruction_or_vision,
    )


def _score_region_resolution(
    plot_entry: Optional["_RawPoly"], building_entry: Optional["_RawPoly"]
) -> float:
    """Generic plausibility score for one region's resolved plot/building pair.

    Every signal here is a relative/structural property (coverage ratio,
    aspect ratio, whether the building came from exact vector geometry vs.
    a reconstruction) -- never an absolute coordinate, area, or threshold
    tied to any specific plan.

    Delegates to `backend.spatial_reasoning.hypothesis_scoring.
    score_plot_building_resolution` (Architecture V2, Phase 4/5 -- see
    ARCHITECTURE_V2.md Deliverable C.2 item 4), which decomposes this exact
    arithmetic into named, auditable `ConstraintResult` terms so the same
    scoring logic can eventually be shared with the PDF path. This function
    stays the DXF-specific adapter: it is what knows that `plot_entry.layer
    == _ENVELOPE_RECONSTRUCTED_LAYER` means "this is an envelope
    reconstruction", not the shared module. Behavior-preserving refactor --
    `test_dxf_extractor.py`/`test_real_plan_regression.py` are the
    regression floor proving this changed nothing observable.
    """
    return _score_region_resolution_detailed(plot_entry, building_entry)[1]


def _polygon_edge_segments(poly: Polygon) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = [(p.x, p.y) for p in poly.points]
    if len(pts) < 2:
        return []
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _oriented_width_depth(polygon: Polygon) -> tuple[float, float, float]:
    """Measure (width, depth, oriented_rectangularity) from a polygon's own
    MINIMUM-AREA ORIENTED bounding rectangle, never its axis-aligned bbox.

    `polygon.bounding_box.width`/`.height` (the axis-aligned bbox) only
    equals a rectangle's own true side lengths when that rectangle happens
    to be drawn axis-aligned on the sheet. For ANY rotated rectangle --
    including a perfectly correct, genuinely closed native DXF polygon,
    not just an approximated/reconstructed one -- the axis-aligned bbox is
    strictly LARGER than the true side lengths on both axes, which
    silently inflates every reported plot/building width and depth. A real
    architectural sheet being drawn at an arbitrary rotation is common, not
    an edge case, so this is not a cosmetic difference.

    This instead computes the polygon's minimum-area bounding rectangle at
    ANY orientation (`geo.min_area_bounding_rectangle`, rotating calipers
    over the convex hull) and reports its own two distinct side lengths.
    For an already-rectangular polygon this is exact regardless of
    rotation; for an irregular polygon it is a consistent, orientation-
    independent generalization of "width x depth" -- `oriented_
    rectangularity` (polygon area / the fitted rectangle's own area, NOT
    the axis-aligned bbox area used elsewhere in this codebase, since that
    would itself be rotation-dependent) tells the caller how well that
    generalization actually fits, so a genuinely irregular plot/building
    can be reported with reduced confidence instead of a falsely precise
    width/depth pair.

    "Width" vs "depth" assignment keeps this codebase's existing
    convention for the common axis-aligned case (whichever side runs
    closer to horizontal is called "width") purely for continuity with
    already-passing behavior -- there is no frontage/road context
    available at this stage of DXF-only extraction to assign the labels
    semantically instead.
    """
    rect = geo.min_area_bounding_rectangle(polygon.points)
    if rect is None or len(rect.points) < 4:
        bbox = polygon.bounding_box
        return bbox.width, bbox.height, 1.0

    p0, p1, p2 = rect.points[0], rect.points[1], rect.points[2]
    side_a = math.hypot(p1.x - p0.x, p1.y - p0.y)
    side_b = math.hypot(p2.x - p1.x, p2.y - p1.y)
    rect_area = side_a * side_b
    oriented_rectangularity = min(1.0, polygon.area / rect_area) if rect_area > 0 else 1.0

    angle_a = geo.line_orientation_degrees(Line(start=p0, end=p1))
    deviation_from_horizontal = min(angle_a, 180.0 - angle_a)
    if deviation_from_horizontal <= 45.0:
        width, depth = side_a, side_b
    else:
        width, depth = side_b, side_a
    return width, depth, oriented_rectangularity


# How close a dimension-chain-verified edge length must be to the already-
# computed width/depth to count as CONFIRMING that specific field (Critical
# Requirement 5's text -> dimension graphic -> measured span -> geometric
# edge -> semantic object chain). This is deliberately a SEPARATE constant
# from `_EVIDENCE_MATCH_RELATIVE_TOLERANCE` (the global whole-sheet
# evidence-recovery system's own numeric-match tolerance) -- the two are
# unrelated evidence paths (an exact geometric edge-endpoint coincidence
# here vs. a proximity/orientation/length score there) and must not share
# tuning.
_DIMENSION_CHAIN_RELATIVE_TOLERANCE = 0.03


def _dimension_chain_notes(
    dims: list["_RawDim"], polygon: Polygon, width: float, depth: float,
) -> tuple[list[str], list[str], list[str]]:
    """Cross-check `width`/`depth` against native DXF DIMENSION entities
    whose own extension-line points exactly match one of `polygon`'s edges
    (see `dxf_dimension_chain.find_dimension_edge_matches`) -- a genuine
    text -> dimension graphic -> measured span -> geometric edge chain,
    never a nearest-number/text-proximity guess. Returns (width_notes,
    depth_notes, mismatch_warnings): notes to append as CONFIRMING
    evidence for whichever field the matched edge's length agrees with,
    or a warning (not a silent drop, not a forced override) when a
    verified edge's length agrees with neither -- that edge is real
    evidence of something (a room, a setback, an unrelated wall), just
    not this field.
    """
    from backend.cv_extraction.dxf_dimension_chain import find_dimension_edge_matches

    matches = find_dimension_edge_matches(dims, polygon)
    width_notes: list[str] = []
    depth_notes: list[str] = []
    mismatch_warnings: list[str] = []
    for m in matches:
        width_diff = abs(m.value - width)
        depth_diff = abs(m.value - depth)
        width_tol = max(0.05, width * _DIMENSION_CHAIN_RELATIVE_TOLERANCE)
        depth_tol = max(0.05, depth * _DIMENSION_CHAIN_RELATIVE_TOLERANCE)
        label = f"'{m.raw_text}' " if m.raw_text else ""
        if width_diff <= width_tol and width_diff <= depth_diff:
            width_notes.append(
                f"Confirmed by a DXF DIMENSION entity ({label}measured value={m.value:.3f} m) whose own "
                "extension-line points exactly match this polygon's edge -- not a proximity match, the "
                "dimension's own definition points coincide with the edge's own endpoints."
            )
        elif depth_diff <= depth_tol:
            depth_notes.append(
                f"Confirmed by a DXF DIMENSION entity ({label}measured value={m.value:.3f} m) whose own "
                "extension-line points exactly match this polygon's edge -- not a proximity match, the "
                "dimension's own definition points coincide with the edge's own endpoints."
            )
        else:
            mismatch_warnings.append(
                f"A DXF DIMENSION entity's extension-line points exactly match a polygon edge "
                f"({label}measured value={m.value:.3f} m), but that value matches neither the computed "
                f"width ({width:.3f} m) nor depth ({depth:.3f} m) within tolerance -- likely dimensioning "
                "a different span (a room, a setback, another wall) on the same polygon; not used to "
                "adjust width/depth confidence."
            )
    return width_notes, depth_notes, mismatch_warnings


# Recovered dimension evidence is trusted as a numeric match to a resolved
# candidate's own width/depth/area only within this relative tolerance --
# generous enough to absorb OCR digit/decimal misreads and rounding, tight
# enough that an unrelated coincidental number can't casually qualify.
_EVIDENCE_MATCH_RELATIVE_TOLERANCE = 0.08

# Two DIFFERENT recovered items can both fall within
# `_EVIDENCE_MATCH_RELATIVE_TOLERANCE` of the same candidate field --
# picking whichever happened to come first in iteration order (the
# original behavior here) is an arbitrary, order-dependent guess dressed
# as a confirmation. These two constants distinguish a genuine conflict
# (refuse the field entirely) from ordinary OCR noise around one real
# reading (still resolve to the stronger one): TODO -- not yet validated
# against a real conflicting-evidence example (none has been observed on
# either real fixture so far; the values below are principled defaults,
# not measured ones, and should be re-examined once one is found).
_EVIDENCE_CONFLICT_WEIGHT_RATIO = 0.7  # a runner-up below this fraction of the best match's specificity is noise, not a competitor
_EVIDENCE_SAME_READING_RELATIVE_TOLERANCE = 0.03  # within this of each other: one real annotation misread twice, not two different ones

# Which independent axis each candidate field belongs to, for the
# single-axis discount in `_score_recovered_evidence` -- width and depth
# are independent, separately-derived measurements of the same candidate
# (as are the plot and building versions of each), and area is a third,
# independent axis again (not simply width*depth for an envelope/
# reconstructed candidate, which is fitted directly from points rather
# than computed from the two side lengths).
_EVIDENCE_FIELD_AXIS = {
    "plot.width": "width", "building.width": "width",
    "plot.depth": "depth", "building.depth": "depth",
    "plot.area": "area", "building.footprint_area": "area",
}

# A candidate confirmed on only ONE independent axis (e.g. depth alone,
# width unconfirmed) gets its evidence bonus discounted by this factor --
# see `_score_recovered_evidence`'s inline comment for the real failure
# mode this targets. Confirmed directly that a partial discount (0.4) is
# not enough: on a real bundled DXF regression fixture, two candidates'
# base structural scores were close enough (4.228 vs 4.255) that even a
# heavily-discounted single-axis bonus (~0.76 from an undiscounted ~1.9)
# was still large enough to flip the winner back to the single-axis
# candidate. A lone digit within tolerance of ONE property is exactly the
# kind of coincidence 121 recovered OCR items on a noisy sheet produce
# often enough that it must not be able to move the score AT ALL on its
# own -- only independent corroboration across both of a candidate's
# spatial extents (width AND depth) is trusted as real evidence.
_SINGLE_AXIS_EVIDENCE_DISCOUNT = 0.0

# How far outside a region's own tight clustered bbox a recovered text
# item's position may still fall and be considered evidence FOR that
# region's numeric-match bonus -- matches the padding convention already
# used for the Vision height-focus region crop elsewhere in this codebase.
# Safe to widen here (unlike widening OCR capture itself, tried and
# reverted -- see `_recover_sheet_wide_evidence`'s docstring): this only
# changes which regions get to check a single, already-uniquely-recognized
# label against their own numbers, not how many times that label gets
# independently (re)discovered.
_EVIDENCE_MEMBERSHIP_MARGIN_FRACTION = 0.25

# Evidence recovery (render + OCR, once per document) is only attempted
# when at least this fraction of the extractor's overall wall-clock
# budget still remains after structural region resolution -- see
# `_resolve_via_regions`'s inline comment. Below this, skip evidence and
# keep the structural-only result rather than risk the whole extraction
# exceeding its timeout and returning nothing at all.
_MIN_REMAINING_TIME_FRACTION_FOR_EVIDENCE = 0.30

# A merge of two genuinely complementary fragments of ONE real boundary
# should pool into an area comparable to the sum of their own (individually
# incomplete) areas, not substantially larger -- see
# `_merge_candidate_region_pairs`'s inline comment. A small allowance above
# an exact 1:1 sum accounts for ordinary fit-quality slack (rounding, minor
# reconstruction imprecision), not a license for a much larger result.
_MAX_MERGE_AREA_SUM_RATIO = 1.15

# Region-merge testing's share of whatever time remains before it runs (see
# `_resolve_via_regions`'s inline comment for why this is deliberately a
# minority share, asymmetric in evidence recovery's favor).
_REGION_MERGE_BUDGET_SHARE = 0.20

# Render-resolution tiers for sheet-wide OCR evidence recovery (see
# `_recover_sheet_wide_evidence`). Cost does not scale linearly with either
# entity count or resolution (measured directly on a real ~22,500-entity,
# no-native-text sheet: 1400px -> 0 items in 1.3s; 3000px -> 95 items, MOSTLY
# NOISE, in 9.3s; 5000px -> 121 items including the exact correct printed
# dimensions plus the sheet's own scale annotation, in 41.3s). Note that the
# 3000px midpoint is NOT "partially legible" in any useful sense -- it
# recovered noise plus at most one of two true values; only 5000px recovered
# BOTH true dimensions cleanly. The OLD design picked a single fixed
# resolution purely from entity count to bound cost, never validated against
# actual OCR legibility at the chosen resolution -- on exactly the large/
# fragmented files that most need this evidence, it silently recovered
# nothing at all. The fix keeps the same entity-count-based STARTING tier
# (so typical cost stays exactly as before for every document), and adds a
# single budget-gated escalation, straight to the high-fidelity ceiling
# rather than an intermediate step, when the starting attempt recovered
# literally nothing -- an intermediate resolution has no evidence behind it
# of being any more useful than the cheap tier, whereas the ceiling is the
# one resolution directly measured to work.
_RENDER_RESOLUTION_TIERS = (1400, 2200, 3000)
_HIGH_FIDELITY_ESCALATION_PX = 5000

# The ceiling has empirically cost roughly 30x the cheapest starting tier
# on a real sheet (1400px->5000px: 1.3s -> 41.3s). Require the remaining
# budget to cover this multiple of the just-observed starting-attempt cost
# before escalating (a conservative bound regardless of which starting tier
# was actually used, since escalating from a higher starting tier is
# strictly cheaper than this), so escalation is only attempted when it can
# plausibly finish -- never a blind gamble that could itself blow the
# overall extraction timeout.
_ESCALATION_PROJECTED_COST_GROWTH_FACTOR = 35.0


def _expand_bbox(bbox: BoundingBox, margin_fraction: float) -> BoundingBox:
    mx = max(bbox.width * margin_fraction, 1e-9)
    my = max(bbox.height * margin_fraction, 1e-9)
    return BoundingBox(min_x=bbox.min_x - mx, min_y=bbox.min_y - my, max_x=bbox.max_x + mx, max_y=bbox.max_y + my)


def _recover_sheet_wide_evidence(
    all_polygon_points: list[list[tuple[float, float]]],
    all_open_chain_points: list[list[tuple[float, float]]],
    region_candidate_edges: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]],
    region_bboxes: dict[int, BoundingBox],
    warnings: list[str],
    remaining_budget_seconds: Optional[float] = None,
) -> tuple[dict[int, list], dict[int, list[str]]]:
    """Run vectorized-text/OCR recovery EXACTLY ONCE for the whole sheet,
    then associate each recovered value against the UNION of every
    region's own candidate edges simultaneously.

    This exists to fix two things a naive per-region "render+OCR each
    region separately" design gets wrong:

      1. COST: OCR (render + tesseract) is real wall-clock cost. Running
         it once per candidate region made a real, large, many-region
         sheet exceed this project's own extraction timeout, and bounding
         it to only the top-N structurally-scoring regions (an earlier
         version of this function) meant a region that scored poorly on
         geometry ALONE could never have its evidence looked at even if
         the sheet's own printed numbers would have confirmed it was
         actually correct -- evidence could only ever demote a candidate
         already in contention, never promote one that structural scoring
         alone had wrongly excluded. Running OCR once, up front, for every
         region at once removes that ordering problem entirely: every
         region gets the same evidence-scoring opportunity, at a fraction
         of the previous total cost (measured: ~10-30s once for a whole
         sheet, vs. 90-170s for 3-14 separate per-region passes).
      2. CORRECTNESS: when two regions' own capture areas were widened and
         OCR'd independently (an approach tried and reverted -- see git
         history), the SAME printed label sitting between two candidates
         got independently "discovered" and associated with a different,
         unrelated edge in each region's own separate call, producing
         several mutually-inconsistent answers. Associating against the
         FULL union of every region's edges in one pass instead lets
         `dimension_candidates`'s own distance/orientation scoring resolve
         the genuine nearest/best match ONCE, across all candidates at
         once, rather than each region competing from inside its own
         blinkered view.

    Returns ({region_id: [RecoveredDimension, ...]}, {region_id: [caption_text, ...]})
    for every region that had at least one candidate edge to associate
    against -- the second dict holds any recognized drawing-type caption
    (e.g. "SITE PLAN") whose OCR'd position falls within that region, read
    off the SAME OCR pass as the dimension evidence at no extra cost (see
    `recover_dimension_and_caption_evidence`).

    `remaining_budget_seconds` (if provided by the caller, see
    `_resolve_via_regions`) is the actual wall-clock seconds left in this
    document's overall extraction budget at the moment evidence recovery is
    about to start -- used only to decide whether a single escalation to
    `_HIGH_FIDELITY_ESCALATION_PX` is affordable after the starting attempt,
    never to change the starting resolution itself (see
    `_RENDER_RESOLUTION_TIERS`).
    """
    from backend.cv_extraction.dxf_text_recovery import recover_dimension_and_caption_evidence

    combined_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    edge_owner: list[int] = []
    for region_id, edges in region_candidate_edges.items():
        combined_edges.extend(edges)
        edge_owner.extend([region_id] * len(edges))

    by_region: dict[int, list] = {region_id: [] for region_id in region_candidate_edges}
    captions_by_region: dict[int, list[str]] = {region_id: [] for region_id in region_candidate_edges}
    if not combined_edges:
        return by_region, captions_by_region

    # Render/OCR cost does not scale linearly with entity count -- measured
    # directly: ~10s for an ~18,700-entity sheet at 3000px, ~57s for an
    # ~27,000-entity sheet at the SAME resolution (a denser render gives
    # tesseract more candidate text-like regions to examine, not just a
    # bigger image to read). Since this whole extraction runs under a
    # single overall wall-clock budget (`bounded_execution.run_with_
    # timeout` in `DXFHybridExtractor.extract`), a large sheet's own
    # structural resolution pass can already consume most of that budget
    # before evidence recovery even starts -- render resolution STARTS
    # scaled down for a larger sheet so the first attempt's own cost stays
    # bounded regardless of how many entities the sheet has.
    total_entities = len(all_polygon_points) + len(all_open_chain_points)
    if total_entities > 20000:
        start_px = 1400
    elif total_entities > 10000:
        start_px = 2200
    else:
        start_px = 3000

    try:
        t0 = time.time()
        recovered, captions = recover_dimension_and_caption_evidence(
            all_polygon_points, all_open_chain_points, combined_edges, _SITE_PLAN_CAPTION_KEYWORD_GROUPS,
            render_target_px=start_px,
        )
        elapsed = time.time() - t0
    except Exception as exc:
        warnings.append(f"Sheet-wide vectorized-text recovery failed: {exc}")
        return by_region, captions_by_region

    # The starting (cost-bounded) attempt found literally nothing -- on a
    # large/fragmented, no-native-text sheet this was previously a silent
    # dead end (the evidence-recovery step exists specifically to catch
    # exactly this kind of file, and was never checked against whether its
    # chosen resolution could actually read anything on one). Escalate
    # straight to the high-fidelity ceiling -- not an intermediate
    # resolution, which measurement showed is not reliably any more useful
    # than the cheap tier -- but only when the actual just-observed cost of
    # the attempt that just ran leaves enough of the remaining budget to
    # plausibly afford it. This reacts to this document's own measured
    # cost, not a guessed formula, and can never be attempted more than
    # once per document.
    if not recovered and _HIGH_FIDELITY_ESCALATION_PX > start_px:
        projected_cost = elapsed * _ESCALATION_PROJECTED_COST_GROWTH_FACTOR
        remaining_after_start = (
            remaining_budget_seconds - elapsed if remaining_budget_seconds is not None else None
        )
        if remaining_after_start is not None and projected_cost <= remaining_after_start:
            try:
                recovered, escalated_captions = recover_dimension_and_caption_evidence(
                    all_polygon_points, all_open_chain_points, combined_edges, _SITE_PLAN_CAPTION_KEYWORD_GROUPS,
                    render_target_px=_HIGH_FIDELITY_ESCALATION_PX,
                )
                captions = captions + escalated_captions
                warnings.append(
                    f"Sheet-wide vectorized-text recovery found nothing at {start_px}px; escalated to "
                    f"{_HIGH_FIDELITY_ESCALATION_PX}px (measured {elapsed:.1f}s for the first attempt, "
                    f"{remaining_after_start:.0f}s of budget remained) and recovered "
                    f"{len(recovered)} item(s)."
                )
            except Exception as exc:
                warnings.append(f"Sheet-wide vectorized-text recovery (escalated) failed: {exc}")

    # A caption describes exactly ONE drawing -- unlike a dimension's own
    # looser, margin-expanded membership check above (deliberately generous
    # because several DIFFERENT regions may legitimately want to check a
    # dimension label that sits in the gap between them against their own
    # numbers), a caption must never be handed to more than one region.
    # Using that same margin-expanded check here was confirmed directly to
    # over-attribute: three separate, unrelated regions' expanded bboxes
    # all reached far enough to swallow the SAME single caption instance,
    # handing an unrelated floor-plan region the exact bonus meant to
    # single out the actual site plan. Ownership here is instead: whichever
    # region's own TIGHT (unexpanded) bbox contains the caption's position,
    # breaking a tie (nested/overlapping regions) by smallest bbox area;
    # falling back to the single CLOSEST region by center distance only
    # when no tight bbox contains it at all (e.g. the caption sits just
    # outside its own drawing's clustered extent).
    for cap in captions:
        cx = (cap.world_bbox[0] + cap.world_bbox[2]) / 2.0
        cy = (cap.world_bbox[1] + cap.world_bbox[3]) / 2.0
        containing = [
            region_id for region_id, bbox in region_bboxes.items()
            if bbox.min_x <= cx <= bbox.max_x and bbox.min_y <= cy <= bbox.max_y
        ]
        if containing:
            owner = min(containing, key=lambda rid: (region_bboxes[rid].max_x - region_bboxes[rid].min_x) * (region_bboxes[rid].max_y - region_bboxes[rid].min_y))
        elif region_bboxes:
            owner = min(
                region_bboxes,
                key=lambda rid: math.hypot(
                    (region_bboxes[rid].min_x + region_bboxes[rid].max_x) / 2.0 - cx,
                    (region_bboxes[rid].min_y + region_bboxes[rid].max_y) / 2.0 - cy,
                ),
            )
        else:
            continue
        captions_by_region.setdefault(owner, []).append(cap.text)

    import dataclasses

    for r in recovered:
        # A NUMERIC-VALUE match (scored in `_score_recovered_evidence`)
        # does not require this text to be geometrically associated with
        # one of a region's own candidate EDGES -- on a badly fragmented
        # drawing, a candidate's edges are themselves only an
        # approximation (an envelope, a reconstruction), so requiring
        # exact edge proximity before a printed number is even considered
        # would silently discard genuine confirming evidence purely
        # because the candidate geometry it should confirm is imprecise.
        # Membership here is instead "this text's own recognized position
        # sits within (a margin around) this region's own clustered
        # extent" -- looser, but still scoped to specific drawings, never
        # "found somewhere on the whole sheet."
        #
        # A margin (same 25% convention used for the Vision height-focus
        # crop elsewhere in this codebase) is safe to use HERE in a way it
        # was NOT safe to use for OCR capture itself (see the "widening
        # OCR capture" attempt earlier -- reverted after it caused the
        # same discovered label to be re-associated inconsistently across
        # overlapping per-region OCR calls): OCR discovery now happens
        # EXACTLY ONCE, sheet-wide, so there is only one canonical
        # recognized instance of any given label to begin with -- widening
        # membership only changes which regions get a chance to check
        # their OWN numbers against it, and a false match still requires
        # this region's own derived value to coincidentally agree with the
        # recognized number within the numeric-match tolerance.
        cx = (r.world_bbox[0] + r.world_bbox[2]) / 2.0
        cy = (r.world_bbox[1] + r.world_bbox[3]) / 2.0
        owner_region = edge_owner[r.associated_segment_index] if r.associated_segment_index is not None else None

        # The OWNER region (whichever region's own edge this text was
        # actually associated with, via `dimension_candidates`' distance/
        # orientation scoring) is always included -- that association is a
        # more precise, geometry-aware signal than a crude bbox check
        # could ever be, and a dimension label is routinely offset a
        # visible gap away from the edge/line it labels, so it can easily
        # sit just outside even a margin-padded region bbox. A bbox check
        # must never be able to override or exclude a real, closer-scored
        # association the ONE global recovery pass already made.
        included_regions = {owner_region} if owner_region is not None else set()
        for region_id, tight_bbox in region_bboxes.items():
            bbox = _expand_bbox(tight_bbox, _EVIDENCE_MEMBERSHIP_MARGIN_FRACTION)
            if bbox.min_x <= cx <= bbox.max_x and bbox.min_y <= cy <= bbox.max_y:
                included_regions.add(region_id)

        for region_id in included_regions:
            if owner_region is not None and owner_region != region_id:
                # This item WAS associated with a DIFFERENT region's edge
                # -- keep it for THIS region's numeric-match bonus (the
                # recognized value is still real evidence), but strip the
                # association so it can never seed a scale correction
                # using an edge length that has nothing to do with this
                # region's own geometry.
                by_region.setdefault(region_id, []).append(
                    dataclasses.replace(r, associated_segment_index=None, associated_segment_length=None)
                )
            else:
                by_region.setdefault(region_id, []).append(r)

    return by_region, captions_by_region


def _specificity(r) -> float:
    # A bare short integer ("2", "7", "8" ...) is common OCR noise/glyph
    # fragments and coincides with a plausible small dimension far too
    # often to count as strong evidence on its own -- weight a match by
    # how SPECIFIC the recovered text actually is (decimal precision, an
    # explicit unit, digit count, OCR confidence).
    score = 0.4
    if "." in r.raw_text:
        score += 1.0
    if r.unit_hint is not None:
        score += 1.0
    digit_count = sum(ch.isdigit() for ch in r.raw_text)
    if digit_count >= 3:
        score += 0.5
    return min(3.0, score) * max(0.2, r.confidence)


def _match_recovered_evidence(
    recovered: list, plot_entry: Optional["_RawPoly"], building_entry: Optional["_RawPoly"],
) -> list[tuple[str, object, float, float]]:
    """Find recovered items that numerically confirm this candidate's own
    plot/building width, depth, or area, within `_EVIDENCE_MATCH_
    RELATIVE_TOLERANCE`. Returns `[(field, recovered_item, weight,
    matched_value), ...]` -- at most one match per field (the best-
    specificity one), the same matching discipline used everywhere
    evidence confirmation matters. Shared by `_score_recovered_evidence`
    (bonus scoring) and `_confirmed_axis_values` (region-merge growth
    veto check) so both agree on what counts as "confirmed".
    """
    candidate_values: list[tuple[str, float]] = []
    if plot_entry is not None:
        b = plot_entry.polygon.bounding_box
        candidate_values += [
            ("plot.width", b.width), ("plot.depth", b.height), ("plot.area", plot_entry.polygon.area),
        ]
    if building_entry is not None:
        b = building_entry.polygon.bounding_box
        candidate_values += [
            ("building.width", b.width), ("building.depth", b.height),
            ("building.footprint_area", building_entry.polygon.area),
        ]

    matched: list[tuple[str, object, float, float]] = []
    for field, candidate_value in candidate_values:
        if candidate_value <= 0:
            continue
        within_tolerance: list[tuple[float, object, float]] = []
        for r in recovered:
            # An unlabeled recovered number is treated as already being in
            # this candidate's own (already-scaled) units -- consistent
            # with how this file's own unit resolution already treats a
            # bare number once some scale has been chosen; a value WITH an
            # explicit unit is converted to metres properly instead.
            value = r.value_metres if r.value_metres is not None else r.value
            if value is None or value <= 0:
                continue
            rel_diff = abs(value - candidate_value) / candidate_value
            if rel_diff > _EVIDENCE_MATCH_RELATIVE_TOLERANCE:
                continue
            within_tolerance.append((_specificity(r), r, value))
        if not within_tolerance:
            continue
        within_tolerance.sort(key=lambda m: m[0], reverse=True)
        best_weight, best_r, best_value = within_tolerance[0]
        # Refuse rather than guess: a second, comparably-specific reading
        # that disagrees with the best one by more than ordinary OCR
        # jitter is a genuine conflict -- two different recovered numbers
        # both plausibly measuring this field, with no principled way to
        # prefer one. Confidently picking either (the previous behavior,
        # via first-seen tie-break) would misrepresent an ambiguous read
        # as a confirmed one, so this field gets NO match at all rather
        # than a coin flip dressed as confirmation.
        conflicting = [
            (w, r, v) for w, r, v in within_tolerance[1:]
            if w >= best_weight * _EVIDENCE_CONFLICT_WEIGHT_RATIO
            and abs(v - best_value) / max(abs(best_value), 1e-9) > _EVIDENCE_SAME_READING_RELATIVE_TOLERANCE
        ]
        if conflicting:
            continue
        matched.append((field, best_r, best_weight, best_value))
    return matched


def _confirmed_axis_values(
    recovered: list, plot_entry: Optional["_RawPoly"], building_entry: Optional["_RawPoly"] = None,
) -> dict[str, float]:
    """Width/depth values for this candidate that are confirmed by
    recovered evidence on BOTH independent spatial axes together (never
    just one -- see `_SINGLE_AXIS_EVIDENCE_DISCOUNT`'s docstring for why a
    single-axis match is not trustworthy enough to act on), by two
    DIFFERENT recovered items. Requiring distinct items closes a real gap
    found directly on a bundled DXF regression fixture: a near-square
    candidate (width and depth both close to the same value) let ONE
    coincidental recovered digit satisfy both fields at once, wrongly
    treated as independent two-axis corroboration when it was really one
    weak signal counted twice -- the same failure mode `_score_recovered_
    evidence`'s own item-identity deduplication already guards against for
    bonus scoring. Returns `{"width": value, "depth": value}` only when
    both are present AND independently sourced; otherwise `{}`. Used by
    `_merge_candidate_region_pairs` to veto a growth step that would move
    a dimension AWAY from a value already confirmed for one of its
    parents.
    """
    if not recovered or plot_entry is None:
        return {}
    matched = _match_recovered_evidence(recovered, plot_entry, building_entry)
    by_axis: dict[str, tuple[float, int]] = {}
    for field, r, _weight, value in matched:
        axis = _EVIDENCE_FIELD_AXIS.get(field)
        # Prefer the plot.* reading over building.* for the same axis --
        # growth changes plot boundaries, so that's the more directly
        # relevant confirmed reference value.
        if axis in ("width", "depth") and (axis not in by_axis or field.startswith("plot.")):
            by_axis[axis] = (value, id(r))
    if not {"width", "depth"} <= by_axis.keys():
        return {}
    if by_axis["width"][1] == by_axis["depth"][1]:
        return {}  # same item "confirmed" both -- not independent corroboration
    return {axis: value for axis, (value, _item_id) in by_axis.items()}


def _score_recovered_evidence(
    recovered: list,
    plot_entry: Optional["_RawPoly"],
    building_entry: Optional["_RawPoly"],
    warnings: list[str],
    context_label: str,
) -> tuple[float, Optional[float]]:
    """Score one region's ALREADY-RECOVERED, already-associated dimension
    evidence (see `_recover_sheet_wide_evidence`) against its own
    candidate plot/building:

      1. a SCORE BONUS when a recovered value numerically matches one of
         THIS candidate's own derived properties (width/depth/area) --
         the strongest possible evidence a candidate is correct: the
         sheet's own printed number confirms the specific geometry, not
         merely "a number exists somewhere nearby";
      2. a SCALE CORRECTION when independently-agreeing recovered values
         carry an explicit unit and are geometrically associated with this
         candidate's own boundary edges -- an independently measured
         scale, not the generic plausible-area guess `_resolve_unit_factor`
         falls back to.

    Pure scoring, no OCR call -- cheap enough to run for every region.
    """
    from backend.cv_extraction.dxf_text_recovery import scale_candidates_from_recovered_dimensions

    if not recovered:
        return 0.0, None

    matches = _match_recovered_evidence(recovered, plot_entry, building_entry)

    matched_fields: list[tuple[str, str, float]] = []
    matched_axes: set[str] = set()
    # Keyed by recovered-item identity: the SAME recovered text must not be
    # counted once per field it happens to satisfy. A candidate whose
    # width/depth/building.width/building.depth are all close to each
    # other (a near-square, likely-wrong-shaped candidate) can otherwise
    # have ONE recovered number satisfy all four independently, inflating
    # its bonus several-fold over a candidate confirmed by several
    # genuinely distinct pieces of evidence -- confirmed directly as a
    # real risk on a bundled DXF regression fixture. Each unique item
    # contributes its weight at most once, at whichever field it best
    # matches.
    item_best: dict[int, float] = {}
    for field, r, weight, value in matches:
        matched_fields.append((field, r.raw_text, value))
        matched_axes.add(_EVIDENCE_FIELD_AXIS.get(field, field))
        key = id(r)
        if key not in item_best or weight > item_best[key]:
            item_best[key] = weight

    bonus = sum(item_best.values())

    # A candidate whose OWN dimensions happen to be close to just ONE
    # recovered number (only depth matched, say, with width unconfirmed)
    # is much weaker evidence than one confirmed on BOTH of its two
    # independent spatial extents together -- directly confirmed as a real
    # failure mode on a bundled DXF regression fixture: a single OCR
    # misread digit (a "9" read as a "7") fell within tolerance of one
    # wrong candidate's own depth and was enough, alone, to outscore a
    # genuinely better-merged candidate that had not (yet) accumulated a
    # matching bonus of its own. Specifically width AND depth, not "any
    # two of width/depth/area": area is not independent of the other two
    # for a roughly-rectangular envelope/reconstructed candidate (it is
    # fitted from the same point cloud, not read as a third, unrelated
    # measurement), so an area+depth match alone is not the same strength
    # of corroboration as width+depth confirmed independently of each
    # other.
    if not {"width", "depth"} <= matched_axes:
        bonus *= _SINGLE_AXIS_EVIDENCE_DISCOUNT

    if plot_entry is not None and building_entry is not None:
        plot_area = plot_entry.polygon.area
        if plot_area > 0 and building_entry.polygon.area / plot_area > 0.95:
            # A real site plan always has SOME setback -- a building
            # reported as (almost) the exact same polygon as its plot is
            # a degenerate resolution (e.g. the same fragment cluster got
            # picked for both roles), not a real zero-setback building,
            # regardless of how many coincidental digit matches it collects.
            bonus -= 5.0

    if matched_fields:
        warnings.append(
            f"[{context_label}] Recovered vectorized-text evidence confirms: "
            + "; ".join(f"{field}~={val:.3f} (printed '{txt}')" for field, txt, val in matched_fields)
        )

    scale_correction = None
    scale_pairs = scale_candidates_from_recovered_dimensions(recovered)
    if len(scale_pairs) >= 2:
        # A SINGLE recovered (value, association) pair is not enough to
        # trust as a scale correction on its own -- measured directly on a
        # real file: a single ambiguous OCR/association match produced a
        # wrong, overconfident correction more often than a right one
        # (the same printed "10m" road-width label mis-associating with an
        # unrelated nearby edge). Only apply a correction when at least
        # TWO independent recovered dimensions in this region agree with
        # each other on the implied scale (within a tight tolerance) --
        # independent agreement is what makes this trustworthy evidence
        # rather than one coincidental association.
        ratios = [ratio for ratio, _ev in scale_pairs]
        ratios_sorted = sorted(ratios)
        median = ratios_sorted[len(ratios_sorted) // 2]
        agreeing = [
            (ratio, ev) for ratio, ev in scale_pairs
            if median > 0 and abs(ratio - median) / median <= 0.10
        ]
        if len(agreeing) >= 2 and 0.2 <= median <= 5.0 and abs(median - 1.0) > 0.05:
            scale_correction = median
            evidence_desc = "; ".join(
                f"'{ev.raw_text}' ({ev.value_metres:.3f} m over a {ev.associated_segment_length:.3f}-unit edge)"
                for _ratio, ev in agreeing
            )
            warnings.append(
                f"[{context_label}] Scale correction from {len(agreeing)} independently agreeing recovered "
                f"dimensions ({evidence_desc}): implies a {scale_correction:.4f}x correction to this "
                "region's measurements."
            )

    return bonus, scale_correction


def _evidence_veto_reason(
    evidence_by_region: dict[int, list],
    region_a: "DxfRegion", region_b: "DxfRegion",
    plot_a: Optional["_RawPoly"], plot_b: Optional["_RawPoly"],
    merged_plot: Optional["_RawPoly"],
) -> Optional[str]:
    """Veto a growth step that would move width or depth AWAY from a value
    already confirmed (on both independent axes -- see
    `_confirmed_axis_values`) by recovered evidence for one of its two
    inputs. Structural plausibility alone is not ground truth: a merge can
    score better while still moving further from what the sheet's own
    printed dimensions say, and growth has no other way to notice that.
    """
    if merged_plot is None:
        return None
    merged_bbox = merged_plot.polygon.bounding_box
    merged_dims = {"width": merged_bbox.width, "depth": merged_bbox.height}
    for region_x, plot_x in ((region_a, plot_a), (region_b, plot_b)):
        if plot_x is None:
            continue
        confirmed = _confirmed_axis_values(evidence_by_region.get(region_x.id, []), plot_x)
        if not confirmed:
            continue
        parent_bbox = plot_x.polygon.bounding_box
        parent_dims = {"width": parent_bbox.width, "depth": parent_bbox.height}
        for axis, confirmed_value in confirmed.items():
            parent_gap = abs(parent_dims[axis] - confirmed_value)
            merged_gap = abs(merged_dims[axis] - confirmed_value)
            if merged_gap > parent_gap:
                return (
                    f"region {region_x.id}'s own {axis} ({parent_dims[axis]:.3f}) was confirmed by "
                    f"recovered evidence near {confirmed_value:.3f}; merging would move {axis} to "
                    f"{merged_dims[axis]:.3f}, farther from that confirmed value"
                )
    return None


def _merge_candidate_region_pairs(
    structural: list[tuple[float, "DxfRegion", Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"]]],
    polygons: list["_RawPoly"],
    open_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    frame_indices: set[int],
    warnings: list[str],
    time_budget_seconds: float,
    evidence_by_region: Optional[dict[int, list]] = None,
) -> list[tuple[float, "DxfRegion", Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"]]]:
    """Agglomeratively grow the sheet's structurally-highest-scoring regions
    into pooled candidates, and return every merge along the way that is
    actually better than both of its own inputs.

    Root cause this targets (confirmed on real bundled DXF regression
    fixtures): density-based region clustering assumes a real drawing's
    interior is locally dense and only thin CONNECTING strokes between
    unrelated drawings are sparse. That is false for a large, sparse,
    uniformly-traced dash-dot boundary -- such a boundary has LOW local
    density along its own straight runs (each dash is isolated; the next
    one over might be a full cell-radius away) and only spikes in density
    at corners, so density-erosion can split one real boundary into
    several disjoint regions along its own straight sides -- not
    necessarily just TWO. A real bundled fixture's boundary was split into
    four regions, not two; a fixed "try pairs only" search can only ever
    recover part of a boundary split more ways than that. Growth here has
    no fixed combination size: it keeps adding whichever next region
    improves the current best candidate, for as many rounds as improvement
    keeps happening, so a 2-way, 3-way, or wider split is all handled by
    the same mechanism without guessing a combination count up front.

    This is deliberately NOT a spatial-proximity merge heuristic (bbox
    containment/overlap were both tried and reverted earlier in this
    project's history -- see git history and this module's own frame-
    exclusion logic for why a purely geometric relationship between two
    regions is not a safe merge signal on its own). Instead, each
    candidate merge is tested by literally re-running the SAME resolution
    used for every individual region (`_resolve_plot_building_road` +
    `_score_region_resolution`) on the pooled polygons/segments, and is
    only ever kept when that pooled resolution scores STRICTLY better than
    BOTH of its own inputs' scores, its area is not implausibly larger
    than the sum of its inputs' areas (see the area-sanity check inline
    below -- guards against an incoherent union of unrelated content
    sprawling across the empty space between two genuinely separate
    regions, confirmed as a real failure mode when area was not checked),
    AND it does not move a dimension AWAY from a value recovered OCR
    evidence already confirmed for one of its own inputs (see
    `_evidence_veto_reason` -- structural plausibility alone can favor a
    merge that scores better while still drifting further from what the
    sheet's own printed dimensions say; confirmed as a real failure mode
    directly: growing a correctly-improving 2-region merge with a third
    region raised its structural score further while moving its depth
    from 9.1% error to 15.5% error against the true printed value). A
    merge can therefore only ever ADD a candidate to the pool
    `_resolve_via_regions` already chooses the best of -- it can never
    replace a working single-region result with a worse one.

    Algorithm (single-linkage agglomerative clustering, using "does
    re-resolving the union score better than both parts" as the merge
    criterion instead of a distance metric): start with the top
    `settings.max_dxf_region_merge_candidates` individual regions by their
    own score. Each round, test every pair of CURRENT clusters (which may
    themselves already be merges from an earlier round), keep the single
    best-scoring pair that passes both gates, and replace those two
    clusters with the new merged one for the next round. Stop when no pair
    improves on both its inputs, when only one cluster remains, or when
    `time_budget_seconds` of actual measured wall-clock time has been
    spent -- a real, per-file structural resolution pass can already
    consume most of the overall extraction budget (real bundled DXF
    regression fixtures measured 90-150s of a 180s budget for structural
    resolution alone), so a fixed "try everything or nothing" policy would
    make this feature's benefit depend on incidental machine timing rather
    than degrading gracefully: grow as far as the actual remaining budget
    allows, most-promising pairs first, and keep whatever was already
    found the moment time runs out.

    The growth loop itself (budget-bounded rounds, priority-vs-evidence-
    extra pair ordering, "keep the single best pair per round") now lives
    in `backend.spatial_reasoning.hypothesis_generation.grow_clusters`
    (Architecture V2, Phase 4/5 -- see ARCHITECTURE_V2.md Deliverable C.2
    item 3): this function is the DXF-specific adapter that translates
    `DxfRegion`/`_RawPoly` bookkeeping into the generic callables that
    module expects. Behavior-preserving refactor -- `test_dxf_extractor.py`
    (including the exact warning-text assertion in
    `test_merge_accepts_two_regions_that_together_resolve_plot_and_building`)
    is the regression floor proving this changed nothing observable, one
    deliberate cosmetic exception: a TRIAL (not-yet-accepted) pooled
    candidate's internal `context_label` passed to `_resolve_plot_
    building_road` is now a generic string instead of naming the two
    source region ids -- that label only ever affects fallback-path
    warning TEXT emitted during resolution, never a returned value, and no
    test asserts on it.
    """
    from backend.config import get_settings
    from backend.cv_extraction.dxf_regions import DxfRegion, _bbox_union
    from backend.spatial_reasoning.hypothesis_generation import HypothesisCluster, grow_clusters

    limit = get_settings().max_dxf_region_merge_candidates
    evidence_by_region = evidence_by_region if evidence_by_region is not None else {}
    # Only regions that resolved to SOMETHING are worth pooling -- merging
    # in an empty/unresolved region can never improve a pooled score.
    resolved_entries = [entry for entry in structural if entry[2] is not None]
    by_score = sorted(resolved_entries, key=lambda entry: entry[0], reverse=True)
    top = by_score[:limit]
    # A region that is genuinely one fragment of a boundary split by
    # density-based clustering (this function's own reason for existing --
    # see its docstring) will, by definition, usually resolve poorly on its
    # own: an isolated arc/run of a dash-dot boundary looks like a bad plot
    # candidate in isolation, however plausible it becomes once pooled with
    # its missing other fragments. Capping candidacy to the top `limit`
    # regions by standalone score alone would make growth structurally
    # incapable of ever recovering such a fragment, no matter how much time
    # budget is available, whenever confirmed OCR/vectorized-text evidence
    # (already recovered before this function runs) has flagged it as part
    # of the right answer. So any region carrying confirmed evidence is
    # always eligible for growth, regardless of its own rank.
    selected_ids = {entry[1].id for entry in top}
    evidence_extras = [
        entry for entry in by_score
        if entry[1].id not in selected_ids and evidence_by_region.get(entry[1].id)
    ]
    if len(top) + len(evidence_extras) < 2:
        return []

    next_synthetic_id_box = [max((r.id for _s, r, _p, _b, _rd in structural), default=0) + 1]

    def _next_id() -> int:
        value = next_synthetic_id_box[0]
        next_synthetic_id_box[0] += 1
        return value

    def _to_cluster(entry, is_priority: bool) -> "HypothesisCluster":
        score, region, plot_entry, building_entry, road_entry = entry
        return HypothesisCluster(
            id=region.id, score=score, resolved=(plot_entry, building_entry, road_entry),
            payload=region, is_priority=is_priority,
        )

    # Pairs drawn purely from the original top-`limit` by-score regions go
    # first, in the same score-sum order as before the evidence carve-out
    # existed. Pairs involving an evidence-only extra are appended after:
    # they get first claim on whatever time is left once every pair among
    # the top-scoring regions has had its turn, never before -- so an extra
    # can never starve out an already-working merge among the core group.
    initial_clusters = [_to_cluster(e, True) for e in top] + [_to_cluster(e, False) for e in evidence_extras]

    def _pool(region_x: "DxfRegion", region_y: "DxfRegion") -> "DxfRegion":
        # bbox is NOT computed here (unlike the real bbox `_on_merge_accepted`
        # builds for an actually-accepted merge) -- it is never read by
        # `_resolve`/`_score`/`_area_sane`/`_veto` below, only by bookkeeping
        # for a WINNING merge, so computing it for every rejected trial pair
        # would be pure waste on a sheet with many candidate pairs. Matches
        # the original `_merge_candidate_region_pairs`, which likewise only
        # ever called `_bbox_union` once per round, for the accepted pair.
        return DxfRegion(
            id=-1,  # placeholder -- only trial pools reach this; an accepted merge gets a real id in _on_merge_accepted
            bbox=region_x.bbox,
            polygon_indices=region_x.polygon_indices + region_y.polygon_indices,
            segment_indices=region_x.segment_indices + region_y.segment_indices,
            text_indices=region_x.text_indices + region_y.text_indices,
            entity_count=region_x.entity_count + region_y.entity_count,
        )

    def _resolve(pooled_region: "DxfRegion"):
        pooled_polygons = [polygons[k] for k in pooled_region.polygon_indices if k not in frame_indices]
        pooled_segments = [open_segments[k] for k in pooled_region.segment_indices]
        return _resolve_plot_building_road(
            pooled_polygons, pooled_segments, [], context_label="merged candidate (pre-acceptance trial)",
            allow_envelope_reconstruction=True,
        )

    def _score(resolved) -> float:
        plot_entry, building_entry, _road_entry = resolved
        return _score_region_resolution(plot_entry, building_entry)

    def _area_sane(resolved_a, resolved_b, resolved_merged) -> bool:
        # Two genuinely complementary fragments of ONE real boundary should
        # pool into an area comparable to -- not substantially larger than
        # -- the sum of their own (individually incomplete) areas: each
        # fragment already sees only PART of the same shape, so the true
        # whole is approximately that sum, not a multiple of it. A pooled
        # area far exceeding that sum is instead the signature of two
        # genuinely unrelated regions whose combined bounding shape
        # sprawls across empty space between them.
        plot_a, _b_a, _rd_a = resolved_a
        plot_b, _b_b, _rd_b = resolved_b
        plot_entry, _b_m, _rd_m = resolved_merged
        return (
            plot_entry is not None
            and plot_a is not None
            and plot_b is not None
            and plot_entry.polygon.area <= (plot_a.polygon.area + plot_b.polygon.area) * _MAX_MERGE_AREA_SUM_RATIO
        )

    def _veto(a: "HypothesisCluster", b: "HypothesisCluster", resolved_merged) -> Optional[str]:
        plot_a, _b_a, _rd_a = a.resolved
        plot_b, _b_b, _rd_b = b.resolved
        plot_entry, _b_m, _rd_m = resolved_merged
        return _evidence_veto_reason(evidence_by_region, a.payload, b.payload, plot_a, plot_b, plot_entry)

    def _on_merge_vetoed(a: "HypothesisCluster", b: "HypothesisCluster", merged_score: float, reason: str) -> None:
        warnings.append(
            f"Region-merge candidate {a.payload.id}+{b.payload.id} scored higher "
            f"({merged_score:.2f} vs {a.score:.2f}/{b.score:.2f}) but was vetoed: {reason}."
        )

    def _on_merge_accepted(a: "HypothesisCluster", b: "HypothesisCluster", new_cluster: "HypothesisCluster") -> None:
        pooled_region: "DxfRegion" = new_cluster.payload
        real_region = DxfRegion(
            id=new_cluster.id,
            # `pooled_region.bbox` is only `_pool`'s cheap trial placeholder
            # (region_x's own bbox, unioned with nothing) -- an ACCEPTED
            # merge needs the real union of both parents' bboxes, computed
            # here (once per accepted merge, matching the original's own
            # "only bbox-union the winner" cost profile) from `a`/`b`'s
            # ORIGINAL payload bboxes, not the discarded trial placeholder.
            bbox=_bbox_union([a.payload.bbox, b.payload.bbox]),
            polygon_indices=pooled_region.polygon_indices,
            segment_indices=pooled_region.segment_indices,
            text_indices=pooled_region.text_indices,
            entity_count=pooled_region.entity_count,
        )
        new_cluster.payload = real_region
        # Carry forward the union of both parents' own recovered evidence
        # (deduped by item identity) so a LATER growth round -- or the
        # final evidence-scoring pass -- can still check this merged
        # candidate's own confirmations, and so `_evidence_veto_reason`
        # has something to compare against if this cluster is grown again.
        parent_evidence = {
            id(item): item
            for item in (evidence_by_region.get(a.payload.id, []) + evidence_by_region.get(b.payload.id, []))
        }
        evidence_by_region[real_region.id] = list(parent_evidence.values())
        warnings.append(
            f"[merged region {real_region.id}] Regions {a.payload.id} (score {a.score:.2f}) and "
            f"{b.payload.id} (score {b.score:.2f}) pooled and re-resolved together score higher "
            f"({new_cluster.score:.2f}) than either alone -- kept as an additional candidate, consistent "
            "with one real boundary having been split across regions by density-based clustering."
        )

    def _on_budget_exhausted() -> None:
        warnings.append(
            f"Region-merge testing stopped early: its {time_budget_seconds:.0f}s time budget was "
            "used before growth could finish exploring this round. Merges already found are "
            "still used; growth simply stops here, same as if no further improvement were "
            "possible."
        )

    grown = grow_clusters(
        initial_clusters,
        pool_fn=_pool,
        resolve_fn=_resolve,
        score_fn=_score,
        next_id_fn=_next_id,
        area_sane_fn=_area_sane,
        veto_fn=_veto,
        time_budget_seconds=time_budget_seconds,
        on_merge_accepted=_on_merge_accepted,
        on_merge_vetoed=_on_merge_vetoed,
        on_budget_exhausted=_on_budget_exhausted,
    )

    return [
        (cluster.score, cluster.payload, cluster.resolved[0], cluster.resolved[1], cluster.resolved[2])
        for cluster in grown
    ]


def _resolve_via_regions(
    polygons: list["_RawPoly"],
    open_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    texts: list["_RawText"],
    warnings: list[str],
    extraction_start_time: Optional[float] = None,
    _dxf_hypothesis_sink: Optional[list["StructuralHypothesis"]] = None,
) -> Optional[tuple[Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"]]]:
    """Try resolving plot/building/road scoped to one spatially distinct
    region of the sheet, instead of globally across the whole DXF.

    Returns None (meaning "no region-based improvement available -- use
    the whole-sheet path instead") whenever:
      - clustering finds only one region (a normal single-drawing DXF,
        where the whole-sheet path IS the region-scoped path already), or
      - no region's resolution clears `_MIN_USABLE_REGION_SCORE` (e.g. a
        real site plan and its building happened to land in different
        regions after clustering -- the whole-sheet path, which considers
        every polygon together, is strictly more capable of finding a
        plot/building CONTAINMENT relationship than any single region
        that severed the two apart).

    This means region-based resolution can only ever IMPROVE on the
    whole-sheet result (by avoiding a cross-region sheet-border/frame
    polygon, and by giving fragmented-geometry reconstruction a small,
    tractable per-region subgraph instead of one sheet-wide component) --
    it never makes a normal single-drawing DXF behave differently, and it
    never overrides the whole-sheet path with something worse.

    `_dxf_hypothesis_sink` (Architecture V2, Phase DXF-1, additive): when
    provided, every region's own scored resolution -- not just the eventual
    `best` winner -- is ALSO projected into `StructuralHypothesis` records
    (via `dxf_evidence_projection.region_resolution_to_structural_
    hypotheses`) and appended to this list. Purely a side effect for a
    caller that wants to inspect/arbitrate over every candidate later; the
    function's own return value and winner-selection logic are completely
    unaffected by whether this is passed. Defaults to `None` (skip), so no
    existing caller's behavior changes.
    """
    from backend.cv_extraction.dxf_regions import _bbox_overlap_fraction, build_regions, detect_frame_polygon_indices

    # Falls back to "clock starts here" only when a caller (e.g. a direct
    # unit test) doesn't have the true extraction start time to pass in --
    # production always passes the real one from `_extract_impl`, since
    # file parsing alone can consume a large fraction of the whole budget
    # before this function is ever entered (see the time-budget check below).
    resolution_start_time = extraction_start_time if extraction_start_time is not None else time.time()

    polygon_points = [[(p.x, p.y) for p in poly.polygon.points] for poly in polygons]
    segment_points = [[start, end] for _layer, start, end in open_segments]
    text_points = [[t.position] for t in texts]

    regions = build_regions(polygon_points, segment_points, text_points)
    if len(regions) < 2:
        return None

    polygon_bboxes = [poly.polygon.bounding_box for poly in polygons]
    frame_indices = detect_frame_polygon_indices(polygon_bboxes, regions)
    # A real plot boundary also covers most of its drawing and encloses the
    # building's region, which is the geometric signature of a page frame. When
    # the architect put the polygon on a plot/boundary-named layer, that
    # declaration outweighs the geometric guess (seen on tagged sheets: three
    # agreeing _Plot/_NETPLOT/DCR_SPLOT polygons were dropped as "frames" and a
    # work-area polygon became the plot).
    plot_layer_polys = {i for i in frame_indices if _PLOT_LAYER_RE.search(polygons[i].layer or "")}
    if plot_layer_polys:
        frame_indices = type(frame_indices)(i for i in frame_indices if i not in plot_layer_polys)
    if frame_indices:
        warnings.append(
            f"Sheet-border/frame rejection: {len(frame_indices)} closed polygon(s) span multiple "
            "independently-clustered drawing regions on this sheet and were excluded from plot/building "
            "candidacy (relational evidence: a shape covering most of the sheet AND materially overlapping "
            "more than one distinct region is a page frame or title-block border, not one drawing's own boundary)."
        )

    # A REGION whose own bbox covers most of the entire sheet, AND
    # substantially OVERLAPS/ENCLOSES at least one other independently-
    # clustered region elsewhere on that same sheet, is itself behaving
    # like sheet-spanning content (a dashed border/frame traced all the
    # way around the page, a repeated background hatch pattern) rather
    # than one specific drawing -- the same "extreme page-level extent AND
    # spans multiple regions" relational evidence used for individual
    # polygons (`detect_frame_polygon_indices`) above, applied at the
    # region level so a whole cluster of small border-dash fragments
    # distributed around the sheet's perimeter doesn't get treated as
    # "the largest, most important drawing" purely because clustering
    # happened to keep it together.
    #
    # The overlap requirement is NOT optional: a large-bbox-share alone is
    # a false-positive trap whenever a real, single, legitimate drawing
    # coexists with ANY smaller, wholly-separate content elsewhere on the
    # sheet (a gate/north-arrow symbol, a dimension's own graphic offset
    # below the plot it measures) -- that unrelated content's mere
    # existence enlarges the union-of-all-regions bbox enough to push the
    # real drawing's own share of it over a raw coverage threshold, even
    # though the two regions never spatially overlap at all and a real
    # frame/border would. Confirmed directly: a plot + a same-layer
    # DIMENSION entity's own rendered graphic (extension lines, arrows,
    # text) clustering into a second region below the plot triggered this
    # exclusion on the plot region using the area-ratio check alone.
    overall_bbox = None
    for r in regions:
        overall_bbox = r.bbox if overall_bbox is None else BoundingBox(
            min_x=min(overall_bbox.min_x, r.bbox.min_x), min_y=min(overall_bbox.min_y, r.bbox.min_y),
            max_x=max(overall_bbox.max_x, r.bbox.max_x), max_y=max(overall_bbox.max_y, r.bbox.max_y),
        )
    overall_area = max(
        (overall_bbox.max_x - overall_bbox.min_x) * (overall_bbox.max_y - overall_bbox.min_y), 1e-9
    ) if overall_bbox is not None else 1e-9

    # Pass 1: cheap structural resolution for every region (no OCR at all).
    # OCR-based evidence recovery (render + tesseract per region) is real
    # wall-clock cost -- on a real, large, many-region sheet (30,111
    # entities, 14 regions) running it for every region pushed a single
    # DXF past this project's own extraction timeout. Structural
    # resolution alone is what determines which regions could plausibly
    # win at all; evidence recovery in pass 2 below only needs to run on
    # the few regions actually in contention.
    structural: list[tuple[float, "DxfRegion", Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"]]] = []
    for region in regions:
        region_area = (region.bbox.max_x - region.bbox.min_x) * (region.bbox.max_y - region.bbox.min_y)
        spans_another_region = any(
            other.id != region.id and _bbox_overlap_fraction(other.bbox, region.bbox) > 0.5
            for other in regions
        )
        if region_area / overall_area > _REGION_MAX_SHEET_COVERAGE and spans_another_region:
            warnings.append(
                f"[region {region.id}] Excluded from plot/building candidacy: this region's own bounding "
                f"box covers {region_area / overall_area:.0%} of the whole sheet's extent AND substantially "
                "overlaps another independently-clustered region elsewhere on it -- consistent with "
                "sheet-spanning content (a dashed border traced around the page, a repeated background "
                "pattern) rather than one specific drawing."
            )
            continue
        region_polygons = [polygons[i] for i in region.polygon_indices if i not in frame_indices]
        region_segments = [open_segments[i] for i in region.segment_indices]
        if not region_polygons and not region_segments:
            continue
        # See DXF_FAILURE_TAXONOMY.md item 1 and dxf_glyph_swarm.py's own
        # module docstring: a region built entirely from vectorized text
        # (a notes paragraph, a data table) can out-score a genuine
        # boundary region on fragment count/envelope size alone. Checked
        # BEFORE spending time on plot/building resolution for a region
        # about to be excluded anyway. Classification thresholds are
        # explicitly unvalidated placeholders (see dxf_glyph_swarm.py) --
        # this only fires on the extreme, already-confirmed shape (many
        # hundreds of polygons, overwhelmingly organized into populated
        # text-line row-bands of consistent height), not a borderline call.
        glyph_swarm_signal = compute_glyph_swarm_signal([polygons[i].polygon.bounding_box for i in region.polygon_indices if i not in frame_indices])
        is_glyph_swarm, glyph_swarm_reason = classify_glyph_swarm(glyph_swarm_signal)
        if is_glyph_swarm:
            warnings.append(
                f"[region {region.id}] Excluded from plot/building candidacy: {glyph_swarm_reason}"
            )
            continue
        plot_entry, building_entry, road_entry = _resolve_plot_building_road(
            region_polygons, region_segments, warnings, context_label=f"region {region.id}",
            allow_envelope_reconstruction=True,
        )
        score = _score_region_resolution(plot_entry, building_entry)
        structural.append((score, region, plot_entry, building_entry, road_entry))

    if not structural:
        return None

    # DXF_FAILURE_TAXONOMY.md item 9: how many ORIGINAL regions (before any
    # merge growth, before any caption/evidence bonus) independently clear
    # the usability bar on their own structural merit. Counted here, before
    # Pass 1.6 can inflate any one region's score with a caption bonus --
    # this must reflect genuine structural plausibility only, otherwise a
    # captioned region would trivially count itself into ambiguity. A
    # sheet with more than one such region is a real multi-drawing sheet
    # (a wall section, a floor plan, AND a site plan can each resolve a
    # plausible-looking plot/building pair on their own) -- the same
    # complexity that makes item 0's caption check necessary in the first
    # place. See the final-selection code below for what this is used for.
    plausible_candidate_count = sum(1 for score, *_ in structural if score >= _MIN_USABLE_REGION_SCORE)

    # Pass 1.5: vectorized-text/OCR evidence recovery, run EXACTLY ONCE for
    # the whole sheet and then associated against the UNION of every
    # region's own candidate edges at once (see
    # `_recover_sheet_wide_evidence`'s docstring for why this replaced an
    # earlier "OCR bounded to the top-N structurally-scoring regions"
    # design: bounding to a structural shortlist meant a region that
    # scored badly on geometry ALONE could never have its evidence looked
    # at even when the sheet's own printed numbers would have confirmed
    # it was actually correct -- evidence could only demote a candidate
    # already in contention, never promote one structural scoring had
    # already wrongly excluded. One sheet-wide OCR pass is both cheaper
    # (measured: ~10-30s once, vs. 90-170s for repeated per-region passes)
    # and gives every region the same opportunity to be confirmed.
    #
    # Deliberately BEFORE region-merge growth (not after): growth needs
    # this evidence to veto a structurally-improving merge that would
    # move a dimension away from a value the sheet's own printed
    # dimensions already confirmed (see `_evidence_veto_reason`) -- it
    # can't do that with evidence that doesn't exist yet. Running it first
    # also means merge-growth's own time-budget share (below) is computed
    # from what's ACTUALLY left afterward, rather than the two stages
    # silently competing for the same window.
    #
    # This whole function runs inside the extractor's own overall
    # `bounded_execution.run_with_timeout` wall-clock budget -- on a very
    # large, many-region sheet, structural resolution ALONE (just above)
    # can already consume most of it (measured directly: ~145s of a 180s
    # budget for a ~27,000-entity, 14-region file). Attempting OCR evidence
    # recovery anyway would not gracefully degrade -- it would make the
    # WHOLE extraction exceed its timeout and return nothing at all, which
    # is strictly worse than skipping evidence and keeping the structural-
    # only result this function would otherwise have returned. Skip
    # evidence recovery once less than this fraction of the configured
    # budget remains; every region still gets its (unmodified) structural
    # score, exactly as if this whole evidence layer did not exist.
    from backend.config import get_settings

    elapsed = time.time() - resolution_start_time
    remaining_fraction = 1.0 - (elapsed / max(get_settings().extraction_timeout_seconds, 1e-6))
    evidence_by_region: dict[int, list] = {}
    captions_by_region: dict[int, list[str]] = {}
    # DXF_FAILURE_TAXONOMY.md item 8: a region whose own isolated
    # resolution found no road polygon may still be able to borrow one
    # from elsewhere on the sheet (see `find_nearby_road_candidate`'s own
    # docstring for the root cause this targets and what it refuses to
    # do). Populated below, consulted again at Pass 3 so the FINAL winner
    # ships the borrowed road (road.width/setbacks) instead of MISSING
    # when its own resolution had none but a plausible, unambiguous
    # neighbor did.
    borrowed_road_by_region: dict[int, "_RawPoly"] = {}
    if remaining_fraction < _MIN_REMAINING_TIME_FRACTION_FOR_EVIDENCE:
        warnings.append(
            f"Skipped vectorized-text/dimension-evidence recovery: {elapsed:.1f}s of this document's "
            "extraction time budget was already used (DXF file parsing plus structural region "
            "resolution), leaving too little remaining to also run OCR safely. Falling back to "
            "structural-only region scoring for this document."
        )
    else:
        region_candidate_edges: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
        region_bboxes: dict[int, BoundingBox] = {}
        all_road_entries_by_region = [
            (other_region.id, other_road)
            for _s, other_region, _p, _b, other_road in structural
            if other_road is not None
        ]
        for _score, region, plot_entry, building_entry, own_road_entry in structural:
            edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
            if plot_entry is not None:
                edges.extend(_polygon_edge_segments(plot_entry.polygon))
            if building_entry is not None:
                edges.extend(_polygon_edge_segments(building_entry.polygon))
            road_entry_for_edges = own_road_entry
            if road_entry_for_edges is None and plot_entry is not None:
                other_candidates = [(rid, r) for rid, r in all_road_entries_by_region if rid != region.id]
                borrowed = find_nearby_road_candidate(plot_entry.polygon.bounding_box, other_candidates)
                if borrowed is not None:
                    road_entry_for_edges = borrowed.road_entry
                    borrowed_road_by_region[region.id] = borrowed.road_entry
                    warnings.append(
                        f"[region {region.id}] Borrowed a road polygon from region "
                        f"{borrowed.source_region_id} (this region's own resolution found none; the "
                        f"nearest plausible candidate sits {borrowed.center_distance:.2f} drawing-units "
                        "from this plot's own center) -- used to let the sheet's own printed road-width "
                        "text associate with the right edge, and (if this region wins) for road.width/"
                        "setback computation."
                    )
            if road_entry_for_edges is not None:
                edges.extend(_polygon_edge_segments(road_entry_for_edges.polygon))
            if edges:
                region_candidate_edges[region.id] = edges
                region_bboxes[region.id] = region.bbox

        remaining_budget_seconds = remaining_fraction * get_settings().extraction_timeout_seconds
        evidence_by_region, captions_by_region = _recover_sheet_wide_evidence(
            polygon_points, segment_points, region_candidate_edges, region_bboxes, warnings,
            remaining_budget_seconds=remaining_budget_seconds,
        )

    # Pass 1.6: apply each region's own drawing-type-caption bonus (see
    # `_SITE_PLAN_CAPTION_KEYWORD_GROUPS`) directly to its stored structural
    # score, BEFORE merge growth and BEFORE Pass 3's final selection --
    # unlike a numeric evidence bonus (Pass 3 only), a caption match must
    # also affect merge candidacy (`_merge_candidate_region_pairs` reads
    # `structural`'s own score to pick its top-N pool) and outright winner
    # selection, since "this region is explicitly captioned as the site
    # plan" should outrank an uncaptioned drawing regardless of which one
    # merge growth happens to try first.
    # Regions whose own PRE-caption score never cleared `_MIN_USABLE_REGION_
    # SCORE` -- i.e. would never have been considered usable at all without
    # the caption override. A caption is strong evidence this is the right
    # DRAWING; it is not evidence that this region's own reconstructed
    # geometry is accurate, so this is tracked separately and used below to
    # cap the SHIPPED measurement confidence, not to change which region wins.
    regions_where_caption_overrode_implausible_score: set[int] = set()
    if captions_by_region:
        rescored: list[tuple[float, "DxfRegion", Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"]]] = []
        for score, region, plot_entry, building_entry, road_entry in structural:
            captions = captions_by_region.get(region.id) or []
            if captions:
                if score < _MIN_USABLE_REGION_SCORE:
                    regions_where_caption_overrode_implausible_score.add(region.id)
                warnings.append(
                    f"[region {region.id}] Drawing-type caption(s) recognized: "
                    f"{', '.join(repr(c) for c in captions)} -- applying a "
                    f"+{_SITE_PLAN_CAPTION_SCORE_BONUS:.0f} structural-score bonus (this region is very "
                    "likely the actual site plan, not a section/floor-plan/elevation/schedule that "
                    "happens to reconstruct into a larger or more complex-looking envelope)."
                    + (
                        f" Its own pre-caption score ({score:.2f}) was below the usability bar "
                        f"({_MIN_USABLE_REGION_SCORE:.1f}) that would normally be required to trust this "
                        "region's geometry at all (most often because its reconstructed plot area is "
                        "implausible at its current scale) -- if this region ends up winning, its shipped "
                        "measurement confidence will be capped accordingly until that is resolved."
                        if score < _MIN_USABLE_REGION_SCORE else ""
                    )
                )
                score += _SITE_PLAN_CAPTION_SCORE_BONUS
            rescored.append((score, region, plot_entry, building_entry, road_entry))
        structural = rescored

    # Pass 2: evidence-guided pairwise region merging (see
    # `_merge_candidate_region_pairs`'s own docstring for the root cause
    # this targets -- density-based clustering fragmenting one real
    # boundary into several disjoint regions). Structural resolution alone
    # (Pass 1) can already consume most of the overall budget on a real,
    # large, many-region sheet (measured directly: 90-150s of a 180s
    # budget) -- a binary "only try this if at least half the ENTIRE
    # budget is still free" gate would make this feature's benefit depend
    # on incidental machine timing (confirmed directly: the exact same
    # file, run moments apart, crossed that line one way and then the
    # other). Instead, always attempt it with whatever time is actually
    # left, spending only a MINORITY share of that -- `_merge_candidate_
    # region_pairs` itself stops trying further pairs once its own share
    # is used, so this degrades smoothly (fewer pairs tried when less time
    # remains) rather than flipping between "all" and "none".
    #
    # The share is deliberately small and asymmetric in evidence recovery's
    # favor, not an even split: evidence recovery (just above) can discover
    # the document's own actual printed dimensions, which region-merging
    # (a purely structural/geometric heuristic with no access to the
    # sheet's printed truth) can never do on its own -- it can only VETO
    # against that evidence once recovered. Confirmed directly on a real
    # bundled DXF regression fixture (a 45MB file whose own ezdxf.readfile
    # alone costs ~40s): an even 50/50 split let region-merge testing
    # consume its full share and push elapsed past the point where
    # evidence recovery's own remaining-budget gate could still pass,
    # starving the mechanism most likely to find the correct answer in
    # favor of one that, on that same file, could not. This is a generic
    # priority ordering (evidence over structural heuristics), not a
    # threshold tuned to any specific file's size.
    elapsed_before_merging = time.time() - resolution_start_time
    remaining_seconds_before_merging = max(
        0.0, get_settings().extraction_timeout_seconds - elapsed_before_merging
    )
    merge_time_budget = remaining_seconds_before_merging * _REGION_MERGE_BUDGET_SHARE
    if merge_time_budget > 0:
        merged_candidates = _merge_candidate_region_pairs(
            structural, polygons, open_segments, frame_indices, warnings, time_budget_seconds=merge_time_budget,
            evidence_by_region=evidence_by_region,
        )
        structural.extend(merged_candidates)
    else:
        warnings.append(
            f"Skipped evidence-guided region-merge testing: {elapsed_before_merging:.1f}s of this "
            "document's extraction time budget was already used, leaving none remaining to also try "
            "it. Falling back to each region's own individual resolution."
        )

    # Pass 3: final scoring -- every structural entry (organic regions AND
    # any merges Pass 2 found) gets its evidence bonus from whatever
    # `evidence_by_region` already holds for its own id (populated in Pass
    # 1.5 for organic regions, and extended by `_merge_candidate_region_
    # pairs` itself for each merged id) -- no second OCR call.
    best: Optional[tuple[float, int, Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"], Optional[float]]] = None
    # Architecture V2, Phase DXF-3: every region's own resolution is ALWAYS
    # projected into StructuralHypothesis records (Phase DXF-1) regardless of
    # `_dxf_hypothesis_sink`, so the new multi-hypothesis decision engine
    # (Phase DXF-2) can be run in shadow mode unconditionally -- mirrors
    # `evidence_decision_bridge.py`'s own "always compute, flag picks which
    # ships" pattern for the PDF path. `region_entries`/`region_scores` let
    # the flagged path below ship a decided region's own tuple using the
    # exact same downstream code (scale correction, caption/identity-flag
    # application, borrowed-road lookup) the existing `best`-based path
    # already runs -- only WHICH region is chosen differs.
    region_entries: dict[int, tuple[Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"], Optional[float]]] = {}
    region_scores: dict[int, float] = {}
    hypotheses_by_identity: dict[str, list["StructuralHypothesis"]] = {}
    hyp_id_to_region_id: dict[str, int] = {}
    for score, region, plot_entry, building_entry, road_entry in structural:
        recovered = evidence_by_region.get(region.id, [])
        evidence_bonus, scale_correction = _score_recovered_evidence(
            recovered, plot_entry, building_entry, warnings, f"region {region.id}",
        )
        score += evidence_bonus
        if best is None or score > best[0]:
            best = (score, region.id, plot_entry, building_entry, road_entry, scale_correction)
        region_entries[region.id] = (plot_entry, building_entry, road_entry, scale_correction)
        region_scores[region.id] = score

        from backend.cv_extraction.dxf_evidence_projection import (
            caption_match_to_evidence_candidate,
            region_resolution_to_structural_hypotheses,
        )

        region_captions = captions_by_region.get(region.id) or []
        id_prefix = f"dxf-r{region.id}"
        caption_evidence = caption_match_to_evidence_candidate(region.id, region_captions, id_prefix=id_prefix)
        constraint_results, structural_score = _score_region_resolution_detailed(plot_entry, building_entry)
        region_hypotheses = region_resolution_to_structural_hypotheses(
            region_id=region.id,
            plot_entry=plot_entry, building_entry=building_entry, road_entry=road_entry,
            constraint_results=constraint_results,
            structural_score=structural_score,
            caption_matched=bool(region_captions),
            caption_bonus_applied=_SITE_PLAN_CAPTION_SCORE_BONUS if region_captions else 0.0,
            plausible_candidate_count=plausible_candidate_count,
            min_plausible_candidates_for_concern=_MIN_PLAUSIBLE_CANDIDATES_FOR_UNCONFIRMED_IDENTITY_CONCERN,
            evidence_bonus=evidence_bonus,
            caption_evidence_id=caption_evidence.id if caption_evidence is not None else None,
            id_prefix=id_prefix,
        )
        if _dxf_hypothesis_sink is not None:
            _dxf_hypothesis_sink.extend(region_hypotheses)
        for h in region_hypotheses:
            hypotheses_by_identity.setdefault(h.identity.value, []).append(h)
            hyp_id_to_region_id[h.id] = region.id

    if best is None or best[0] < _MIN_USABLE_REGION_SCORE:
        return None

    from backend.schemas.enums import DecisionStatus, HypothesisIdentity
    from backend.spatial_reasoning.structural_hypothesis_decision import (
        MEASURED_MARGIN_FOR_HIGH,
        MEASURED_MARGIN_FOR_MEDIUM,
        decide_structural_hypothesis,
    )

    structural_decisions = {
        identity: decide_structural_hypothesis(
            identity, hypotheses_by_identity.get(identity.value, []),
            min_usable_score=_MIN_USABLE_REGION_SCORE,
            margin_for_high=MEASURED_MARGIN_FOR_HIGH, margin_for_medium=MEASURED_MARGIN_FOR_MEDIUM,
        )
        for identity in (HypothesisIdentity.PLOT_BOUNDARY, HypothesisIdentity.BUILDING, HypothesisIdentity.ROAD)
    }
    _log_dxf_structural_decision_comparison(structural_decisions, best, hyp_id_to_region_id, warnings)

    chosen = best
    if get_settings().use_dxf_evidence_decision_engine:
        plot_decision, _plot_level = structural_decisions[HypothesisIdentity.PLOT_BOUNDARY]
        if plot_decision.status != DecisionStatus.ACCEPT:
            # ABSTAIN or CONFLICT: the new engine found >=2 regions
            # independently plausible enough to clear the usability floor
            # with no way to confidently pick between them (a genuine tie,
            # or every candidate individually too weak) -- confirmed live
            # on real fixtures (PLAN2/PLAN6.dxf both land here). Shipping
            # the OLD max-score winner anyway here would silently discard
            # exactly the uncertainty this engine exists to surface ("wrong
            # is worse than missing").
            #
            # Deliberately returns (None, None, None) here, NOT bare `None`
            # -- an earlier version of this fix returned `None` to re-enter
            # `_build_independent_cv`'s own whole-sheet fallback
            # (`_resolve_plot_building_road` over ALL polygons, ignoring
            # the region split), reasoning that its own containment logic
            # would recover a reasonable answer independent of which region
            # "won." Verified false on a real regression fixture
            # (`test_flag_true_agrees_with_flag_off_on_a_real_multi_region_
            # fixture`, `multi_drawing_sheet_with_frame_dxf`): the
            # whole-sheet path shipped `plot.width=90.0` -- the SHEET
            # BORDER/FRAME polygon itself, a confidently WRONG answer worse
            # than either the old region-based winner or nothing at all.
            # Frame rejection is evidently not as robust on the whole-sheet
            # path as on the region-scoped one; falling back to it silently
            # traded a merely-uncertain case for a newly-confident wrong
            # one, exactly backwards for "wrong is worse than missing."
            # Shipping (None, None, None) instead means this document's
            # plot/building/road stay honestly MISSING when the new engine
            # cannot confidently identify the region, with no untested
            # fallback path's own risk inherited along the way.
            warnings.append(
                f"[use_dxf_evidence_decision_engine] PLOT_BOUNDARY: {plot_decision.status.value} -- "
                f"{plot_decision.confidence_derivation} Shipping no region-based plot/building/road for "
                "this document rather than the max-score winner (unconfirmed) or a whole-sheet fallback "
                "(its own frame-rejection is not validated for this case)."
            )
            return None, None, None
        decided_region_id = (
            hyp_id_to_region_id.get(plot_decision.accepted_candidate_id)
            if plot_decision.accepted_candidate_id is not None else None
        )
        # Deliberately conservative: the decided region's OWN plot/building/
        # road entries ship together, never mixed with a different region's
        # -- BUILDING/ROAD are decided above for shadow-logging/auditability
        # only. Mixing entries across regions for different roles is a
        # larger behavior change explicitly out of scope for this phase.
        if decided_region_id is not None and decided_region_id in region_entries:
            decided_plot, decided_building, decided_road, decided_scale = region_entries[decided_region_id]
            chosen = (
                region_scores[decided_region_id], decided_region_id,
                decided_plot, decided_building, decided_road, decided_scale,
            )

    score, region_id, plot_entry, building_entry, road_entry, scale_correction = chosen
    if scale_correction is not None:
        if plot_entry is not None:
            plot_entry = plot_entry.scaled(scale_correction)
        if building_entry is not None:
            building_entry = building_entry.scaled(scale_correction)
    if road_entry is None and region_id in borrowed_road_by_region:
        # Only organic (non-merged) regions ever get a chance to borrow --
        # `borrowed_road_by_region` is populated from Pass 1's per-region
        # loop, before Pass 2's merge growth can add a synthetic region id
        # to `structural`. A merged winner's own road_entry (if any) comes
        # from re-resolving its pooled geometry directly in `_merge_
        # candidate_region_pairs`, which does not consult this borrowing
        # step -- a known, honest gap (see DXF_FAILURE_TAXONOMY.md item 8),
        # not attempted here.
        road_entry = borrowed_road_by_region[region_id]
    if region_id in regions_where_caption_overrode_implausible_score:
        import dataclasses as _dataclasses

        if plot_entry is not None:
            plot_entry = _dataclasses.replace(plot_entry, caption_overrode_implausible_score=True)
        if building_entry is not None:
            building_entry = _dataclasses.replace(building_entry, caption_overrode_implausible_score=True)
    elif (
        plausible_candidate_count >= _MIN_PLAUSIBLE_CANDIDATES_FOR_UNCONFIRMED_IDENTITY_CONCERN
        # `captions_by_region` is pre-populated with an empty list per
        # region (for `.setdefault` convenience elsewhere) -- the dict
        # itself is never empty, so "no caption found anywhere" must check
        # whether any of ITS VALUES are non-empty, not whether the dict
        # has keys at all. Caught directly: this check originally read
        # `not captions_by_region`, which is always False once any region
        # had candidate edges, silently never firing on PLAN6's own real,
        # motivating case.
        and not any(captions_by_region.values())
    ):
        # DXF_FAILURE_TAXONOMY.md item 9: multiple regions independently
        # looked like plausible drawings, and not one of them was
        # confirmed by a caption anywhere on the sheet -- this pipeline
        # has no positive signal that the winner (however it won) is
        # actually the site plan rather than a section, floor plan, or
        # fragment merge that happened to score well. `elif`, not a
        # separate `if`: mutually exclusive with the caption-override
        # branch above by construction (that branch only ever applies
        # when a caption WAS found somewhere, this one only when none was).
        import dataclasses as _dataclasses

        warnings.append(
            f"[region {region_id}] No drawing-type caption was recognized anywhere on this sheet, yet "
            f"{plausible_candidate_count} regions independently scored as plausible drawings on their "
            "own structural merit -- this pipeline cannot positively confirm the winning region is "
            "actually the site plan rather than an unrelated drawing that happened to score well "
            "(e.g. a section view, a floor plan, or a fragment merge). Shipped measurement confidence "
            "is capped accordingly."
        )
        if plot_entry is not None:
            plot_entry = _dataclasses.replace(plot_entry, unconfirmed_drawing_identity=True)
        if building_entry is not None:
            building_entry = _dataclasses.replace(building_entry, unconfirmed_drawing_identity=True)
    warnings.append(
        f"Resolved plot/building from region {region_id} of {len(regions)} detected sheet regions "
        f"(region-resolution score={score:.2f}); this is scoped to one spatially distinct drawing on the "
        "sheet, rather than the whole-sheet geometry (see sheet-border/frame rejection above)."
    )
    return plot_entry, building_entry, road_entry


def _log_dxf_structural_decision_comparison(
    structural_decisions: dict["HypothesisIdentity", tuple["Decision", "ConfidenceLevel"]],
    best: Optional[tuple[float, int, Optional["_RawPoly"], Optional["_RawPoly"], Optional["_RawPoly"], Optional[float]]],
    hyp_id_to_region_id: dict[str, int],
    warnings: list[str],
) -> None:
    """Architecture V2, Phase DXF-3: log whenever the new multi-hypothesis
    decision engine would ship something different from the existing
    max-score `best` selection, or would ABSTAIN/CONFLICT where `best`
    still ships a value regardless -- mirrors `evidence_decision_bridge.
    log_comparison`'s own style (log only, never raise, never change what
    ships unless the feature flag is on). This is the raw material a future
    Phase-DXF-4-style comparison report reads from, the same way Phase 8's
    shadow logging fed Phase 9's PDF comparison report.
    """
    from backend.schemas.enums import DecisionStatus
    from backend.tools.logging_config import get_logger

    logger = get_logger(__name__)
    best_region_id = best[1] if best is not None else None
    for identity, (decision, level) in structural_decisions.items():
        if decision.status in (DecisionStatus.CONFLICT, DecisionStatus.ABSTAIN):
            logger.info(
                "dxf_structural_decision %s: %s (max-score region %s still ships by default unless "
                "use_dxf_evidence_decision_engine is set) -- %s",
                identity.value, decision.status.value, best_region_id, decision.confidence_derivation,
            )
            continue
        decided_region_id = (
            hyp_id_to_region_id.get(decision.accepted_candidate_id)
            if decision.accepted_candidate_id is not None else None
        )
        if decided_region_id is not None and decided_region_id != best_region_id:
            logger.info(
                "dxf_structural_decision %s: new engine would accept region %s (%s) instead of the "
                "max-score winner region %s -- %s",
                identity.value, decided_region_id, level.value, best_region_id, decision.confidence_derivation,
            )


def _entry_source_confidence(
    entry: "_RawPoly", base_confidence: float, unit_confidence: float = 1.0
) -> tuple[str, float]:
    """Vision-identified regions and reconstructed-from-fragments outlines
    each carry a distinct source and a capped, lower confidence than exact
    vector geometry (a single closed polygon straight from the DXF) -- see
    the DXF Vision-render fallback notes in PHASE4_SPATIAL_AWARENESS_NOTES.md
    and dxf_reconstruction.py's module docstring. Reconstruction is capped
    less aggressively than Vision: it is still exact DXF coordinates, just
    stitched together by a heuristic (snap/merge/cycle-detect) rather than
    read as one already-closed polygon, so its uncertainty is lower than a
    Vision bounding-box guess but still real."""
    # A curved boundary (ARC/CIRCLE/bulge) was flattened into straight
    # segments to compute area/bbox -- a controlled, disclosed
    # approximation, not exact vector geometry in the same sense as a
    # polygon built entirely from straight LINE/LWPOLYLINE segments.
    curve_cap = 0.90 if entry.approximated_curve else 1.0
    # This region only cleared the usability bar because of a drawing-type
    # caption match, not on its own structural merit -- see
    # `_CAPTION_OVERRIDE_CONFIDENCE_CAP`'s own docstring. OR: no caption
    # was found anywhere, on a sheet with multiple plausible-looking
    # regions -- see `_RawPoly.unconfirmed_drawing_identity`'s own
    # docstring (item 9). Either way, applied ahead of every source-
    # specific branch below: whatever the source would otherwise imply,
    # this pipeline cannot vouch for this region's identity or geometry.
    implausibility_cap = (
        _CAPTION_OVERRIDE_CONFIDENCE_CAP
        if entry.caption_overrode_implausible_score or entry.unconfirmed_drawing_identity
        else 1.0
    )
    if entry.layer == _VISION_FALLBACK_LAYER:
        return "VISION_DXF_RENDER", min(base_confidence, 0.65, unit_confidence, curve_cap, implausibility_cap)
    if entry.layer == _RECONSTRUCTED_LAYER:
        return "VECTOR_GEOMETRY_RECONSTRUCTED", min(base_confidence, 0.82, unit_confidence, curve_cap, implausibility_cap)
    if entry.layer == _WALL_UNION_RECONSTRUCTED_LAYER:
        # A real polygon shape traced through wall-thickness buffering and
        # union (not a bounding rectangle), but it still depends on an
        # estimated wall thickness and inferred opening bridges rather
        # than reading an already-closed outline directly -- capped
        # between exact closed-loop reconstruction and the coarser
        # bounding-envelope fallback.
        return "VECTOR_GEOMETRY_RECONSTRUCTED", min(base_confidence, 0.75, unit_confidence, curve_cap, implausibility_cap)
    if entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER:
        # A boundary reconstructed from fragments, not read as one already-
        # closed polygon -- the least certain of the non-Vision sources,
        # capped accordingly. `fragment_boundary` (fitted straight sides
        # through well-supported collinear runs) earns a higher cap than
        # the cruder `bounding_rectangle` fallback (one rectangle wrapped
        # around every point, with no per-side evidence at all).
        envelope_cap = 0.80 if entry.envelope_method == "fragment_boundary" else 0.70
        return "VECTOR_GEOMETRY_RECONSTRUCTED", min(base_confidence, envelope_cap, unit_confidence, curve_cap, implausibility_cap)
    return "VECTOR_GEOMETRY", min(base_confidence, unit_confidence, curve_cap, implausibility_cap)


def _raw_poly_from_bbox(bbox: BoundingBox, layer: str) -> "_RawPoly":
    poly = Polygon(points=[
        Point(x=bbox.min_x, y=bbox.min_y), Point(x=bbox.max_x, y=bbox.min_y),
        Point(x=bbox.max_x, y=bbox.max_y), Point(x=bbox.min_x, y=bbox.max_y),
    ])
    return _RawPoly(layer=layer, polygon=poly)


def _normalize_vision_bbox(
    bbox: Optional[list[float]], width_px: int, height_px: int
) -> Optional[tuple[float, float, float, float]]:
    """Vision-family models emit bboxes on a normalized 0-1000 grid regardless
    of true render size (see `vision_extraction.spatial.vision_bbox_to_page_points`'s
    docstring for the same auto-detection heuristic on the PDF side) -- this
    is a small, self-contained equivalent for a raw-pixel DXF render rather
    than reusing that function's PDF-point/DPI-based unit assumptions, which
    don't apply here."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    looks_normalized = max(x1, y1, x2, y2) <= 1024.0 and (width_px > 1024 or height_px > 1024)
    if looks_normalized:
        x1, x2 = x1 / 1000.0 * width_px, x2 / 1000.0 * width_px
        y1, y2 = y1 / 1000.0 * height_px, y2 / 1000.0 * height_px
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _vision_fallback_regions(
    document_id: str,
    polygons: list["_RawPoly"],
    open_chains: list[list[tuple[float, float]]],
    warnings: list[str],
) -> dict[str, BoundingBox]:
    """
    Render this DXF's own line-work and ask Vision to visually identify
    which region is the building/road/site boundary, for the case the
    deterministic closed-polygon-only search finds nothing plausible --
    e.g. a DXF produced by vectorizing a scanned sheet, where walls were
    never captured as a single closed polygon at all (confirmed on a real
    bundled sample: 6,346 tiny closed fragments, 12,389 tiny open
    segments, zero text/layer information -- see
    PHASE4_SPATIAL_AWARENESS_NOTES.md).

    Vision supplies ONLY a region bounding box (semantic identification
    from pixels, per `DXF_VISION_FALLBACK_PROMPT`, which explicitly tells
    it not to report any measurement). That bbox is mapped back to exact
    DXF world coordinates via the render's own self-computed, directly-
    verified affine transform (`dxf_render.RasterTransform`) -- never
    through any number Vision itself reports. This keeps the same "Vision
    decides semantics, geometry decides numbers" boundary the PDF pipeline
    already enforces, just with DXF's own exact vector coordinates
    standing in for OpenCV as the measurement source.

    Returns {} (never raises) if Vision is disabled/unavailable or finds
    nothing plausible -- this is strictly an additive fallback; the
    deterministic path above is unaffected when this returns nothing.
    """
    from backend.config import get_settings

    settings = get_settings()
    if not settings.vision_enabled:
        return {}

    try:
        from backend.cv_extraction import dxf_render
        from backend.vision_extraction import get_vision_extractor
        from backend.vision_extraction.prompts import DXF_VISION_FALLBACK_PROMPT

        closed_chains = [[(p.x, p.y) for p in rp.polygon.points] for rp in polygons]
        render_dir = settings.resolve(settings.upload_dir) / "vision_rendered" / f"{document_id}_dxf"
        image_path = render_dir / "dxf_render.png"
        transform = dxf_render.render_polylines_to_image(
            closed_chains, open_chains, image_path, target_max_px=1600
        )
        if transform is None:
            return {}

        result = get_vision_extractor().analyze_image(
            image_path,
            page_number=1,
            native_spans=[],
            page_width_pts=float(transform.width_px),
            page_height_pts=float(transform.height_px),
            ground_against_native_text=False,
            prompt_override=DXF_VISION_FALLBACK_PROMPT,
            max_new_tokens=settings.vision_focus_max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001 -- a Vision failure must not break the deterministic DXF path
        warnings.append(f"DXF Vision-render fallback unavailable: {exc}")
        return {}

    found: dict[str, BoundingBox] = {}
    for region in result.regions:
        rtype = (region.type or "").upper()
        if rtype not in ("SITE_PLAN", "BUILDING", "ROAD") or rtype in found:
            continue
        px_bbox = _normalize_vision_bbox(region.bbox, transform.width_px, transform.height_px)
        if px_bbox is None:
            continue
        wx0, wy0, wx1, wy1 = transform.pixel_bbox_to_world_bbox(px_bbox)
        if wx1 - wx0 <= 0 or wy1 - wy0 <= 0:
            continue
        found[rtype] = BoundingBox(min_x=wx0, min_y=wy0, max_x=wx1, max_y=wy1)

    if found:
        warnings.append(
            f"Vision-render fallback identified: {', '.join(sorted(found))} (rendered from "
            f"{len(closed_chains)} closed + {len(open_chains)} open DXF line chains; no native "
            "text/dimension data was involved -- the reported measurement still comes from exact "
            "DXF world coordinates within the identified region, not from Vision's own reading)."
        )
    return found


def _iter_entities_flat(msp, warnings: list[str], max_depth: int = 3):
    """Yield modelspace entities, transparently exploding INSERT block
    references (up to `max_depth` nesting levels) so geometry drawn inside a
    block (a common way to package a repeated site-plan title block, or a
    unit-plan block placed on a layout) is not silently skipped.

    `INSERT.virtual_entities()` deliberately does NOT include the block's
    ATTDEF entities (per ezdxf's own docs: "Do not explode ATTDEF entities.
    Already available in Insert.attribs") -- those are template placeholders
    in the block *definition*, not this particular insertion's actual
    values. The real per-instance values (e.g. a title block's "DRAWING NO"
    filled in for this one insertion) live on `INSERT.attribs` as real
    ATTRIB entities and were previously never read at all, silently
    dropping every block-attribute value in the drawing.
    """

    def _walk(entities, depth: int):
        for e in entities:
            yield e
            if depth < max_depth:
                try:
                    dxftype = e.dxftype()
                except Exception:
                    continue
                if dxftype == "INSERT":
                    try:
                        virtual = list(e.virtual_entities())
                    except Exception as exc:  # pragma: no cover - malformed block
                        warnings.append(f"Could not explode an INSERT block reference: {exc}")
                        virtual = []
                    if virtual:
                        yield from _walk(virtual, depth + 1)
                    try:
                        attribs = list(e.attribs)
                    except Exception as exc:  # pragma: no cover - malformed block
                        warnings.append(f"Could not read INSERT attribute values: {exc}")
                        attribs = []
                    if attribs:
                        yield from _walk(attribs, depth + 1)

    yield from _walk(msp, 0)


class DXFHybridExtractor(GeometryExtractor):
    """
    Deterministic DXF extractor: reads plot/building/road geometry, text
    labels, and DIMENSION entities directly from DXF vector data via
    `ezdxf`. No rasterization, no OpenCV, no OCR, no scale estimation --
    the drawing's own coordinates and $INSUNITS are the source of truth.
    """

    def supported_document_type(self) -> DocumentType:
        return DocumentType.DXF

    # -- Contract method ---------------------------------------------------

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        """Hard wall-clock budget around the whole DXF extraction.

        Mirrors `PDFHybridExtractor.extract()`'s bounded-execution wrapper
        (see `backend/tools/bounded_execution.py`). `_build_role_inference_graph`
        already caps the O(n^2) role-inference graph to a bounded polygon
        count, but this is defense in depth for any other pathological
        input (a DXF with an enormous entity count elsewhere, a malformed
        file that makes `ezdxf` itself slow) -- the caller must always get
        an explicit, bounded failure rather than a hang, never mind why.
        """
        from backend.config import get_settings

        settings = get_settings()
        try:
            return bounded_execution.run_with_timeout(
                lambda: self._extract_impl(document_path, document_id),
                timeout_seconds=settings.extraction_timeout_seconds,
                operation=f"DXF extraction of document '{document_id}'",
            )
        except bounded_execution.ExtractionTimeoutError as exc:
            logger.warning(str(exc))
            return ExtractionResult(
                document_id=document_id,
                document_type=DocumentType.DXF,
                page_count=1,
                warnings=[
                    f"TIMEOUT: {exc}. DXF extraction was aborted with no candidates/dimensions "
                    "returned -- this is an explicit failure, not a silently degraded result."
                ],
                extractor_name=EXTRACTOR_NAME,
                extractor_version=EXTRACTOR_VERSION,
            )

    def _extract_impl(self, document_path: Path, document_id: str) -> ExtractionResult:
        import ezdxf
        from ezdxf import recover as ezdxf_recover

        # The TRUE start of this document's extraction wall-clock budget --
        # threaded down to `_resolve_via_regions`'s evidence-recovery time
        # check. `ezdxf.readfile` alone measured at 63.6s for a real
        # 45,000-entity/45MB file -- over a third of the whole 180s
        # extraction timeout consumed before a single line of this
        # extractor's own logic runs. A time-budget check that only
        # started its own clock later (e.g. at the start of region
        # resolution) would silently under-count how much of the overall
        # budget was already gone and could still let evidence recovery
        # push the whole extraction over its timeout on a large file.
        extraction_start_time = time.time()
        warnings: list[str] = []

        try:
            doc = ezdxf.readfile(str(document_path))
        except ezdxf.DXFStructureError:
            try:
                doc, auditor = ezdxf_recover.readfile(str(document_path))
                warnings.append(
                    "DXF file had structural errors; recovered with ezdxf's audit/recovery mode -- "
                    "some entities may be missing or approximated."
                )
                if auditor.has_errors:
                    warnings.append(f"{len(auditor.errors)} unresolved structural issue(s) remained after recovery.")
            except Exception as exc:
                return self._failure(document_id, f"Could not parse DXF file, even in recovery mode: {exc}")
        except Exception as exc:
            return self._failure(document_id, f"Could not open DXF file: {exc}")

        try:
            msp = doc.modelspace()
        except Exception as exc:
            return self._failure(document_id, f"DXF has no readable modelspace: {exc}")

        raw_polygons, raw_texts, raw_dims, raw_open_chains, raw_open_segments = self._collect_raw(msp, warnings)

        insunits = None
        try:
            insunits = doc.header.get("$INSUNITS")
        except Exception:
            pass

        largest_raw_area = max((p.polygon.area for p in raw_polygons), default=None)
        factor, unit_reason, unit_confidence = _resolve_unit_factor(insunits, largest_raw_area)
        warnings.append(f"Unit resolution: {unit_reason}")
        if unit_confidence < 0.8:
            warnings.append(
                f"Unit confidence is LOW ({unit_confidence:.2f}) -- every length/area measurement "
                "below derived from DXF coordinates has its confidence capped accordingly."
            )

        polygons = [p.scaled(factor) for p in raw_polygons]
        texts = [t.scaled(factor) for t in raw_texts]
        dims = [d.scaled(factor) for d in raw_dims]
        open_chains = [[(x * factor, y * factor) for x, y in chain] for chain in raw_open_chains]
        open_segments = [
            (layer, (sx * factor, sy * factor), (ex * factor, ey * factor))
            for layer, (sx, sy), (ex, ey) in raw_open_segments
        ]

        if not polygons:
            warnings.append(
                "No closed polygon geometry (LWPOLYLINE/POLYLINE) found in this DXF -- "
                "plot/building geometry cannot be derived."
            )

        text_evidence = [
            TextEvidence(
                source_type=SourceType.TEXT,
                raw_text=t.text,
                page=0,
                bounding_box=_point_bbox(t.position),
            )
            for t in texts
        ]

        dimensions = [
            Dimension(
                label=d.raw_text or f"{d.value:.3f} m (DIMENSION entity)",
                magnitude=round(d.value, 4),
                unit="m",
                page=0,
            )
            for d in dims
            if d.value
        ]

        independent_cv = self._build_independent_cv(
            document_id, polygons, texts, dims, open_chains, open_segments, warnings, unit_confidence,
            extraction_start_time,
        )

        return ExtractionResult(
            document_id=document_id,
            document_type=DocumentType.DXF,
            page_count=1,
            plot_candidates=[],
            building_candidates=[],
            road_candidates=[],
            dimensions=dimensions,
            text_evidence=text_evidence,
            spatial_relations=[],
            vision_pages=[],
            independent_cv=independent_cv,
            warnings=warnings,
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
        )

    # -- Failure helper ------------------------------------------------------

    @staticmethod
    def _failure(document_id: str, message: str) -> ExtractionResult:
        logger.warning(message)
        return ExtractionResult(
            document_id=document_id,
            document_type=DocumentType.DXF,
            page_count=1,
            warnings=[message],
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
        )

    # -- Raw entity collection -----------------------------------------------

    def _collect_raw(
        self, msp, warnings: list[str]
    ) -> tuple[
        list[_RawPoly],
        list[_RawText],
        list[_RawDim],
        list[list[tuple[float, float]]],
        list[tuple[str, tuple[float, float], tuple[float, float]]],
    ]:
        polygons: list[_RawPoly] = []
        texts: list[_RawText] = []
        dims: list[_RawDim] = []
        # Open line-work (walls drawn as disconnected LINE entities, or an
        # open/unclosed polyline) is never used directly for measurement --
        # only a CLOSED polygon can be a plot/building/road candidate, and a
        # fabricated closure would be exactly the kind of invented geometry
        # this project avoids. It IS useful for two additive, strictly-
        # deterministic-or-nothing paths when no closed building polygon
        # exists: `dxf_reconstruction.reconstruct_building_polygon` (tried
        # first -- stitches real wall segments back into a closed outline
        # from their own exact coordinates) and, only if that also finds
        # nothing, `_vision_fallback_regions` (renders the line-work as an
        # image and asks Vision to visually identify a region). `open_chains`
        # keeps whole polylines together (what the renderer wants);
        # `open_segments` below decomposes the same entities into individual
        # (layer, start, end) segments (what the reconstruction pipeline
        # wants, since it must be able to drop segments by layer and re-graph
        # them at the segment level).
        open_chains: list[list[tuple[float, float]]] = []
        open_segments: list[tuple[str, tuple[float, float], tuple[float, float]]] = []

        for e in _iter_entities_flat(msp, warnings):
            try:
                dxftype = e.dxftype()
            except Exception:
                continue
            try:
                layer = e.dxf.layer
            except Exception:
                layer = "0"

            try:
                if dxftype == "LWPOLYLINE":
                    has_bulge = bool(getattr(e, "has_arc", False))
                    if has_bulge:
                        pts = _flatten_lwpolyline_points(e)
                        if not pts:
                            # Flattening failed for some reason (malformed
                            # geometry) -- fall back to the straight-line
                            # approximation rather than dropping the entity
                            # entirely.
                            pts = [(p[0], p[1]) for p in e.get_points("xy")]
                            has_bulge = False
                    else:
                        pts = [(p[0], p[1]) for p in e.get_points("xy")]
                    if not e.closed:
                        if len(pts) >= 2:
                            open_chains.append(pts)
                            for a, b in zip(pts, pts[1:]):
                                open_segments.append((layer, a, b))
                        continue
                    poly = _make_polygon(pts)
                    if poly is not None:
                        polygons.append(_RawPoly(layer=layer, polygon=poly, approximated_curve=has_bulge))

                elif dxftype == "POLYLINE" and e.is_2d_polyline:
                    # Old-style 2D POLYLINE vertices can also carry a bulge
                    # (`vertex.dxf.bulge`), but this project's real DXF
                    # samples exclusively use LWPOLYLINE for boundaries; bulge
                    # flattening is implemented above for LWPOLYLINE only.
                    # Note this as a known, narrow limitation rather than
                    # silently pretending old-style POLYLINE bulges don't exist.
                    if any(getattr(v.dxf, "bulge", 0.0) for v in e.vertices):
                        warnings.append(
                            f"POLYLINE on layer '{layer}' has bulge (arc) vertices, which are "
                            "approximated as straight segments -- only LWPOLYLINE bulges are "
                            "curve-flattened."
                        )
                    pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                    if not e.is_closed:
                        if len(pts) >= 2:
                            open_chains.append(pts)
                            for a, b in zip(pts, pts[1:]):
                                open_segments.append((layer, a, b))
                        continue
                    poly = _make_polygon(pts)
                    if poly is not None:
                        polygons.append(_RawPoly(layer=layer, polygon=poly))

                elif dxftype == "LINE":
                    start, end = e.dxf.start, e.dxf.end
                    open_chains.append([(start.x, start.y), (end.x, end.y)])
                    open_segments.append((layer, (start.x, start.y), (end.x, end.y)))

                elif dxftype == "ARC":
                    # A standalone ARC is structural line-work (a curved wall
                    # segment or boundary fillet), not a closed shape on its
                    # own -- feed it into the open-chain/segment pools
                    # exactly like a LINE, so it participates in fragmented-
                    # geometry reconstruction (dxf_reconstruction.py) instead
                    # of being silently invisible to the whole pipeline.
                    try:
                        radius = float(e.dxf.radius)
                    except Exception:
                        radius = 0.0
                    pts = _flatten_curve_points(e, radius)
                    if len(pts) >= 2:
                        open_chains.append(pts)
                        for a, b in zip(pts, pts[1:]):
                            open_segments.append((layer, a, b))

                elif dxftype == "CIRCLE":
                    # A full circle (a circular column, a curved planter/
                    # driveway edge) is always closed by construction --
                    # flatten it directly into a polygon candidate.
                    try:
                        radius = float(e.dxf.radius)
                    except Exception:
                        radius = 0.0
                    pts = _flatten_curve_points(e, radius)
                    poly = _make_polygon(pts)
                    if poly is not None:
                        polygons.append(_RawPoly(layer=layer, polygon=poly, approximated_curve=True))

                elif dxftype in ("TEXT", "ATTRIB", "ATTDEF"):
                    txt = (e.dxf.text or "").strip()
                    if txt:
                        pos = e.dxf.insert
                        texts.append(_RawText(text=txt, position=(pos.x, pos.y), layer=layer))

                elif dxftype == "MTEXT":
                    try:
                        txt = e.plain_text().strip()
                    except Exception:
                        txt = (getattr(e, "text", "") or "").strip()
                    if txt:
                        pos = e.dxf.insert
                        texts.append(_RawText(text=txt, position=(pos.x, pos.y), layer=layer))

                elif dxftype == "DIMENSION":
                    try:
                        value = e.get_measurement()
                    except Exception:
                        value = None
                    if value:
                        try:
                            pos = e.dxf.text_midpoint
                            position = (pos.x, pos.y)
                        except Exception:
                            position = (0.0, 0.0)
                        raw_text = ""
                        try:
                            raw_text = e.dxf.text or ""
                        except Exception:
                            pass
                        if raw_text == "<>":
                            raw_text = ""
                        # LINEAR (0) / ALIGNED (1) dimensions measure the
                        # straight-line span between their two extension-
                        # line origin points -- `e.dimtype` is already
                        # stripped of ezdxf's binary flag bits, so this is
                        # a clean type check. Other dimension types
                        # (radius/diameter/angular/ordinate) don't have a
                        # "defpoint2 to defpoint3" span in this sense, so
                        # defpoint2/defpoint3 are left None for them --
                        # `_find_dimension_edge_matches` skips those.
                        defpoint2 = defpoint3 = None
                        try:
                            if e.dimtype in (0, 1):
                                p2, p3 = e.dxf.defpoint2, e.dxf.defpoint3
                                defpoint2, defpoint3 = (p2.x, p2.y), (p3.x, p3.y)
                        except Exception:
                            pass
                        dims.append(_RawDim(
                            value=float(value), position=position, raw_text=raw_text, layer=layer,
                            defpoint2=defpoint2, defpoint3=defpoint3,
                        ))
            except Exception as exc:  # pragma: no cover - defensive against malformed entities
                warnings.append(f"Skipped an unparsable {dxftype} entity: {exc}")
                continue

        return polygons, texts, dims, open_chains, open_segments

    # -- IndependentCVResult construction ------------------------------------

    def _build_independent_cv(
        self,
        document_id: str,
        polygons: list[_RawPoly],
        texts: list[_RawText],
        dims: list[_RawDim],
        open_chains: list[list[tuple[float, float]]],
        open_segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
        warnings: list[str],
        unit_confidence: float = 1.0,
        extraction_start_time: Optional[float] = None,
    ) -> IndependentCVResult:
        measurements: list[IndependentMeasurement] = []

        # Region-scoped resolution first: on a sheet with more than one
        # spatially distinct drawing (site plan + floor plans + elevations
        # + a title block, all commonly sharing layer '0' with no
        # distinguishing names on a real vectorized-trace DXF), this both
        # avoids selecting a cross-cutting sheet-border/frame polygon as
        # "the plot" and gives fragmented-line-work reconstruction a small,
        # tractable per-region subgraph instead of one sheet-wide connected
        # component. See `_resolve_via_regions`'s docstring for exactly
        # when this can and cannot change the result -- it is a strict
        # no-op on a normal single-drawing DXF.
        region_result = _resolve_via_regions(polygons, open_segments, texts, warnings, extraction_start_time)
        if region_result is not None:
            plot_entry, building_entry, road_entry = region_result
        else:
            # Built ONCE, capped, and shared across all three role queries --
            # see _build_role_inference_graph's docstring for why (this used to
            # rebuild an O(n^2) graph three separate times over the FULL,
            # uncapped polygon list, which is what made a DXF with many
            # thousands of tiny closed fragments appear to hang indefinitely).
            plot_entry, building_entry, road_entry = _resolve_plot_building_road(
                polygons, open_segments, warnings
            )

        # Vision-render fallback: only when the deterministic closed-polygon
        # search AND the fragmented-geometry reconstruction above both found
        # no plausible building (and/or no road, which reconstruction does
        # not attempt). Renders this DXF's own line-work (open AND closed
        # chains -- see _collect_raw) and asks Vision to visually identify
        # which region is which; the returned bbox is mapped back to exact
        # DXF world coordinates via the render's own verified affine
        # transform, never through any measurement Vision itself reports
        # (see _vision_fallback_regions's docstring). No-ops (returns {}) if
        # Vision is disabled/unavailable or finds nothing, so the
        # deterministic paths above are unaffected when this doesn't apply.
        if building_entry is None or road_entry is None:
            fallback_regions = _vision_fallback_regions(document_id, polygons, open_chains, warnings)
            if building_entry is None and "BUILDING" in fallback_regions:
                building_entry = _reject_implausible_building(
                    _raw_poly_from_bbox(fallback_regions["BUILDING"], layer=_VISION_FALLBACK_LAYER),
                    plot_entry, warnings,
                )
            if road_entry is None and "ROAD" in fallback_regions:
                road_entry = _raw_poly_from_bbox(fallback_regions["ROAD"], layer=_VISION_FALLBACK_LAYER)

        if plot_entry is not None:
            plot_poly = plot_entry.polygon
            plot_bbox = plot_poly.bounding_box
            is_envelope = plot_entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER
            is_fitted_boundary = is_envelope and plot_entry.envelope_method == "fragment_boundary"
            plot_source = "VECTOR_GEOMETRY_RECONSTRUCTED" if is_envelope else "VECTOR_GEOMETRY"
            # The exact-geometry confidence (0.98/0.95) only holds when the
            # drawing-unit -> metre scale itself is trustworthy AND the
            # polygon itself is an exact closed boundary rather than a
            # fitted envelope; an uncertain unit resolution (e.g. no
            # $INSUNITS header and no plausible-area sanity check) or an
            # envelope-reconstructed plot must not be hidden behind a
            # hardcoded high confidence on every downstream measurement.
            # The line-fitting reconstruction (`fragment_boundary`) earns a
            # higher cap than the cruder bounding-rectangle fallback: it
            # traces the fragments' own straight runs rather than just
            # wrapping every point in one rectangle, so it is measurably
            # less exposed to a stray unrelated point inflating the result.
            plot_curve_cap = 0.90 if plot_entry.approximated_curve else 1.0
            # This region only cleared the usability bar because of a
            # drawing-type caption match (see `_CAPTION_OVERRIDE_CONFIDENCE_
            # CAP`'s own docstring), OR no caption was found anywhere on a
            # sheet with multiple plausible-looking regions (item 9, see
            # `_RawPoly.unconfirmed_drawing_identity`). Applied to every
            # plot confidence below, including the DIMENSION-cross-check-
            # confirmed path, which otherwise bypasses `plot_envelope_cap`
            # entirely.
            plot_implausibility_cap = (
                _CAPTION_OVERRIDE_CONFIDENCE_CAP
                if plot_entry.caption_overrode_implausible_score or plot_entry.unconfirmed_drawing_identity
                else 1.0
            )
            if not is_envelope:
                plot_envelope_cap = 1.0
            elif is_fitted_boundary:
                plot_envelope_cap = 0.85
            else:
                plot_envelope_cap = 0.75
            # Width/depth are measured from the polygon's own minimum-area
            # ORIENTED bounding rectangle, never `plot_bbox.width/height`
            # (axis-aligned) -- see `_oriented_width_depth`'s docstring.
            # `oriented_rectangularity` additionally caps confidence when
            # the polygon is a poor fit to any rectangle at all (a
            # genuinely irregular plot), independent of the sheet's rotation.
            plot_width, plot_depth, plot_oriented_rectangularity = _oriented_width_depth(plot_poly)
            plot_shape_cap = 1.0 if plot_oriented_rectangularity >= 0.85 else max(0.5, plot_oriented_rectangularity)
            plot_area_conf = min(0.98, unit_confidence, plot_curve_cap, plot_envelope_cap, plot_implausibility_cap)
            plot_len_conf = min(0.95, unit_confidence, plot_curve_cap, plot_envelope_cap, plot_shape_cap, plot_implausibility_cap)
            # Critical Requirement 5: cross-check against native DXF
            # DIMENSION entities via the real text -> dimension graphic ->
            # measured span -> geometric edge chain (see
            # `_dimension_chain_notes`), not a nearest-number guess. A
            # confirmed field's confidence is lifted to what it would be
            # WITHOUT the envelope/shape-fit discount (an exact independent
            # geometric annotation match outweighs "this edge came from a
            # fitted approximation"), while still respecting the unit-scale
            # and curve-flattening caps, which a same-drawing-units
            # dimension is equally subject to.
            plot_width_notes, plot_depth_notes, plot_dim_warnings = _dimension_chain_notes(
                dims, plot_poly, plot_width, plot_depth,
            )
            warnings.extend(plot_dim_warnings)
            plot_width_conf = (
                min(0.98, unit_confidence, plot_curve_cap, plot_implausibility_cap) if plot_width_notes else plot_len_conf
            )
            plot_depth_conf = (
                min(0.98, unit_confidence, plot_curve_cap, plot_implausibility_cap) if plot_depth_notes else plot_len_conf
            )
            if is_fitted_boundary:
                plot_evidence = (
                    f"Boundary reconstructed by fitting straight sides through collinear, well-supported "
                    f"runs of fragment/line-work points on layer '{plot_entry.layer}' (no single closed "
                    f"polygon was plot-sized)."
                )
                plot_provenance = (
                    "Plot boundary reconstructed from a scattered field of small closed fragments/line-work "
                    "(e.g. a dashed/dash-dot property boundary): fragments were grouped onto shared straight "
                    "sides by orientation and distance to a common supporting line, then those sides were "
                    "intersected to find the corners -- not read directly as one closed DXF polygon, and not "
                    "assuming any two fragments touch."
                )
            elif is_envelope:
                plot_evidence = (
                    f"Minimum-area bounding envelope fitted to fragment/line-work points on layer "
                    f"'{plot_entry.layer}' (no single closed polygon was plot-sized, and too few fragments "
                    f"shared a common supporting line to fit individual sides)."
                )
                plot_provenance = (
                    "Plot boundary reconstructed as the minimum-area oriented bounding rectangle of a "
                    "scattered field of small closed fragments/line-work (e.g. a dashed/dash-dot property "
                    "boundary) -- not read directly as one closed DXF polygon."
                )
            else:
                plot_evidence = f"Shoelace area of the closed polygon on layer '{plot_entry.layer}'."
                plot_provenance = "Exact area computed from DXF vector geometry -- no rasterization or scale estimation involved."
            if plot_entry.caption_overrode_implausible_score:
                plot_provenance += (
                    " This region was only selected because it carries a recognized drawing-type caption "
                    "(e.g. 'SITE PLAN') confirming it is the right DRAWING -- its own structural resolution "
                    "score was below this pipeline's normal usability bar (most often because its "
                    "reconstructed plot area looks implausible at its current scale), so confidence here is "
                    "capped low regardless of source until that scale disagreement is separately resolved."
                )
            if plot_entry.unconfirmed_drawing_identity:
                plot_provenance += (
                    " No drawing-type caption was recognized anywhere on this sheet, yet multiple regions "
                    "independently scored as plausible drawings on their own structural merit -- this "
                    "pipeline cannot positively confirm this region is actually the site plan rather than "
                    "an unrelated drawing (a section view, a floor plan, a fragment merge) that happened "
                    "to score well, so confidence here is capped low regardless of source."
                )
            measurements.append(_numeric_measurement(
                "plot.area", plot_poly.area, "m2", plot_source, plot_area_conf,
                [plot_evidence], plot_provenance, plot_bbox,
            ))
            measurements.append(_length_measurement(
                "plot.width", plot_width, plot_source, plot_width_conf,
                [f"Minimum-area oriented bounding rectangle width of the plot polygon (layer '{plot_entry.layer}')."]
                + plot_width_notes,
                "Plot width measured from the polygon's own orientation, not an axis-aligned bounding "
                "box -- correct regardless of the sheet's rotation.", plot_bbox,
            ))
            measurements.append(_length_measurement(
                "plot.depth", plot_depth, plot_source, plot_depth_conf,
                [f"Minimum-area oriented bounding rectangle depth of the plot polygon (layer '{plot_entry.layer}')."]
                + plot_depth_notes,
                "Plot depth measured from the polygon's own orientation, not an axis-aligned bounding "
                "box -- correct regardless of the sheet's rotation.", plot_bbox,
            ))
        else:
            warnings.append(
                "Could not identify a plot-boundary polygon: no layer name matched PLOT/BOUNDARY/SITE "
                "and no closed polygon was found at all."
            )

        if building_entry is not None:
            b_poly = building_entry.polygon
            b_bbox = b_poly.bounding_box
            b_source, b_conf = _entry_source_confidence(building_entry, 0.97, unit_confidence)
            if building_entry.layer == _VISION_FALLBACK_LAYER:
                b_provenance = (
                    "Building outline visually identified by Vision on a rendered image of this DXF's own "
                    "line-work (no closed building polygon existed in the vector data, and fragmented-"
                    "geometry reconstruction from the open line-work also found nothing plausible); the "
                    "reported area/width/depth are still computed directly from exact DXF world coordinates "
                    "via the render's own verified pixel<->world transform, not read off the image by Vision "
                    "itself."
                )
            elif building_entry.layer == _RECONSTRUCTED_LAYER:
                b_provenance = (
                    "No closed building polygon existed in the DXF vector data; reconstructed from the "
                    "DXF's own disconnected wall/boundary line-work (endpoint snapping, collinear-segment "
                    "merging, and closed-loop cycle detection over the resulting graph -- no rasterization, "
                    "no model involved). Area/width/depth are computed from the reconstructed polygon's "
                    "exact DXF coordinates."
                )
            elif building_entry.layer == _WALL_UNION_RECONSTRUCTED_LAYER:
                b_provenance = (
                    "No single closed wall loop could be traced from the DXF's wall line-work (a door/"
                    "window/opening gap, a T-junction from an interior partition wall, or overlapping/"
                    "duplicated line-work can all prevent one from existing even when the wall network is "
                    "otherwise intact); reconstructed instead by buffering each wall segment into a thin area "
                    "using a wall thickness estimated from this document's own parallel wall-face line-work, "
                    "bridging document-relative opening-sized gaps between dangling wall endpoints, unioning "
                    "everything, and taking the union's outer boundary. This is an area-based reconstruction, "
                    "not a traced footprint outline -- treat it as approximate, but closer to the real "
                    "footprint shape than a single bounding rectangle."
                )
            elif building_entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER and building_entry.envelope_method == "fragment_boundary":
                b_provenance = (
                    "No closed building outline could be traced from the DXF's wall line-work (a real "
                    "door/window gap in a wall centerline trace is larger than the endpoint-snapping "
                    "tolerance closed-loop reconstruction uses); reconstructed instead by fitting straight "
                    "sides through collinear, well-supported runs of the wall/line-work segment endpoints and "
                    "intersecting those sides to find the corners. This is a boundary reconstruction from "
                    "fragments, not a traced footprint outline -- treat it as a coarser approximation than a "
                    "reconstructed closed loop, but a closer fit than a single bounding rectangle."
                )
            elif building_entry.layer == _ENVELOPE_RECONSTRUCTED_LAYER:
                b_provenance = (
                    "No closed building outline could be traced from the DXF's wall line-work (a real "
                    "door/window gap in a wall centerline trace is larger than the endpoint-snapping "
                    "tolerance closed-loop reconstruction uses), and too few of the wall/line-work segment "
                    "endpoints shared a common supporting line to fit individual sides; approximated instead "
                    "as the minimum-area oriented bounding rectangle of those endpoints. This is an "
                    "envelope around the building's own line-work, not a traced footprint outline -- treat "
                    "it as a coarser approximation than a reconstructed closed loop."
                )
            else:
                b_provenance = "Building footprint area computed from DXF vector geometry."
            if building_entry.caption_overrode_implausible_score:
                b_provenance += (
                    " This region was only selected because it carries a recognized drawing-type caption "
                    "(e.g. 'SITE PLAN') confirming it is the right DRAWING -- its own structural resolution "
                    "score was below this pipeline's normal usability bar (most often because its "
                    "reconstructed plot area looks implausible at its current scale), so confidence here is "
                    "capped low regardless of source until that scale disagreement is separately resolved."
                )
            if building_entry.unconfirmed_drawing_identity:
                b_provenance += (
                    " No drawing-type caption was recognized anywhere on this sheet, yet multiple regions "
                    "independently scored as plausible drawings on their own structural merit -- this "
                    "pipeline cannot positively confirm this region is actually the site plan rather than "
                    "an unrelated drawing (a section view, a floor plan, a fragment merge) that happened "
                    "to score well, so confidence here is capped low regardless of source."
                )
            # Width/depth measured from the building polygon's own minimum-
            # area ORIENTED bounding rectangle -- see `_oriented_width_
            # depth`'s docstring; same reasoning as the plot measurement
            # above, and just as applicable here (a rotated real building
            # outline is not an edge case).
            b_width, b_depth, b_oriented_rectangularity = _oriented_width_depth(b_poly)
            b_shape_cap = 1.0 if b_oriented_rectangularity >= 0.85 else max(0.5, b_oriented_rectangularity)
            b_len_conf = min(b_conf, 0.95, b_shape_cap)
            # Critical Requirement 5: same native-DXF-DIMENSION verification
            # chain as the plot section above (see `_dimension_chain_notes`).
            b_width_notes, b_depth_notes, b_dim_warnings = _dimension_chain_notes(dims, b_poly, b_width, b_depth)
            warnings.extend(b_dim_warnings)
            b_width_conf = min(0.98, unit_confidence, b_conf) if b_width_notes else b_len_conf
            b_depth_conf = min(0.98, unit_confidence, b_conf) if b_depth_notes else b_len_conf
            measurements.append(_numeric_measurement(
                "building.footprint_area", b_poly.area, "m2", b_source, b_conf,
                [f"Shoelace area of the building-footprint polygon (layer '{building_entry.layer}')."],
                b_provenance, b_bbox,
            ))
            measurements.append(_length_measurement(
                "building.width", b_width, b_source, b_width_conf,
                [f"Minimum-area oriented bounding rectangle width of the building polygon (layer '{building_entry.layer}')."]
                + b_width_notes,
                "Building width measured from the polygon's own orientation, not an axis-aligned "
                "bounding box -- correct regardless of the sheet's rotation.", b_bbox,
            ))
            measurements.append(_length_measurement(
                "building.depth", b_depth, b_source, b_depth_conf,
                [f"Minimum-area oriented bounding rectangle depth of the building polygon (layer '{building_entry.layer}')."]
                + b_depth_notes,
                "Building depth measured from the polygon's own orientation, not an axis-aligned "
                "bounding box -- correct regardless of the sheet's rotation.", b_bbox,
            ))
        else:
            warnings.append(
                "Could not identify a building-footprint polygon nested within the plot boundary "
                "(no BUILDING/BLDG/WALL layer match, no plausible nested closed polygon, and either "
                "Vision is disabled or the render-fallback found nothing plausible either)."
            )

        road_text = _find_road_text(texts)
        road_bbox = road_entry.polygon.bounding_box if road_entry is not None else None
        if road_text is not None:
            value, raw = road_text
            measurements.append(_length_measurement(
                "road.width", value, "NATIVE_TEXT", 0.97,
                [raw], "Explicit road-width label read directly from DXF text (no OCR).",
            ))
        elif road_entry is not None:
            rb = road_entry.polygon.bounding_box
            width = min(rb.width, rb.height)
            r_source, r_conf = _entry_source_confidence(road_entry, 0.85, unit_confidence)
            measurements.append(_length_measurement(
                "road.width", width, r_source, r_conf,
                [f"Shorter side of the road polygon's bounding box (layer '{road_entry.layer}')."],
                "Road width taken as the narrow dimension of the road-layer polygon; verify manually if "
                "the road is drawn at an angle to the plot.",
                rb,
            ))

        for field, pattern, unit in _AREA_FIELD_PATTERNS:
            candidates: list[tuple[_RawText, float]] = []
            for t in texts:
                haystack = _without_row_formulas(t.text)
                m = pattern.search(haystack)
                if not m:
                    continue
                if unit != "%" and _PERCENT_TAIL_RE.match(haystack, m.end("value")):
                    continue  # "Proposed Coverage Area (62.83 %)" is a percentage, not 62.83 m2
                try:
                    value = float(m.group("value"))
                except (TypeError, ValueError):
                    continue
                candidates.append((t, value))
            selected = _select_labeled_area_match(candidates)
            if selected is None:
                continue
            t, value = selected
            measurements.append(IndependentMeasurement(
                field=field, value=round(value, 4), unit=unit, source="NATIVE_TEXT", confidence=0.9,
                evidence=[t.text.strip()],
                note="Explicit labelled value read directly from DXF text -- no OCR involved.",
                page=0,
                geometry_bbox_pts=_bbox_list(_point_bbox(t.position)),
            ))

        if plot_entry is not None and building_entry is not None:
            access_evidence = collect_access_evidence(
                [
                    TextEvidence(source_type=SourceType.TEXT, raw_text=t.text, page=0, bounding_box=_point_bbox(t.position))
                    for t in texts
                ],
                page=0,
            )
            fsr = resolve_front_side(plot_entry.polygon, road_bbox, access_evidence)
            side_confidence = {
                ConfidenceLevel.HIGH: 0.9,
                ConfidenceLevel.MEDIUM: 0.75,
                ConfidenceLevel.LOW: 0.5,
                ConfidenceLevel.MISSING: 0.0,
            }.get(fsr.confidence, 0.5)

            if side_confidence > 0:
                side_edges = {
                    "front": fsr.front_edges, "rear": fsr.rear_edges,
                    "left": fsr.left_edges, "right": fsr.right_edges,
                }
                for side, edges in side_edges.items():
                    if not edges:
                        continue
                    # Distance from the building polygon's own boundary
                    # (vertices AND edges, so a near-parallel wall-to-wall
                    # gap is measured correctly too) to this plot side --
                    # NEVER the building's axis-aligned bbox corners. A
                    # bbox corner is a synthetic point that generally isn't
                    # any real point on a rotated building at all, and
                    # measuring from it against a plot edge that is ALSO
                    # not page-axis-aligned (a rotated sheet) silently
                    # produces a wrong gap -- confirmed directly: a 20x15
                    # plot and an 11x7 building both rotated 30 degrees
                    # together, true setbacks {3, 5, 3, 6} m, reported
                    # {1.76, 0.24, 2.97, 0.03} m from bbox corners before
                    # this fix. Matches `setbacks.py`'s own already-correct
                    # PDF-pipeline computation (`geo.polygon_to_edges_
                    # distance`) instead of re-deriving a second, buggy
                    # version of the same idea.
                    gap = geo.polygon_to_edges_distance(building_entry.polygon, edges)
                    field = f"setbacks.{side}"
                    explicit = _explicit_setback_value(texts, side)
                    note = (
                        f"Perpendicular gap between the building footprint and the {side} plot edge, "
                        f"measured directly from DXF vector geometry. Front side resolved via: {fsr.reasoning}"
                    )
                    if explicit is not None:
                        exp_value, exp_text = explicit
                        tol = max(0.15, 0.1 * max(exp_value, gap))
                        if abs(exp_value - gap) <= tol:
                            measurements.append(_length_measurement(
                                field, (exp_value + gap) / 2.0, "VECTOR_GEOMETRY", min(0.99, side_confidence + 0.05),
                                [f"{side} edge <-> building gap = {gap:.2f} m (geometry)", f"printed label: '{exp_text}'"],
                                f"Geometric gap and printed '{side} setback' label agree. {note}",
                            ))
                            continue
                        warnings.append(
                            f"{side} setback: geometric gap ({gap:.2f} m) and printed label "
                            f"('{exp_text}') disagree by more than tolerance; using the geometric gap "
                            "(exact vector data) and flagging the discrepancy."
                        )
                    measurements.append(_length_measurement(
                        field, gap, "VECTOR_GEOMETRY", side_confidence,
                        [f"{side} edge <-> building footprint gap"], note,
                    ))
        elif plot_entry is not None and building_entry is None:
            warnings.append("Setbacks not computed: no building-footprint polygon was identified.")

        return IndependentCVResult(
            document_id=document_id,
            pages_analyzed=[0],
            measurements=measurements,
            warnings=[],
            site_plan_page=0,
            site_plan_bbox_pts=_bbox_list(plot_entry.polygon.bounding_box) if plot_entry else None,
            scale_points_per_metre=None,
            scale_confidence=None,
        )


def _length_measurement(
    field: str, value_m: Optional[float], source: str, confidence: float,
    evidence: list[str], note: str, bbox: Optional[BoundingBox] = None,
) -> IndependentMeasurement:
    return IndependentMeasurement(
        field=field,
        value_m=None if value_m is None else round(float(value_m), 4),
        source=source,
        confidence=confidence,
        evidence=evidence,
        note=note,
        page=0,
        geometry_bbox_pts=_bbox_list(bbox),
    )


def _numeric_measurement(
    field: str, value: Optional[float], unit: str, source: str, confidence: float,
    evidence: list[str], note: str, bbox: Optional[BoundingBox] = None,
) -> IndependentMeasurement:
    return IndependentMeasurement(
        field=field,
        value=None if value is None else round(float(value), 4),
        unit=unit,
        source=source,
        confidence=confidence,
        evidence=evidence,
        note=note,
        page=0,
        geometry_bbox_pts=_bbox_list(bbox),
    )


__all__ = ["DXFHybridExtractor", "EXTRACTOR_NAME", "EXTRACTOR_VERSION"]
