"""
Printed drawing-scale notes ("Scale 1:200") as a first-class scale source.

Every architectural sheet that came out of a CAD package states its own
scale in print, usually once per drawing view. That note is an exact,
authoritative statement of the page-points -> metres transform, and it is
available even on sheets that carry no numeric dimension labels at all.

This matters because the rest of the pipeline derived scale exclusively
from `dimension_line_length_pts / dimension_magnitude_metres`
(`spatial_reasoning/scale.py`). That derivation requires the drawing to
carry printed edge dimensions AND requires each one to be correctly
associated with its own dimension line. On a real BBMP sanctioned plan
(`the original permit drawing`) neither holds: the site plan has no printed
plot-dimension labels whatsoever, so plot width/depth, building
width/depth and all four setbacks resolved to null -- while the sheet
said `Scale 1:200` in plain text the whole time.

Accuracy of the printed note, measured against the one bundled plan whose
label-derived scale does resolve (PLAN2):

    printed "1:200"                       -> 14.1732 pt/m
    PLAN2's label-derived scale           -> 14.1668 pt/m
    relative error                        ->  0.045%

This module only READS the note. Deciding whether to trust it over a
label-derived estimate, and cross-checking it against the sheet's own
area statement, is the caller's job -- see
`cv_extraction/site_plan.py::_resolve_scale`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from backend.cv_extraction.raw_types import RawTextItem
from backend.schemas.geometry import BoundingBox

# One PostScript point is 1/72 inch, i.e. 25.4/72 mm on the physical sheet.
_MM_PER_POINT = 25.4 / 72.0

# "Scale 1:200", "SCALE - 1:100", "1:50", "SCALE 1 : 75". The "scale" word is
# optional because CAD title blocks frequently print a bare ratio under a
# view, but a note that does carry the word is ranked higher (see
# `ScaleNote.has_scale_keyword`).
#
# `(?!\s*[:=]\s*\d)` rejects three-term ratios: "P.C.C in mix 1:5:10" is a
# cement/sand/aggregate proportion, not a drawing scale.
_SCALE_NOTE_RE = re.compile(
    r"(?P<keyword>\bscale\b\s*[:\-]?\s*)?"
    r"\b1\s*[:=]\s*(?P<denominator>\d{1,4})\b(?!\s*[:=]\s*\d)",
    re.IGNORECASE,
)

# Architectural drawing scales start at 1:10 (1:1 through 1:8 are shop-detail
# scales that do not appear on a plan sheet, and colliding with them costs
# far more than losing them -- see `_MIX_RATIO_CONTEXT_RE`). Above 1:5000 a
# ratio is not a drawing scale at all.
_MIN_DENOMINATOR = 10
_MAX_DENOMINATOR = 5000

# A ratio inside one of these contexts is a regulation ratio, a date, or a
# revision number -- never a drawing scale.
_NOT_A_SCALE_CONTEXT_RE = re.compile(
    r"\b(far|f\.a\.r|ratio|floor\s+area|coverage|date|version|ward|no\.?\s*of)\b",
    re.IGNORECASE,
)

# Cement-mortar and concrete MIX proportions are written in exactly the same
# "1:N" shape as a drawing scale and appear all over a working drawing's
# specification notes: "0.15th in C.M 1:6", "BRICK WORK IN CM 1:5",
# "P.C.C in mix 1:5:10", "FLOORING CONCRETE 1:5:10". Read as scales they
# would imply 472-567 pt/m, roughly 40x too large, which would silently
# shrink every derived measurement by the same factor.
_MIX_RATIO_CONTEXT_RE = re.compile(
    r"\b(c\.?m|c\.?c|p\.?c\.?c|r\.?c\.?c|r\.?r|mix|mortar|cement|concrete|"
    r"brick|block\s*work|plaster|screed|grout|proportion)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScaleNote:
    """A drawing-scale ratio as printed on the sheet."""

    denominator: int
    """The N in '1:N' -- one unit on paper equals N units in reality."""

    points_per_metre: float
    """The page-points-per-real-metre implied by `denominator`."""

    raw_text: str
    bounding_box: BoundingBox
    page: int

    has_scale_keyword: bool = False
    """
    Whether the literal word "scale" preceded the ratio. A sheet that prints
    several ratios is more trustworthy where it says so explicitly, so
    callers rank keyword-bearing notes above bare ones.
    """

    def distance_to(self, point) -> float:
        """Euclidean distance from this note's centre to `point`."""
        centre = self.bounding_box.center
        return math.hypot(centre.x - point.x, centre.y - point.y)


def points_per_metre_for_denominator(denominator: float) -> float:
    """
    Convert a '1:N' drawing scale into page-points per real-world metre.

        1 pt on paper  = 25.4/72 mm on paper
                       = N * 25.4/72 mm in reality
                       = N * 25.4/72 / 1000 m in reality

    so   points_per_metre = 1 / (N * 25.4/72/1000) = 72000 / (25.4 * N).
    """
    if denominator <= 0:
        raise ValueError(f"drawing-scale denominator must be positive, got {denominator!r}")
    metres_per_point = denominator * _MM_PER_POINT / 1000.0
    return 1.0 / metres_per_point


def detect_scale_notes(text_items: list[RawTextItem], page: int) -> list[ScaleNote]:
    """
    Find every printed '1:N' drawing-scale note among `text_items`.

    Returns them in page order. A sheet legitimately carries several (one
    per view -- PLAN4 prints 1:25, 1:50, 1:75 and 1:100 on a single page),
    so this never collapses them to one: choosing which note governs a
    given drawing region is a spatial decision the caller makes with
    `nearest_scale_note`.
    """
    notes: list[ScaleNote] = []
    for item in text_items:
        text = (item.text or "").strip()
        if not text:
            continue
        if _NOT_A_SCALE_CONTEXT_RE.search(text) or _MIX_RATIO_CONTEXT_RE.search(text):
            continue
        for match in _SCALE_NOTE_RE.finditer(text):
            try:
                denominator = int(match.group("denominator"))
            except (TypeError, ValueError):  # pragma: no cover - regex guarantees digits
                continue
            if not (_MIN_DENOMINATOR <= denominator <= _MAX_DENOMINATOR):
                continue
            notes.append(
                ScaleNote(
                    denominator=denominator,
                    points_per_metre=points_per_metre_for_denominator(denominator),
                    raw_text=text,
                    bounding_box=item.bounding_box,
                    page=page,
                    has_scale_keyword=match.group("keyword") is not None,
                )
            )
    return notes


def nearest_scale_note(notes: list[ScaleNote], point) -> ScaleNote | None:
    """
    The scale note printed closest to `point` (normally a drawing region's
    centre).

    A sheet with several views prints each view's scale beside that view,
    so proximity is the correct association rule. When the sheet carries
    exactly one note it governs the whole sheet and this degenerates to
    "return it".
    """
    if not notes:
        return None
    # Keyword-bearing notes win over bare ratios at comparable distance; among
    # equals, the nearest one governs.
    return min(notes, key=lambda note: (not note.has_scale_keyword, note.distance_to(point)))


__all__ = [
    "ScaleNote",
    "detect_scale_notes",
    "nearest_scale_note",
    "points_per_metre_for_denominator",
]
