"""
Deterministic text-based extraction of plan-level classification context:
building use, development-area class (Karnataka Table 6 / draft Table 8
A/B/C), and an explicit "height excluding stilt" figure.

These are NOT geometric measurements -- they are a classification label or
a single explicitly-labeled figure, printed as ordinary text (a title
block, a property-particulars table, or an elevation/section annotation)
on the plan sheet itself. Extracted directly from `text_evidence`, which
works identically for a PDF's native/OCR text and a DXF's TEXT/MTEXT
entities since both populate the same `ExtractionResult.text_evidence`
list -- never inferred from geometry, and never guessed when the sheet
doesn't print one. `development_area` in particular is often a purely
administrative, location-based zoning classification that a drawing does
not always carry at all; when this module finds nothing, the field must
stay MISSING rather than defaulting to any particular class.

Before this module existed, these three fields were wired ONLY to a
`VisionPageResult.metadata` dict (see `pipeline.py` and
`NormalizedPlan._derive_context_from_metadata`) that nothing in the
Vision prompt/response-parsing pipeline ever actually populated -- so in
production these fields were unconditionally MISSING regardless of what
the source plan printed. This closes that gap with a real, deterministic
extractor; the existing Vision-metadata path is left in place as a
harmless secondary fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.schemas.evidence import TextEvidence

# A label explicitly naming what kind of building/use/occupancy this is,
# immediately followed by the classification word -- the common Indian
# sanction-plan phrasing ("NATURE OF WORK: RESIDENTIAL BUILDING", "USE OF
# BUILDING : COMMERCIAL"). Matched terms feed
# `NormalizedPlan._normalize_building_use` (the single canonical alias
# table already used for the Vision-metadata path), never duplicated here.
_BUILDING_USE_LABELED_RE = re.compile(
    r"\b(?:nature\s+of\s+(?:work|building|development)|type\s+of\s+building|"
    r"use\s+of\s+(?:building|premises|site)|building\s+use|occupanc(?:y|ies))\s*[:\-]?\s*"
    r"(?P<use>residential|commercial|public\s*(?:&|and)\s*semi[\s-]?public|"
    r"traffic\s*(?:&|and)\s*transportation|public\s+utilit(?:y|ies)|hospital|"
    r"(?:health\s+cent(?:re|er)|nursing\s+home)|(?:nursery|primary)\s+school|"
    r"secondary\s+school|college)\b",
    re.I,
)
# Common sheet TITLE phrasing with no explicit "nature of work:" label at
# all, e.g. "PROPOSED RESIDENTIAL BUILDING" printed as the drawing's own
# heading. Deliberately narrower than the labeled pattern above (only
# residential/commercial, the two by far most common unlabeled titles) to
# avoid over-matching incidental prose elsewhere on the sheet.
_BUILDING_USE_TITLE_RE = re.compile(
    r"\bproposed\s+(?P<use>residential|commercial)\s+(?:building|complex|apartment|house)\b",
    re.I,
)
# A longer title where the use comes at the END of the phrase, e.g. "PLAN
# SHOWING THE PROPOSED CONSTRUCTION OF GROUND FLOOR, FIRST FLOOR ... RCC ROOF
# RESIDENTIAL BUILDING, IN S.NO ...". The bare "<use> building" phrase is far
# too common in prose (regulation notes say "residential building" constantly),
# so it only counts inside a text item that ALSO carries a construction-title
# cue -- notes never open with "plan showing" or "proposed construction of".
_BUILDING_USE_IN_TITLE_RE = re.compile(
    r"\b(?P<use>residential|commercial)\s+(?:building|complex|apartment|house)\b", re.I
)
_TITLE_CUE_RE = re.compile(r"\b(?:plan\s+showing|proposed\s+construction|construction\s+of|erection\s+of)\b", re.I)

# "DEVELOPMENT AREA : A/B/C" -- the exact phrase Table 6/8 requires, with a
# tight gap and a standalone letter, so this can't accidentally match an
# unrelated nearby "A"/"B"/"C" elsewhere in the sheet's text.
_DEVELOPMENT_AREA_RE = re.compile(r"\bdevelopment\s+area\b[^A-Za-z0-9]{0,10}(?P<area>[ABC])\b", re.I)

# "HEIGHT OF BUILDING EXCLUDING STILT (FLOOR) : 12.50" and label variants
# ("HT EXCL. STILT", "HEIGHT EXCLUDING STILT FLOOR"). `h(?:eigh)?t` matches
# both "ht" (h + t) and "height" (h + "eigh" + t).
_HEIGHT_EXCLUDING_STILT_RE = re.compile(
    r"h(?:eigh)?t\.?\s*(?:of\s+(?:the\s+)?building\s*)?excl(?:uding|\.)?\s*stilt(?:\s+floor)?"
    r"[^0-9\-]{0,20}(?P<value>\d+(?:\.\d+)?)",
    re.I,
)


@dataclass
class PlanContextEvidence:
    building_use_raw: Optional[str] = None
    building_use_source_text: Optional[str] = None
    development_area: Optional[str] = None
    development_area_source_text: Optional[str] = None
    height_excluding_stilt_m: Optional[float] = None
    height_excluding_stilt_source_text: Optional[str] = None


def extract_plan_context(text_evidence: list[TextEvidence]) -> PlanContextEvidence:
    """Scan every text span once, taking the first match found for each of
    the three fields (plans repeat title-block information rarely enough
    that "first found" is the label itself, not noise)."""
    result = PlanContextEvidence()
    for t in text_evidence:
        text = (t.raw_text or "").strip()
        if not text:
            continue

        if result.building_use_raw is None:
            m = _BUILDING_USE_LABELED_RE.search(text) or _BUILDING_USE_TITLE_RE.search(text)
            if m is None and _TITLE_CUE_RE.search(text):
                m = _BUILDING_USE_IN_TITLE_RE.search(text)
            if m:
                # Normalized to lowercase here so callers (and
                # `NormalizedPlan._normalize_building_use`, whose alias
                # table is itself keyed in lowercase) get a predictable,
                # case-independent raw value regardless of how the sheet
                # printed it ("RESIDENTIAL", "Residential", "residential").
                result.building_use_raw = re.sub(r"\s+", " ", m.group("use")).strip().lower()
                result.building_use_source_text = text

        if result.development_area is None:
            m = _DEVELOPMENT_AREA_RE.search(text)
            if m:
                result.development_area = m.group("area").upper()
                result.development_area_source_text = text

        if result.height_excluding_stilt_m is None:
            m = _HEIGHT_EXCLUDING_STILT_RE.search(text)
            if m:
                try:
                    result.height_excluding_stilt_m = float(m.group("value"))
                    result.height_excluding_stilt_source_text = text
                except ValueError:
                    pass

    return result


__all__ = ["PlanContextEvidence", "extract_plan_context"]
