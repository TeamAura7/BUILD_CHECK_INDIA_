"""
Floor count from NAMED floor levels.

`pipeline._detect_floor_count` only understands numeric phrasings ("G+2",
"GF+2UF", "3 floors"). A great many sheets never print a number and instead
NAME their floors -- in the title, in one caption per floor plan, or as one
row per floor in an area table. Counting the distinct named levels is the
same information, read the way a person reads it.

Four independent forms are recognised, each reduced to a set of levels
(ground = 0, first = 1, ...):

  enumeration  one text item naming >=2 different floors
               ("... GROUND FLOOR, FIRST FLOOR AND SECOND FLOOR ...")
  captions     "<ORDINALS> FLOOR PLAN" items, where a caption may cover several
               levels ("1ST, 2ND & 3RD FLOOR PLAN", "TYPICAL FIRST & SECOND
               FLOOR PLAN")
  table cells  a text item that is just "<ORDINAL> FLOOR" (an area-table label)
  table rows   "<...> AREA IN G.F" / "F.F" / "S.F" / "T.F" rows

A form only counts if its levels are contiguous from ground: a lone
"FIRST FLOOR PLAN" (an extension sheet), or a set with a gap, says nothing
reliable about the building's height. Forms that disagree about the count
abstain rather than pick one -- a wrong count silently scales FAR, and a
missing caption (a sheet that captions ground and first but not second while
its area table lists all three) is exactly how a single form goes wrong.

CONVENTION (flag for domain review): the count is ground floor plus upper
floors. A STILT level (parking at ground level), a TERRACE/roof and open
terrace are not counted; the ruleset itself treats stilt separately
(`building_height_excluding_stilt`), and the project's audited ground truth
counts "STILT, GF+2UF" as 3. BASEMENT / CELLAR / MEZZANINE / PODIUM captions
make the count ambiguous, so those abstain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

_ORDINALS = {
    "GROUND": 0, "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
    "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
}
_ORD_WORDS = "|".join(_ORDINALS)
# "1ST" / "2ND" / "3RD" / "4TH" ... (ground has no digit form).
_ORD_TOKEN = rf"(?:{_ORD_WORDS}|\d{{1,2}}\s*(?:ST|ND|RD|TH))"
_TOKEN_RE = re.compile(rf"({_ORD_WORDS})|(\d{{1,2}})\s*(?:ST|ND|RD|TH)", re.I)

_NAMED_FLOOR_RE = re.compile(rf"\b({_ORD_WORDS})\s+FLOORS?\b", re.I)
_CAPTION_RE = re.compile(
    rf"\b(?P<list>{_ORD_TOKEN}(?:\s*(?:,|&|AND)\s*{_ORD_TOKEN})*)\s+FLOORS?\s+PLANS?\b", re.I
)
_CELL_RE = re.compile(rf"^\s*(?P<one>{_ORD_TOKEN})\s+FLOORS?\s*$", re.I)

# Conventional abbreviations for the first four levels only; beyond that
# sheets do not agree on one, so nothing is inferred.
_ABBREV_LEVELS = {"G": 0, "F": 1, "S": 2, "T": 3}
_TABLE_ROW_RE = re.compile(
    r"\b(?:PLINTH|BUILT[\s-]?UP|COVERAGE|FLOOR|CARPET)?\s*AREA\s+IN\s+([GFST])\.?\s*F\b", re.I
)

# A drawn level whose contribution to "the floor count" is genuinely
# ambiguous. Only a caption ("... FLOOR PLAN") counts as evidence such a level
# is drawn; boilerplate like "(Excluding Stilt Floor)" does not.
_AMBIGUOUS_CAPTION_RE = re.compile(r"\b(BASEMENT|CELLAR|MEZZANINE|PODIUM)\s+FLOORS?\s+PLANS?\b", re.I)


@dataclass(frozen=True)
class FloorLevelEvidence:
    count: int
    forms: tuple[str, ...]
    note: str


def _contiguous_from_ground(levels: set[int]) -> bool:
    return bool(levels) and levels == set(range(max(levels) + 1))


def _levels_in(fragment: str) -> set[int]:
    out: set[int] = set()
    for m in _TOKEN_RE.finditer(fragment):
        out.add(_ORDINALS[m.group(1).upper()] if m.group(1) else int(m.group(2)))
    return out


def _levels_by_form(texts: Iterable[str]) -> dict[str, set[int]]:
    enumeration: set[int] = set()
    captions: set[int] = set()
    cells: set[int] = set()
    table_rows: set[int] = set()
    for text in texts:
        named = {_ORDINALS[m.group(1).upper()] for m in _NAMED_FLOOR_RE.finditer(text)}
        if len(named) >= 2:
            enumeration |= named
        for m in _CAPTION_RE.finditer(text):
            captions |= _levels_in(m.group("list"))
        cell = _CELL_RE.match(text)
        if cell:
            cells |= _levels_in(cell.group("one"))
        row = _TABLE_ROW_RE.search(text)
        if row:
            table_rows.add(_ABBREV_LEVELS[row.group(1).upper()])
    return {
        "title/notes enumeration": enumeration,
        "floor-plan captions": captions,
        "area-table labels": cells,
        "area-table rows": table_rows,
    }


def floor_count_from_named_levels(texts: Iterable[str]) -> Optional[FloorLevelEvidence]:
    """The building's floor count from named floor levels, or None to abstain."""
    texts = list(texts)
    if any(_AMBIGUOUS_CAPTION_RE.search(text) for text in texts):
        return None
    forms = _levels_by_form(texts)
    # A form whose levels are not a clean ground-up run is ignored (it may be a
    # table that lists only some floors, e.g. a parking check).
    valid = {form: levels for form, levels in forms.items() if _contiguous_from_ground(levels)}
    if not valid:
        return None
    counts = {form: max(levels) + 1 for form, levels in valid.items()}
    if len(set(counts.values())) != 1:
        return None
    count = next(iter(counts.values()))
    used = tuple(sorted(counts))
    return FloorLevelEvidence(
        count=count,
        forms=used,
        note=(
            f"{count} floor level(s) named on the sheet (ground through "
            f"{'ground' if count == 1 else 'level ' + str(count - 1)}; stilt/terrace not counted), "
            f"agreed by: {', '.join(used)}."
        ),
    )


__all__ = ["FloorLevelEvidence", "floor_count_from_named_levels"]
