"""
Cross-region numeric consistency guard for Vision output.

A VLM asked to describe several distinct regions on one architectural sheet
(SITE_PLAN, GROUND_FLOOR_PLAN, FIRST_FLOOR_PLAN, ...) sometimes pattern-
completes rather than reading each region's own pixels: it repeats a value
it already emitted for one region as the answer for a DIFFERENT semantic
field in a DIFFERENT, unrelated region. Confirmed live on
`data/test_plans/PLAN5.pdf`: Vision's `SITE_PLAN` region reported
`PLOT_WIDTH=17.59` / `PLOT_DEPTH=9.14`; its `GROUND_FLOOR_PLAN` region then
reported the exact same two numbers, byte-identical, as
`BUILDING_WIDTH`/`BUILDING_DEPTH` -- directly violating the extraction
prompt's own instruction (`ARCHITECTURAL_PLAN_PROMPT` rule #9: "Read each
region's dimensions independently... Never reuse or infer a width/depth
value from a different region").

This is a plausibility signal, not a certainty -- a plot and its ground
floor plan CAN legitimately share a dimension by coincidence. So a flagged
duplicate is downgraded (confidence reduced, evidence annotated) rather
than deleted outright, and `final_fusion`'s candidate selection is made to
prefer an unflagged reading of the same field over a flagged one when both
exist, rather than picking arbitrarily between two internally-disagreeing
Vision readings by tie-break order alone. Independent CV corroboration
downstream can still promote a flagged value back to a trusted AGREED
result; nothing here talks to CV.
"""

from __future__ import annotations

from typing import Optional

from backend.schemas.vision import VisionDocumentResult, VisionPageResult

# How close two values must be to count as "the same number", in absolute
# terms. Printed dimensions are read to 2 decimal places, so exact equality
# would miss e.g. a 17.590 vs 17.59 unit-conversion rounding difference.
_DUPLICATE_VALUE_TOLERANCE = 0.005

# Multiplicative confidence penalty applied to a flagged duplicate. Not
# zeroed out -- a coincidental match is possible, and a low-but-nonzero
# confidence still lets independent CV corroboration promote it back to a
# usable value.
_DUPLICATE_CONFIDENCE_PENALTY = 0.5

#: Substring marker used to detect a flagged dimension/area's evidence from
#: outside this module (see `final_fusion._is_flagged_duplicate`), so the
#: candidate-selection sort can deprioritize it without re-running the
#: cross-region comparison itself.
DUPLICATE_MARKER = "[CROSS-REGION DUPLICATE]"


def _region_type_by_id(page: VisionPageResult) -> dict[str, str]:
    return {r.id: (r.type or "").upper() for r in page.regions if r.id}


def _flag_pair(a, b, a_region: str, b_region: str) -> str:
    note = (
        f"{a.type}={a.value} in region {a_region} ({a.region_id}) is byte-identical to "
        f"{b.type}={b.value} in unrelated region {b_region} ({b.region_id}) -- likely "
        "cross-region pattern completion rather than an independent per-region read."
    )
    for item in (a, b):
        item.confidence = min(item.confidence, item.confidence * _DUPLICATE_CONFIDENCE_PENALTY)
        marker = f"{DUPLICATE_MARKER} {note}"
        item.evidence = f"{item.evidence}; {marker}" if item.evidence else marker
    return note


def flag_cross_region_duplicates(page: VisionPageResult) -> list[str]:
    """Downgrade dimensions/areas that duplicate a DIFFERENT semantic-type value from a DIFFERENT region.

    Mutates `page.dimensions`/`page.areas` in place (confidence + evidence
    annotation) and returns warning strings describing what was flagged.
    """
    region_type = _region_type_by_id(page)
    notes: list[str] = []

    for items in (page.dimensions, page.areas):
        present = [d for d in items if d.value is not None]
        for i, a in enumerate(present):
            a_region = region_type.get(a.region_id or "", "")
            for b in present[i + 1:]:
                b_region = region_type.get(b.region_id or "", "")
                if not a_region or not b_region or a_region == b_region:
                    continue  # same region, or region unknown: not the pattern-completion signature
                if a.type.upper() == b.type.upper():
                    continue  # two regions legitimately sharing a semantic type (e.g. two floors with the same room size) is not suspicious
                if abs(a.value - b.value) > _DUPLICATE_VALUE_TOLERANCE:
                    continue
                notes.append(_flag_pair(a, b, a_region, b_region))
    return notes


def sanitize_vision_result(vision: Optional[VisionDocumentResult]) -> Optional[VisionDocumentResult]:
    """Apply the cross-region duplicate check to every page of a Vision result. Returns the same object (mutated)."""
    if vision is None:
        return None
    for page in vision.pages:
        notes = flag_cross_region_duplicates(page)
        if notes:
            page.warnings.extend(f"[CROSS_REGION_CONSISTENCY] {n}" for n in notes)
    return vision


def is_flagged_duplicate(evidence: Optional[str]) -> bool:
    return bool(evidence) and DUPLICATE_MARKER in evidence


__all__ = ["flag_cross_region_duplicates", "sanitize_vision_result", "is_flagged_duplicate", "DUPLICATE_MARKER"]
