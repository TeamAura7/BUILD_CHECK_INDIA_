"""
Canonical field definitions and the on-disk truth format for the corpus.

Every quantity has ONE definition here, because the earlier ground-truth
files disagreed with each other about what the fields mean (a "coverage"
that held an area in m2, plot width/depth swapped between files, setback
roles rotated by a quarter turn). Definitions:

  plot.width / plot.depth      the plot's two side lengths, in metres. For a plot
                               that is NOT a rectangle (a skewed quadrilateral),
                               the LONGER side of each opposite pair (convention;
                               needs domain confirmation). Their
                               ORDER is not defined: on a sheet whose road is
                               on a side "width" is the horizontal extent, on
                               another it is the frontage. Scoring accepts
                               either assignment (see `AXIS_PAIRS`).
  building.width / .depth      the same, for the building footprint.
  building.footprint_area      ground-floor footprint, m2.
  building.floor_count         ground floor plus upper floors. A STILT level, a
                               terrace/roof and any basement are NOT counted (the
                               ruleset treats stilt separately from height). This
                               convention comes from the earlier audited ground
                               truth and needs domain-expert confirmation.
  road.width                   width of the road the plot fronts, m. Must be
                               null when the sheet names a road but prints no
                               width -- asserting one would be fabrication.
  setbacks.front               gap between building and the ROAD-facing plot
                               edge, m.
  setbacks.rear                gap to the edge opposite the road, m.
  setbacks.left / .right       seen by someone standing on the road looking
                               into the plot.
  coverage                     footprint / plot area x 100, in PERCENT.
  far                          floor area ratio (a ratio, not an area).
  building_use                 categorical, normalised (e.g. "residential").
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = 1

CANONICAL_FIELDS: dict[str, str] = {
    "plot.width": "m", "plot.depth": "m", "plot.area": "m2",
    "building.width": "m", "building.depth": "m", "building.footprint_area": "m2",
    "building.floor_count": "count",
    "road.width": "m",
    "setbacks.front": "m", "setbacks.rear": "m", "setbacks.left": "m", "setbacks.right": "m",
    "coverage": "%", "far": "ratio",
    "building_use": "category",
}

# Pairs whose axis assignment is not defined; scored as an unordered pair.
AXIS_PAIRS: tuple[tuple[str, str], ...] = (
    ("plot.width", "plot.depth"),
    ("building.width", "building.depth"),
)

# Strongest first. A truth value is only as good as its tier.
#   printed        a number printed on the sheet for exactly this quantity
#   derived        arithmetic on printed numbers (e.g. area = w x d)
#   inspected      read by eye from the drawing (no printed number)
#   legacy         migrated from an earlier hand-verified file, not re-checked
#   unverified     migrated from a source that itself called it insufficient
#   must_abstain   the truth is "cannot be determined from this document"
Verification = Literal["printed", "derived", "inspected", "legacy", "unverified", "must_abstain", "unannotated"]
VERIFICATION_ORDER: tuple[str, ...] = ("printed", "derived", "inspected", "legacy", "unverified")
SCORABLE_DEFAULT: tuple[str, ...] = ("printed", "derived", "inspected", "legacy")

Split = Literal["dev", "heldout"]


class TruthField(BaseModel):
    value: Optional[Union[float, int, str]] = None
    verification: Verification = "unannotated"
    evidence: str = ""

    @field_validator("value")
    @classmethod
    def _bool_is_not_a_number(cls, v):
        if isinstance(v, bool):
            raise ValueError("boolean is not a valid truth value")
        return v


class Annotation(BaseModel):
    annotator: str = ""
    annotated_on: str = ""
    human_verified: bool = False
    notes: str = ""


class PlanTruth(BaseModel):
    plan_id: str
    schema_version: int = SCHEMA_VERSION
    annotation: Annotation = Field(default_factory=Annotation)
    # Page side that faces the road, when it can be seen. Documentation for
    # reviewers; setback roles above are already road-relative.
    road_side: Optional[Literal["top", "right", "bottom", "left"]] = None
    fields: dict[str, TruthField] = Field(default_factory=dict)
    discrepancies: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    source: str = ""
    licence: str = ""
    redacted: bool = False


class PlanEntry(BaseModel):
    id: str
    split: Split
    pdf: Optional[str] = None
    dxf: Optional[str] = None
    sha256: dict[str, str] = Field(default_factory=dict)
    truth: str
    provenance: Provenance = Field(default_factory=Provenance)
    tags: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    plans: list[PlanEntry] = Field(default_factory=list)
