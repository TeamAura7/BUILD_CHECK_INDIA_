"""
Shared enums used across the entire BUILDCheck India pipeline.

These enums are the vocabulary every teammate's module must speak.
Do NOT redefine equivalent enums locally in cv_extraction / rag / rasE /
runtime_rules / compliance — import from here.
"""

from enum import Enum


class ConfidenceLevel(str, Enum):
    """How much we trust a single extracted/derived value."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CONFLICTING = "CONFLICTING"
    MISSING = "MISSING"


# Thresholds for mapping a raw 0-1 Vision confidence score onto
# ConfidenceLevel. Named and shared so every call site uses the same
# three-way cutoff instead of re-deriving its own binary one -- a binary
# "HIGH if conf >= .85 else MEDIUM" mapping (found duplicated at several call
# sites) has no LOW branch, so an unset confidence (schema default 0.0)
# reports as MEDIUM and silently bypasses the compliance engine's
# LOW -> REQUIRES_REVIEW safety gate.
VISION_CONFIDENCE_HIGH_THRESHOLD = 0.85
VISION_CONFIDENCE_LOW_THRESHOLD = 0.75


def confidence_level_from_score(confidence: float) -> ConfidenceLevel:
    """Map a raw Vision confidence score to a three-way ConfidenceLevel.

    confidence <= VISION_CONFIDENCE_LOW_THRESHOLD (including an
    unset/default 0.0) maps to LOW, not MEDIUM.
    """
    if confidence >= VISION_CONFIDENCE_HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    if confidence > VISION_CONFIDENCE_LOW_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def confidence_from_source(confidence: "float | None", reason: str) -> "Confidence":
    """The one sanctioned way to build a `Confidence` from a raw source score.

    This exists because "a value with no real confidence score silently
    becomes MEDIUM" is a bug pattern that has recurred at least three times
    in this codebase (see ARCHITECTURE_V2.md, Deliverable B.5/B.9 and D.5):
    once in `final_fusion._value_field`'s CV-only/Vision-only branches, once
    in `pipeline.build_normalized_plan`'s `fv()` closure, and it is still
    live in `final_fusion.apply_final_agreement_to_plan`'s `vf()` closure.
    Each was fixed independently by remembering to route through
    `confidence_level_from_score` instead of a literal `ConfidenceLevel.MEDIUM`
    fallback -- which is exactly how it kept coming back. Routing every new
    call site through this single function, instead, makes "MEDIUM with no
    score" impossible to construct by accident: `confidence is None` always
    maps to LOW (an explicit "no source score available" state), never MEDIUM.
    """
    # Imported lazily to avoid a circular import: `evidence.py` imports
    # `ConfidenceLevel` from this module, so this module cannot import
    # `Confidence` from `evidence.py` at module load time.
    from backend.schemas.evidence import Confidence

    if confidence is None:
        return Confidence(level=ConfidenceLevel.LOW, score=None, reason=f"{reason} (no source score)")
    return Confidence(level=confidence_level_from_score(confidence), score=confidence, reason=reason)


# Severity ordering for capping, not comparison in the mathematical sense --
# CONFLICTING/MISSING are deliberately excluded (see `cap_confidence_level`).
_CONFIDENCE_LEVEL_SEVERITY = {ConfidenceLevel.LOW: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.HIGH: 2}


def cap_confidence_level(level: ConfidenceLevel, cap: ConfidenceLevel) -> ConfidenceLevel:
    """Return whichever of `level`/`cap` is LOWER severity, among HIGH/MEDIUM/LOW.

    Exists to fix a specific, confirmed bug class: an upstream stage (e.g.
    `plot_resolution.plot_confidence`, which margin-scores how much a
    candidate-selection step itself trusts its own winner) can correctly
    compute a LOW/MEDIUM judgment that a downstream field-construction step
    (e.g. `evidence_reconciliation.to_value_field`, which only asks "do
    these numbers happen to agree with each other") never sees or applies --
    letting a field derived from a weak, low-confidence candidate ship at
    HIGH purely because two readings of that SAME weak candidate agree with
    themselves. See ARCHITECTURE_V2.md's Implementation log for the real,
    live instance this was found fixing (PLAN7: a photograph with no real
    site plan produced a spurious geometric "candidate" whose own front/
    rear edges trivially agreed with each other, shipping plot.width at
    ConfidenceLevel.HIGH).

    `CONFLICTING`/`MISSING` are passed through unchanged regardless of
    `cap` -- both are already a stronger "do not trust this" signal than
    any HIGH/MEDIUM/LOW cap could add, and reinterpreting them numerically
    against a HIGH/MEDIUM/LOW scale would not be meaningful.
    """
    if level not in _CONFIDENCE_LEVEL_SEVERITY or cap not in _CONFIDENCE_LEVEL_SEVERITY:
        return level
    return level if _CONFIDENCE_LEVEL_SEVERITY[level] <= _CONFIDENCE_LEVEL_SEVERITY[cap] else cap


class ComplianceStatus(str, Enum):
    """
    Outcome of checking ONE rule against the normalized plan.

    Uncertainty must never be silently collapsed into PASS/FAIL.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class SourceType(str, Enum):
    """Where a piece of evidence originated in the source document."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    DIMENSION_LINE = "DIMENSION_LINE"
    VECTOR_GEOMETRY = "VECTOR_GEOMETRY"
    RASTER_GEOMETRY = "RASTER_GEOMETRY"
    OCR = "OCR"
    ANNOTATION = "ANNOTATION"
    DERIVED = "DERIVED"  # computed from other fields, not read directly


class DocumentType(str, Enum):
    """Kind of source document a plan was extracted from."""

    VECTOR_PDF = "VECTOR_PDF"
    RASTER_PDF = "RASTER_PDF"
    SCANNED_IMAGE = "SCANNED_IMAGE"
    DXF = "DXF"          # future — interface only, not implemented in Phase 1
    BIM = "BIM"           # future — interface only, not implemented in Phase 1


class EntityKind(str, Enum):
    """High-level category of a geometric candidate on a plan."""

    PLOT = "PLOT"
    BUILDING = "BUILDING"
    ROAD = "ROAD"
    SETBACK = "SETBACK"
    DIMENSION = "DIMENSION"
    ANNOTATION = "ANNOTATION"
    # A title block / sheet border / drawing-management rectangle -- a
    # drawing-management concept, not a site-plan entity, and distinct from
    # UNKNOWN (which means "not yet classified", not "confidently not a
    # site-plan entity"). See ARCHITECTURE_V2.md Deliverable B.9/D.3.
    SHEET_FRAME = "SHEET_FRAME"
    UNKNOWN = "UNKNOWN"


class SpatialRelationType(str, Enum):
    """Qualitative spatial relation between two geometric entities."""

    ADJACENT = "ADJACENT"
    OVERLAPPING = "OVERLAPPING"
    CONTAINS = "CONTAINS"
    CONTAINED_BY = "CONTAINED_BY"
    DISJOINT = "DISJOINT"
    FACING = "FACING"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Drawing Evidence Graph / Structural Hypothesis / Evidence Decision vocabulary
# (Architecture V2, Phase 2 — see ARCHITECTURE_V2.md, Deliverable D).
# Additive: nothing below removes or renames an existing member.
# ---------------------------------------------------------------------------


class EvidenceKind(str, Enum):
    """What kind of source observation an `EvidenceCandidate` represents.

    This is deliberately a different, finer-grained vocabulary than
    `SourceType` above: `SourceType` describes *where in the document* a
    piece of text/geometry evidence was anchored (used by `TextEvidence`/
    `GeometryEvidence`, already attached to an *accepted* `ValueField`).
    `EvidenceKind` describes what a *not-yet-decided* `EvidenceCandidate` is,
    before any acceptance decision has been made.
    """

    GEOMETRY = "GEOMETRY"                    # a polygon/segment/edge as drawn
    OCR_TEXT = "OCR_TEXT"                     # recognized raster/rendered text
    NATIVE_TEXT = "NATIVE_TEXT"               # PDF native text / DXF TEXT-MTEXT
    VISION_SEMANTIC = "VISION_SEMANTIC"       # a VLM-derived tag or value
    DOCUMENT_STATEMENT = "DOCUMENT_STATEMENT"  # e.g. an "AREA STATEMENT" table row
    RECONSTRUCTED_GEOMETRY = "RECONSTRUCTED_GEOMETRY"  # fragment-fitted boundary


class RelationType(str, Enum):
    """Typed edge in the Drawing Evidence Graph.

    NEAR/COLLINEAR/PARALLEL/PERPENDICULAR/INTERSECTS/CONTINUES/ENCLOSES/
    INSIDE/ADJACENT/ALIGNED_WITH are deterministic geometric relations,
    computable directly from geometry with no learned component. MEASURES/
    BELONGS_TO are association relations produced by scoring logic (e.g. a
    dimension label associated with the edge it measures). CONFLICTS_WITH
    records a detected disagreement between two nodes/candidates.
    """

    NEAR = "NEAR"
    COLLINEAR = "COLLINEAR"
    PARALLEL = "PARALLEL"
    PERPENDICULAR = "PERPENDICULAR"
    INTERSECTS = "INTERSECTS"
    CONTINUES = "CONTINUES"
    ENCLOSES = "ENCLOSES"
    INSIDE = "INSIDE"
    ADJACENT = "ADJACENT"
    ALIGNED_WITH = "ALIGNED_WITH"
    MEASURES = "MEASURES"
    BELONGS_TO = "BELONGS_TO"
    CONFLICTS_WITH = "CONFLICTS_WITH"


class HypothesisIdentity(str, Enum):
    """What a `StructuralHypothesis` claims to be.

    Widens `EntityKind` with `SHEET_FRAME` (a hypothesis can now claim to
    be, and be scored/rejected as, sheet furniture rather than a real
    site-plan entity) rather than inventing a parallel vocabulary; see
    ARCHITECTURE_V2.md Deliverable B.9/D.3 for why `EntityKind` was chosen
    as the base instead of a new enum from scratch.
    """

    PLOT_BOUNDARY = "PLOT_BOUNDARY"
    BUILDING = "BUILDING"
    ROAD = "ROAD"
    SHEET_FRAME = "SHEET_FRAME"
    OTHER = "OTHER"

    @property
    def entity_kind(self) -> "EntityKind":
        """The corresponding `EntityKind`, for interop with existing candidate schemas."""
        return {
            HypothesisIdentity.PLOT_BOUNDARY: EntityKind.PLOT,
            HypothesisIdentity.BUILDING: EntityKind.BUILDING,
            HypothesisIdentity.ROAD: EntityKind.ROAD,
            HypothesisIdentity.SHEET_FRAME: EntityKind.SHEET_FRAME,
            HypothesisIdentity.OTHER: EntityKind.UNKNOWN,
        }[self]


class DecisionStatus(str, Enum):
    """Outcome of the evidence decision engine for one field.

    Never a residual "else" branch: every field-level decision is exactly
    one of these four, always with a `reason`/provenance trail (see
    `Decision` in `backend/schemas/decision.py`). Deliberately mirrors
    `document_evidence.py`'s existing (currently unwired) vocabulary rather
    than inventing a new one.
    """

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CONFLICT = "CONFLICT"
    ABSTAIN = "ABSTAIN"
