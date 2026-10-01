"""Document Evidence Layer (Phase 3 New).

Implements the design in PHASEE3NEW.md sections 12-22:

    "The PDF is the source of truth."

CV and Vision are independent *observers*. This module is the third,
deterministic layer that both of them feed into: it asks "which
candidate value is actually supported by the document itself?" instead
of "which model do we trust more?".

It never talks to Vision or CV extraction code directly -- it only
consumes the *evidence already produced* by each independent pipeline
(`IndependentMeasurement` objects from CV, plus the raw PDF text/OCR
layer) and returns a deterministic verdict.

Nothing here is PLAN-specific: every function operates purely on the
numeric candidate values, their provenance metadata, and a full-document
text index built once per PDF.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

from backend.schemas.independent_measurements import IndependentMeasurement
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Status vocabulary (PHASEE3NEW.md section 12)
# ---------------------------------------------------------------------------

FOUND = "FOUND"
NOT_FOUND_BY_CV = "NOT_FOUND_BY_CV"
NOT_FOUND_BY_VISION = "NOT_FOUND_BY_VISION"
NOT_FOUND_BY_EITHER = "NOT_FOUND_BY_EITHER"
DOCUMENT_EVIDENCE_FOUND = "DOCUMENT_EVIDENCE_FOUND"
DOCUMENT_EVIDENCE_NOT_FOUND = "DOCUMENT_EVIDENCE_NOT_FOUND"
CONFLICT = "CONFLICT"
AGREED = "AGREED"
UNRESOLVED = "UNRESOLVED"

AGREED_DOCUMENT_VERIFIED = "AGREED_DOCUMENT_VERIFIED"
AGREED_UNVERIFIED = "AGREED_UNVERIFIED"
CV_ONLY_DOCUMENT_VERIFIED = "CV_ONLY_DOCUMENT_VERIFIED"
VISION_ONLY_DOCUMENT_VERIFIED = "VISION_ONLY_DOCUMENT_VERIFIED"
CV_ONLY_UNVERIFIED = "CV_ONLY_UNVERIFIED"
VISION_ONLY_UNVERIFIED = "VISION_ONLY_UNVERIFIED"
CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE = "CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE"
UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"

# ---------------------------------------------------------------------------
# Evidence scoring (PHASEE3NEW.md section 18)
# ---------------------------------------------------------------------------

# Configurable, documented weights -- never hardcoded silently.
EVIDENCE_SCORE_WEIGHTS = {
    "exact_printed_numeric_value": 5,
    "dimension_line_association": 4,
    "semantic_association": 3,
    "ocr_confirmation": 3,
    "geometry_confirmation": 3,
    "repeated_occurrence": 2,
    "cv_extraction": 1,
    "vision_extraction": 1,
}

# A candidate becomes eligible to win a conflict only when its score
# clears this threshold *and* it has at least one direct document
# evidence source (native text, OCR confirmation, or geometry
# confirmation) -- see PHASEE3NEW.md section 18.
DEFAULT_SCORE_THRESHOLD = 6

# Relative + absolute tolerance used when matching a candidate value
# against document text / other measurements of the same field.
DEFAULT_ABS_TOL = 0.05
DEFAULT_REL_TOL = 0.01


def _close(a: float, b: float, abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


# ---------------------------------------------------------------------------
# Document text index
# ---------------------------------------------------------------------------


@dataclass
class DocumentTextIndex:
    """Full native-text index of a PDF, built once and reused for every
    field's document-evidence lookup (targeted re-check, occurrence
    counting). Independent of, and built without reference to, any CV or
    Vision *semantic* conclusions -- it is just the document's raw text.
    """

    pages: list[str] = field(default_factory=list)
    available: bool = True

    def occurrences(self, value: float, abs_tol: float = 0.01) -> int:
        """Count how many times a numeric value appears (as printed text)
        anywhere in the document, tolerant of formatting variants
        (12.19 / 12.190 / 12,19 / "12.19m" / "40'0\"" style annotations are
        not attempted here -- this only matches decimal numerals, which is
        what native PDF text layers and OCR both normally produce).
        """
        if not self.pages:
            return 0
        count = 0
        pattern = re.compile(r"\d+\.\d+|\d+")
        for text in self.pages:
            for m in pattern.finditer(text):
                try:
                    v = float(m.group())
                except ValueError:
                    continue
                if _close(v, value, abs_tol=abs_tol, rel_tol=0.0):
                    count += 1
        return count


def build_document_text_index(pdf_path: Optional[str]) -> DocumentTextIndex:
    """Build a full-document text index for targeted document re-checks:
    native PDF text PLUS a best-effort OCR pass per page.

    Native text alone is not enough on plans like the common "print to
    PDF from CAD" style, where the whole drawing (including every
    dimension label) is a single rasterized image and `page.get_text()`
    returns nothing or only incidental vector-text fragments (e.g. a
    schedule-of-openings table done as real text while the site plan
    itself is a picture). On such a page this index was previously blind
    to every genuine dimension value -- meaning it could never confirm a
    correct Vision reading OR catch a hallucinated one; every field came
    back UNVERIFIED regardless of correctness. Worse, a small amount of
    unrelated native text (a schedule entry) could still outscore a
    correct-but-unconfirmable OCR-only value in `evaluate_candidate`,
    since native-text presence is worth real points and OCR-invisible
    correctness is worth none. Adding OCR text closes that gap: the
    document's real printed values become checkable even when they only
    exist as pixels, which is what "the PDF is the source of truth" (see
    module docstring) actually requires for this class of document.

    Falls back to an empty (unavailable) index if the PDF cannot be
    opened or PyMuPDF is unavailable -- callers must treat
    `available=False` as "cannot confirm", never as "value absent". OCR
    failures degrade to native-text-only (never to fully unavailable) --
    callers already correctly treat a small text index as "less able to
    confirm", not as "confirmed absent", so partial coverage is safe.
    """
    if not pdf_path:
        return DocumentTextIndex(pages=[], available=False)
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.warning("document_evidence: PyMuPDF unavailable; document text index disabled.")
        return DocumentTextIndex(pages=[], available=False)
    try:
        doc = fitz.open(pdf_path)
        pages = [p.get_text("text") or "" for p in doc]
        try:
            from backend.cv_extraction import ocr_fallback

            for i, page in enumerate(doc):
                try:
                    image = ocr_fallback.rasterize_page(page, dpi=ocr_fallback.DEFAULT_OCR_DPI)
                    ocr_items = ocr_fallback.ocr_page(image, i, dpi=ocr_fallback.DEFAULT_OCR_DPI)
                    # Word-level items only -- line-group items (see
                    # `RawTextItem.is_line_group`) repeat the same numbers
                    # already covered by their constituent words and would
                    # only inflate `occurrences()` counts without adding
                    # any new value.
                    ocr_text = " ".join(it.text for it in ocr_items if not getattr(it, "is_line_group", False))
                    if ocr_text:
                        pages[i] = f"{pages[i]} {ocr_text}"
                except Exception as page_exc:
                    logger.warning("document_evidence: OCR failed for page %d: %s", i, page_exc)
        except Exception as exc:
            logger.warning("document_evidence: OCR stack unavailable, text index is native-text-only: %s", exc)
        doc.close()
        return DocumentTextIndex(pages=pages, available=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("document_evidence: failed to build text index: %s", exc)
        return DocumentTextIndex(pages=[], available=False)


# ---------------------------------------------------------------------------
# Per-candidate document evidence
# ---------------------------------------------------------------------------


@dataclass
class CandidateEvidence:
    value: float
    raw_value: Optional[str] = None
    native_text_confirmed: bool = False
    ocr_confirmed: bool = False
    geometry_supported: bool = False
    dimension_line_associated: bool = False
    semantic_associated: bool = False
    occurrence_count: int = 0
    cv_present: bool = False
    vision_present: bool = False
    sources: list[str] = field(default_factory=list)
    score: int = 0

    def has_direct_document_evidence(self) -> bool:
        return self.native_text_confirmed or self.ocr_confirmed or self.geometry_supported

    def to_dict(self) -> dict:
        return {
            "raw_value": self.raw_value,
            "native_text_confirmed": self.native_text_confirmed,
            "ocr_confirmed": self.ocr_confirmed,
            "geometry_supported": self.geometry_supported,
            "dimension_line_associated": self.dimension_line_associated,
            "semantic_associated": self.semantic_associated,
            "occurrence_count": self.occurrence_count,
            "sources": sorted(set(self.sources)),
            "score": self.score,
        }


def _cv_source_flags(measurements: Iterable[IndependentMeasurement], value: float) -> CandidateEvidence:
    ev = CandidateEvidence(value=value)
    for m in measurements:
        mv = m.value_m if m.value_m is not None else m.value
        if mv is None or not _close(mv, value):
            continue
        ev.cv_present = True
        if ev.raw_value is None and m.evidence:
            ev.raw_value = m.evidence[0]
        if m.source == "NATIVE_TEXT":
            ev.native_text_confirmed = True
            ev.sources.append("native_pdf_text")
        elif m.source == "OCR":
            ev.ocr_confirmed = True
            ev.sources.append("ocr")
        elif m.source == "VECTOR_GEOMETRY":
            ev.geometry_supported = True
            ev.sources.append("vector_geometry")
        elif m.source == "DERIVED":
            ev.sources.append("derived_geometry")
        if m.geometry_bbox_pts:
            ev.dimension_line_associated = True
        if m.field:
            ev.semantic_associated = True
    return ev


def evaluate_candidate(
    value: float,
    field_measurements: Iterable[IndependentMeasurement],
    text_index: DocumentTextIndex,
    vision_present: bool,
) -> CandidateEvidence:
    """Gather all document evidence supporting a single candidate value
    for one field, and compute its deterministic evidence score
    (PHASEE3NEW.md section 18).
    """
    ev = _cv_source_flags(field_measurements, value)
    ev.vision_present = vision_present
    if text_index.available:
        ev.occurrence_count = text_index.occurrences(value)
        if ev.occurrence_count > 0:
            ev.native_text_confirmed = ev.native_text_confirmed or True
            ev.sources.append("document_text_occurrence")

    score = 0
    if ev.native_text_confirmed:
        score += EVIDENCE_SCORE_WEIGHTS["exact_printed_numeric_value"]
    if ev.dimension_line_associated:
        score += EVIDENCE_SCORE_WEIGHTS["dimension_line_association"]
    if ev.semantic_associated:
        score += EVIDENCE_SCORE_WEIGHTS["semantic_association"]
    if ev.ocr_confirmed:
        score += EVIDENCE_SCORE_WEIGHTS["ocr_confirmation"]
    if ev.geometry_supported:
        score += EVIDENCE_SCORE_WEIGHTS["geometry_confirmation"]
    if ev.occurrence_count > 1:
        score += EVIDENCE_SCORE_WEIGHTS["repeated_occurrence"]
    if ev.cv_present:
        score += EVIDENCE_SCORE_WEIGHTS["cv_extraction"]
    if ev.vision_present:
        score += EVIDENCE_SCORE_WEIGHTS["vision_extraction"]
    ev.score = score
    return ev


# ---------------------------------------------------------------------------
# Top level decision helpers (sections 14-17, 22)
# ---------------------------------------------------------------------------


@dataclass
class FieldVerdict:
    field: str
    final_value: Optional[float]
    unit: Optional[str]
    status: str
    winner: Optional[str]
    reason: str
    cv_evidence: Optional[CandidateEvidence]
    vision_evidence: Optional[CandidateEvidence]


def resolve_field(
    field_name: str,
    cv_value: Optional[float],
    vision_value: Optional[float],
    unit: Optional[str],
    field_measurements: Iterable[IndependentMeasurement],
    text_index: DocumentTextIndex,
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
) -> FieldVerdict:
    """Deterministic conflict-resolution / verification decision for one
    field. Pure function of the evidence already collected -- no model
    confidence scores are used to break ties (PHASEE3NEW.md section 15/32).
    """
    field_measurements = list(field_measurements)

    if cv_value is None and vision_value is None:
        return FieldVerdict(field_name, None, unit, NOT_FOUND_BY_EITHER, None,
                             "Neither CV nor Vision produced a candidate for this field.", None, None)

    # --- Agreement -----------------------------------------------------
    if cv_value is not None and vision_value is not None and _close(cv_value, vision_value):
        cv_ev = evaluate_candidate(cv_value, field_measurements, text_index, vision_present=True)
        if cv_ev.has_direct_document_evidence():
            return FieldVerdict(field_name, round((cv_value + vision_value) / 2, 4), unit,
                                 AGREED_DOCUMENT_VERIFIED, "cv+vision",
                                 "CV and Vision independently agree and the document (native text/OCR/geometry) "
                                 "confirms the value.", cv_ev, cv_ev)
        return FieldVerdict(field_name, round((cv_value + vision_value) / 2, 4), unit,
                             AGREED_UNVERIFIED, "cv+vision",
                             "CV and Vision independently agree, but no direct document evidence (native "
                             "text/OCR/geometry) could confirm it -- treat with caution.", cv_ev, cv_ev)

    # --- CV only ---------------------------------------------------------
    if cv_value is not None and vision_value is None:
        cv_ev = evaluate_candidate(cv_value, field_measurements, text_index, vision_present=False)
        status = CV_ONLY_DOCUMENT_VERIFIED if cv_ev.has_direct_document_evidence() else CV_ONLY_UNVERIFIED
        reason = ("Only CV produced a candidate; the document independently confirms it."
                  if status == CV_ONLY_DOCUMENT_VERIFIED else
                  "Only CV produced a candidate and the document could not independently confirm it "
                  "(Vision did not fail this field's existence -- it simply did not emit a value; "
                  "status is NOT_FOUND_BY_VISION for that side).")
        return FieldVerdict(field_name, cv_value, unit, status, "cv", reason, cv_ev, None)

    # --- Vision only -----------------------------------------------------
    if vision_value is not None and cv_value is None:
        vision_ev = evaluate_candidate(vision_value, field_measurements, text_index, vision_present=True)
        status = VISION_ONLY_DOCUMENT_VERIFIED if vision_ev.has_direct_document_evidence() else VISION_ONLY_UNVERIFIED
        reason = ("Only Vision produced a candidate; the document independently confirms it "
                  "(matched in native text/OCR or CV geometry)."
                  if status == VISION_ONLY_DOCUMENT_VERIFIED else
                  "Only Vision produced a candidate and the document could not independently confirm it "
                  "(CV did not fail this field's existence -- it simply did not emit a value; "
                  "status is NOT_FOUND_BY_CV for that side).")
        return FieldVerdict(field_name, vision_value, unit, status, "vision", reason, None, vision_ev)

    # --- Conflict (section 15-17) -----------------------------------------
    assert cv_value is not None and vision_value is not None
    cv_ev = evaluate_candidate(cv_value, field_measurements, text_index, vision_present=False)
    vision_ev = evaluate_candidate(vision_value, field_measurements, text_index, vision_present=True)

    cv_wins = cv_ev.score >= score_threshold and cv_ev.has_direct_document_evidence()
    vision_wins = vision_ev.score >= score_threshold and vision_ev.has_direct_document_evidence()

    if cv_wins and not vision_wins:
        return FieldVerdict(field_name, cv_value, unit, CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE, "cv",
                             f"CV={cv_value} and Vision={vision_value} disagreed; document evidence "
                             f"(score {cv_ev.score} vs {vision_ev.score}) supports CV.", cv_ev, vision_ev)
    if vision_wins and not cv_wins:
        return FieldVerdict(field_name, vision_value, unit, CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE, "vision",
                             f"CV={cv_value} and Vision={vision_value} disagreed; document evidence "
                             f"(score {vision_ev.score} vs {cv_ev.score}) supports Vision.", cv_ev, vision_ev)
    if cv_wins and vision_wins:
        # Both have strong, independent document evidence -- section 17.
        if cv_ev.score != vision_ev.score:
            winner, wv, we, le = (("cv", cv_value, cv_ev, vision_ev) if cv_ev.score > vision_ev.score
                                   else ("vision", vision_value, vision_ev, cv_ev))
            return FieldVerdict(field_name, wv, unit, CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE, winner,
                                 f"CV={cv_value} and Vision={vision_value} both had direct document evidence; "
                                 f"{winner} scored higher ({we.score} vs {le.score}).", cv_ev, vision_ev)
        return FieldVerdict(field_name, None, unit, UNRESOLVED_CONFLICT, None,
                             f"CV={cv_value} and Vision={vision_value} both have equally strong direct "
                             "document evidence; refusing to arbitrarily pick one. Needs manual review or "
                             "higher-resolution targeted re-check.", cv_ev, vision_ev)

    # Neither candidate clears the threshold with direct document evidence.
    return FieldVerdict(field_name, None, unit, UNRESOLVED_CONFLICT, None,
                         f"CV={cv_value} and Vision={vision_value} disagreed and neither is clearly "
                         "supported by direct document evidence (native text/OCR/geometry).", cv_ev, vision_ev)


@dataclass
class ReportedVsCalculated:
    reported_value: Optional[float]
    calculated_value: Optional[float]
    final_value: Optional[float]
    status: str


def reported_vs_calculated(
    field_measurements: Iterable[IndependentMeasurement],
    tolerance_abs: float = 0.05,
    tolerance_rel: float = 0.001,
) -> Optional[ReportedVsCalculated]:
    """PHASEE3NEW.md section 19: never silently overwrite an explicitly
    reported document value with a derived/calculated one. `reported`
    means the value came straight from native text (an explicit area
    statement in the PDF); `calculated` means it was derived from
    geometry (DERIVED source, e.g. width * depth).
    """
    reported = None
    calculated = None
    for m in field_measurements:
        v = m.value_m if m.value_m is not None else m.value
        if v is None:
            continue
        if m.source == "NATIVE_TEXT" and reported is None:
            reported = v
        elif m.source == "DERIVED" and calculated is None:
            calculated = v
    if reported is None and calculated is None:
        return None
    if reported is not None and calculated is not None:
        if _close(reported, calculated, abs_tol=tolerance_abs, rel_tol=tolerance_rel):
            return ReportedVsCalculated(reported, calculated, reported, "REPORTED_VALUE_WITH_GEOMETRIC_CHECK")
        return ReportedVsCalculated(reported, calculated, reported, "REPORTED_CALCULATION_DIFFER")
    if reported is not None:
        return ReportedVsCalculated(reported, None, reported, "REPORTED_VALUE_ONLY")
    return ReportedVsCalculated(None, calculated, calculated, "CALCULATED_VALUE_ONLY")


__all__ = [
    "FOUND", "NOT_FOUND_BY_CV", "NOT_FOUND_BY_VISION", "NOT_FOUND_BY_EITHER",
    "DOCUMENT_EVIDENCE_FOUND", "DOCUMENT_EVIDENCE_NOT_FOUND", "CONFLICT", "AGREED", "UNRESOLVED",
    "AGREED_DOCUMENT_VERIFIED", "AGREED_UNVERIFIED", "CV_ONLY_DOCUMENT_VERIFIED",
    "VISION_ONLY_DOCUMENT_VERIFIED", "CV_ONLY_UNVERIFIED", "VISION_ONLY_UNVERIFIED",
    "CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE", "UNRESOLVED_CONFLICT",
    "EVIDENCE_SCORE_WEIGHTS", "DEFAULT_SCORE_THRESHOLD",
    "DocumentTextIndex", "build_document_text_index",
    "CandidateEvidence", "evaluate_candidate",
    "FieldVerdict", "resolve_field",
    "ReportedVsCalculated", "reported_vs_calculated",
]
