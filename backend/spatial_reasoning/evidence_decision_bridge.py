"""
Architecture V2, Phase 8 — feature-flagged bridge from `pipeline.py` into
`evidence_decision.py`.

See ARCHITECTURE_V2.md's phased order: "Phase 8: connect pipeline.py to
evidence_decision.py behind a feature flag (both paths computed; flag
picks which one ships)."

This module deliberately does not reimplement field coverage:
`LENGTH_FIELDS`/`NUMERIC_FIELDS`/`_cv_map`/`_vision_length`/`_vision_numeric`
are the exact same helpers `final_fusion.build_final_agreement` (the OLD
path) already uses, imported here rather than duplicated, so the two paths
are computed over an identical field set and are directly comparable
field-for-field.

Status: shadow/flag-gated only. `backend.config.Settings.
use_evidence_decision_engine` defaults to `False` — the OLD path
(`final_fusion.apply_final_agreement_to_plan`) remains what ships by
default. This module's output is ALWAYS computed and logged for
comparison regardless of the flag (see `pipeline.build_normalized_plan`'s
call site), so real discrepancies accumulate in the logs before Phase 9's
old/new comparison ever gates a default flip.

Known scope limitation of this phase: `text_index` is always built with
`pdf_path=None` (i.e. always `DocumentTextIndex(available=False)`) — no
`pdf_path` is threaded through `pipeline.build_normalized_plan` yet, so
`evidence_decision.decide_field`'s document-text-occurrence confirmation
(the `*_DOCUMENT_VERIFIED` statuses) can never fire through this bridge
today. The per-measurement evidence flags (`source`, `geometry_bbox_pts`,
`field`) that `document_evidence.evaluate_candidate` also reads directly
from each `IndependentMeasurement` still apply and still meaningfully
differ from the OLD path's blind CV+Vision averaging. Threading a real
`pdf_path` through is a natural, low-risk follow-up once this wiring
itself is proven stable.
"""

from __future__ import annotations

import uuid
from typing import Optional

from backend.schemas.decision import Decision
from backend.schemas.enums import ConfidenceLevel, DecisionStatus, cap_confidence_level
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.independent_measurements import IndependentCVResult
from backend.schemas.normalized_plan import NormalizedPlan
from backend.schemas.vision import VisionDocumentResult
from backend.spatial_reasoning import document_evidence as doc_ev
from backend.spatial_reasoning.areas import coverage_field, far_field
from backend.spatial_reasoning.evidence_decision import decide_field
from backend.spatial_reasoning.final_fusion import (
    LENGTH_FIELDS,
    NUMERIC_FIELDS,
    _cv_map,
    _vision_length,
    _vision_numeric,
)
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

# Fields eligible for the legacy-fallback (Phase 9 Finding 1). Deliberately
# the same set `apply_evidence_decision_fields_to_plan` actually ships
# (`_SHIPPABLE_FIELD_PATHS`, defined below) minus "coverage"/"far", which get
# their own derive-before-fallback ordering in `_restore_derived_fields`
# rather than a direct legacy copy -- see that function's docstring.
_DERIVED_FIELDS = {"coverage", "far"}

# A legacy-resolved value is only trusted as a fallback when the OLD path's
# own existing confidence signal already marks it as more than a weak/lone
# geometric guess. This is the SAME `ConfidenceLevel` the OLD path itself
# already assigns (e.g. via `pipeline._cap_field_confidence`'s
# `plot_conf_level` cap, already keyed off `plot_resolution.
# _MIN_USABLE_PLOT_SCORE`) -- not a new, second-guessed threshold. This is
# precisely what keeps PLAN7 (a photograph with no real site-plan geometry,
# whose lone weak plot/building candidate is already capped to LOW by that
# existing mechanism) from regressing back into shipping a confident false
# positive: LOW-confidence legacy values are never eligible, on ANY plan,
# without any plan-specific check.
_LEGACY_FALLBACK_ELIGIBLE_LEVELS = (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)


def _new_decision_id(field_name: str, kind: str) -> str:
    return f"decision-{field_name}-{kind}-{uuid.uuid4().hex[:8]}"


def _legacy_field_map(plan: NormalizedPlan) -> dict[str, ValueField]:
    """The OLD path's own already-resolved value for every field this bridge
    covers, read from `plan` (pipeline.py's `old_plan` -- the plan AFTER
    `final_fusion.apply_final_agreement_to_plan`, so it already reflects
    whichever of {CV+Vision fusion, the legacy geometry/vision-semantic
    candidate pool, a printed Area Statement reading} the OLD path itself
    considered authoritative for that field). Shared by `log_comparison`
    (which only reads it for comparison) and `_apply_legacy_fallback`/
    `_restore_derived_fields` (which may use it as a lower-priority
    fallback), so both stay defined over the exact same field set."""
    return {
        "plot.width": plan.plot.width, "plot.depth": plan.plot.depth, "plot.area": plan.plot.area,
        "building.width": plan.building.width, "building.depth": plan.building.depth,
        "building.footprint_area": plan.building.footprint_area,
        "road.width": plan.road.width,
        "setbacks.front": plan.setbacks.front, "setbacks.rear": plan.setbacks.rear,
        "setbacks.left": plan.setbacks.left, "setbacks.right": plan.setbacks.right,
        "coverage": plan.coverage, "far": plan.far,
    }


def _apply_legacy_fallback(
    field_name: str,
    decision: Decision,
    value_field: ValueField,
    legacy_map: dict[str, ValueField],
) -> tuple[Decision, ValueField]:
    """Phase 9 Finding 1's fix: when `evidence_decision.decide_field` had
    NOTHING to work with for this field (ABSTAIN, `independent_cv` and
    Vision both empty for it -- see `compute_evidence_decision_fields`, the
    only caller, which only invokes this on that exact ABSTAIN case), fall
    back to the OLD path's own already-resolved value for the SAME field,
    but only when that legacy value is itself at least MEDIUM confidence
    (see `_LEGACY_FALLBACK_ELIGIBLE_LEVELS`'s docstring for why this is what
    keeps PLAN7 safe generically).

    This never runs when `independent_cv`/Vision DID produce a candidate --
    `decision.status != ABSTAIN` in every other case -- so the legacy value
    never competes with a real independent_cv/Vision candidate; it only
    fills a genuine gap. The shipped confidence is capped to at most MEDIUM
    regardless of what the legacy path itself assigned: it was never
    independently corroborated by this engine's own evidence-scoring, so it
    must not ship at the same HIGH tier as a value the engine actually
    verified.
    """
    if decision.status != DecisionStatus.ABSTAIN:
        return decision, value_field

    legacy_vf = legacy_map.get(field_name)
    if (
        legacy_vf is None
        or legacy_vf.value is None
        or legacy_vf.confidence.level not in _LEGACY_FALLBACK_ELIGIBLE_LEVELS
    ):
        return decision, value_field

    fallback_id = _new_decision_id(field_name, "legacy-fallback")
    capped_level = cap_confidence_level(legacy_vf.confidence.level, ConfidenceLevel.MEDIUM)
    fallback_value = value_field.model_copy(update={
        "value": legacy_vf.value,
        "raw_value": legacy_vf.raw_value,
        "normalized_value": legacy_vf.normalized_value,
        "confidence": Confidence(
            level=capped_level,
            score=legacy_vf.confidence.score,
            reason=(
                f"{legacy_vf.confidence.reason} (capped to {capped_level.value}: used as a "
                "lower-priority legacy fallback -- independent_cv and vision produced no "
                "candidate for this field, so the evidence decision engine could not "
                "independently corroborate it)."
            ),
        ),
        "source": f"legacy_fallback:{legacy_vf.source or 'unknown'}",
        "evidence": legacy_vf.evidence,
        "conflict": None,
        "decision_id": fallback_id,
    })
    fallback_decision = Decision(
        id=fallback_id, field_name=field_name, status=DecisionStatus.ACCEPT,
        accepted_candidate_id="legacy_resolved_plan",
        considered_candidate_ids=list(decision.considered_candidate_ids) + ["legacy_resolved_plan"],
        resulting_value_field=field_name,
        confidence_derivation=(
            f"independent_cv and vision produced no candidate for this field ({decision.confidence_derivation}); "
            f"used the OLD path's own already-resolved value as a lower-priority fallback "
            f"(legacy confidence: {legacy_vf.confidence.level.value}, capped to {capped_level.value})."
        ),
    )
    return fallback_decision, fallback_value


def _restore_derived_fields(
    results: dict[str, tuple[Decision, ValueField]], legacy_plan: NormalizedPlan,
) -> None:
    """Phase 9 Finding 1's "derived field restoration": `coverage`/`far` are
    *defined* as functions of `building.footprint_area`/`plot.area`
    (`areas.coverage_field`/`areas.far_field`, reused unchanged here, never
    reimplemented) -- neither CV nor Vision has to have printed an explicit
    coverage%/FAR figure for these to be computable once footprint/plot area
    are already resolved. Mutates `results` in place, and only when
    `decide_field` itself found no DIRECT coverage/FAR evidence (status
    ABSTAIN) -- a direct reading, or a genuine CONFLICT, is never overridden.

    Uses `results["building.footprint_area"]`/`results["plot.area"]` --
    i.e. THIS function's own already-resolved values (which may themselves
    already be `_apply_legacy_fallback` results) -- never `legacy_plan`'s
    stale pre-bridge numbers directly, so a derived coverage/FAR always
    reflects the new engine's own freshest inputs.
    """
    footprint_pair = results.get("building.footprint_area")
    plot_area_pair = results.get("plot.area")
    if footprint_pair is None or plot_area_pair is None:
        return
    footprint_decision, footprint_vf = footprint_pair
    plot_area_decision, plot_area_vf = plot_area_pair

    coverage_pair = results.get("coverage")
    if coverage_pair is not None and coverage_pair[0].status == DecisionStatus.ABSTAIN:
        computed = coverage_field(footprint_vf, plot_area_vf)
        if computed.value is not None:
            decision_id = _new_decision_id("coverage", "derived")
            results["coverage"] = (
                Decision(
                    id=decision_id, field_name="coverage", status=DecisionStatus.ACCEPT,
                    accepted_candidate_id="derived:coverage_field",
                    considered_candidate_ids=[footprint_decision.id, plot_area_decision.id],
                    resulting_value_field="coverage",
                    confidence_derivation=(
                        "No direct coverage evidence from independent_cv/vision; derived via "
                        "areas.coverage_field from this engine's own resolved "
                        "building.footprint_area/plot.area."
                    ),
                ),
                computed.model_copy(update={
                    "decision_id": decision_id, "source": "derived:coverage_field(evidence_decision)",
                }),
            )

    far_pair = results.get("far")
    if far_pair is not None and far_pair[0].status == DecisionStatus.ABSTAIN:
        computed = far_field(footprint_vf, plot_area_vf, legacy_plan.building.floor_count)
        if computed.value is not None:
            decision_id = _new_decision_id("far", "derived")
            results["far"] = (
                Decision(
                    id=decision_id, field_name="far", status=DecisionStatus.ACCEPT,
                    accepted_candidate_id="derived:far_field",
                    considered_candidate_ids=[footprint_decision.id, plot_area_decision.id],
                    resulting_value_field="far",
                    confidence_derivation=(
                        "No direct FAR evidence from independent_cv/vision; derived via "
                        "areas.far_field from this engine's own resolved "
                        "building.footprint_area/plot.area (+ legacy floor_count)."
                    ),
                ),
                computed.model_copy(update={
                    "decision_id": decision_id, "source": "derived:far_field(evidence_decision)",
                }),
            )


# Fields both LENGTH_FIELDS/NUMERIC_FIELDS and NormalizedPlan represent
# directly, as a top-level ValueField -- the only ones this bridge actually
# ships when the flag is on. Fields covered by `evidence_decision` but with
# no clean 1:1 `NormalizedPlan` attribute today (`building.plinth_area`,
# `far.area`, `building.gross_built_up_area`, `plot.net_area`, the height
# fields) are still computed and logged for visibility, never applied --
# forcing them into the wrong slot would be worse than leaving them alone.
_SHIPPABLE_FIELD_PATHS = {
    "plot.width", "plot.depth", "plot.area",
    "building.width", "building.depth", "building.footprint_area",
    "road.width",
    "setbacks.front", "setbacks.rear", "setbacks.left", "setbacks.right",
    "coverage", "far",
}


def compute_evidence_decision_fields(
    cv: Optional[IndependentCVResult],
    vision: Optional[VisionDocumentResult],
    pdf_path: Optional[str] = None,
    legacy_plan: Optional[NormalizedPlan] = None,
) -> dict[str, tuple[Decision, ValueField]]:
    """Returns `{field_name: (Decision, ValueField)}` for every field
    `final_fusion.build_final_agreement` also covers, resolved via
    `evidence_decision.decide_field` instead of `_value_field`.

    `legacy_plan` (Phase 9 Finding 1) is `pipeline.py`'s `old_plan` -- the
    OLD path's own already-resolved plan, passed in as a LOWER-PRIORITY
    fallback (see `_apply_legacy_fallback`) for exactly the fields where
    `independent_cv`/vision gave `decide_field` nothing to work with, plus
    coverage/FAR derivation from this engine's own resolved areas (see
    `_restore_derived_fields`). `None` (the default) preserves this
    function's original, fallback-free behavior exactly.
    """
    cm = _cv_map(cv) if cv is not None else {}
    all_measurements = list(cv.measurements) if cv is not None else []
    text_index = doc_ev.build_document_text_index(pdf_path)

    results: dict[str, tuple[Decision, ValueField]] = {}

    for field, semantic in LENGTH_FIELDS.items():
        cvm = cm.get(field)
        cv_value = None if cvm is None else (cvm.value_m if cvm.value_m is not None else cvm.value)
        vision_res = _vision_length(vision, semantic) if vision is not None else None
        vision_value = None if vision_res is None else vision_res[0]
        field_measurements = [m for m in all_measurements if m.field == field]
        results[field] = decide_field(field, cv_value, vision_value, "m", field_measurements, text_index)

    for field, semantics in NUMERIC_FIELDS.items():
        cvm = cm.get(field)
        cv_value = None if cvm is None else (cvm.value_m if cvm.value_m is not None else cvm.value)
        vision_res = _vision_numeric(vision, semantics) if vision is not None else None
        vision_value = None if vision_res is None else vision_res[0]
        unit = (vision_res[4] if vision_res else (cvm.unit if cvm and cvm.unit else "m2"))
        field_measurements = [m for m in all_measurements if m.field == field]
        results[field] = decide_field(field, cv_value, vision_value, unit, field_measurements, text_index)

    if legacy_plan is not None:
        legacy_map = _legacy_field_map(legacy_plan)
        for field, pair in list(results.items()):
            if field in _DERIVED_FIELDS:
                continue
            results[field] = _apply_legacy_fallback(field, pair[0], pair[1], legacy_map)

        _restore_derived_fields(results, legacy_plan)

        for field in _DERIVED_FIELDS:
            pair = results.get(field)
            if pair is None:
                continue
            results[field] = _apply_legacy_fallback(field, pair[0], pair[1], legacy_map)

    return results


def log_comparison(field_decisions: dict[str, tuple[Decision, ValueField]], old_plan: NormalizedPlan) -> None:
    """Log every field where the OLD (shipped-by-default) and NEW
    (evidence_decision) paths disagree on value or confidence level --
    the raw material for Phase 9's old/new comparison."""
    old_by_path = _legacy_field_map(old_plan)
    for field_name, old_vf in old_by_path.items():
        pair = field_decisions.get(field_name)
        if pair is None:
            continue
        _decision, new_vf = pair
        if old_vf.value != new_vf.value or old_vf.status != new_vf.status:
            logger.info(
                "evidence_decision Phase 8 comparison for %s: old=%s/%s new=%s/%s (%s)",
                field_name, old_vf.value, old_vf.status.value, new_vf.value, new_vf.status.value,
                _decision.confidence_derivation,
            )


def apply_evidence_decision_fields_to_plan(
    plan: NormalizedPlan, field_decisions: dict[str, tuple[Decision, ValueField]],
) -> NormalizedPlan:
    """Overwrite `plan`'s shippable fields (see `_SHIPPABLE_FIELD_PATHS`)
    with `evidence_decision`'s own `ValueField`s. Mutates and returns
    `plan` in place, matching `final_fusion.apply_final_agreement_to_plan`'s
    own existing mutation convention."""
    for field_name in _SHIPPABLE_FIELD_PATHS:
        pair = field_decisions.get(field_name)
        if pair is None:
            continue
        _decision, new_vf = pair
        if field_name == "plot.width":
            plan.plot.width = new_vf
        elif field_name == "plot.depth":
            plan.plot.depth = new_vf
        elif field_name == "plot.area":
            plan.plot.area = new_vf
        elif field_name == "building.width":
            plan.building.width = new_vf
        elif field_name == "building.depth":
            plan.building.depth = new_vf
        elif field_name == "building.footprint_area":
            plan.building.footprint_area = new_vf
        elif field_name == "road.width":
            plan.road.width = new_vf
        elif field_name == "setbacks.front":
            plan.setbacks.front = new_vf
        elif field_name == "setbacks.rear":
            plan.setbacks.rear = new_vf
        elif field_name == "setbacks.left":
            plan.setbacks.left = new_vf
        elif field_name == "setbacks.right":
            plan.setbacks.right = new_vf
        elif field_name == "coverage":
            plan.coverage = new_vf
        elif field_name == "far":
            plan.far = new_vf
    return plan


__all__ = ["compute_evidence_decision_fields", "log_comparison", "apply_evidence_decision_fields_to_plan"]
