"""
Phase 3 orchestration: `ExtractionResult` -> `NormalizedPlan`.

    build_normalized_plan(extraction_result, plan_id) -> NormalizedPlan

This is the ONLY function later phases (RuleEngine, report generation,
frontend) should call from this package. Everything else in
`backend/spatial_reasoning/` is an implementation detail.

Known limitation (Phase 1 contract, not fixable here without a breaking
schema change): `schemas.geometry.Dimension` carries no `page` field, so
when a document has multiple pages this pipeline reasons about
dimensions/candidates across the whole document rather than strictly
per-page. Real-world plan PDFs in this project are one plan per sheet,
so in practice this only matters for multi-sheet documents, and is
surfaced in `overall_confidence_note` when more than one page exists.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.config import get_settings
from backend.schemas.candidates import BuildingCandidate
from backend.schemas.enums import ConfidenceLevel, cap_confidence_level, confidence_level_from_score
from backend.schemas.evidence import Confidence, Conflict, GeometryEvidence, ValueField
from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import CoordinateSpace, NormalizedGeometry, Polygon
from backend.schemas.normalized_plan import (
    BuildingSection,
    NormalizedPlan,
    PlotSection,
    RoadSection,
    SetbackSection,
)
from backend.schemas.units import CanonicalUnit, UnitValue
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.areas import (
    _MAX_PHYSICALLY_PLAUSIBLE_COVERAGE_RATIO,
    coverage_field,
    far_field,
    find_area_statement,
    find_labeled_areas,
    polygon_area_field,
)
from backend.spatial_reasoning.building_context import extract_plan_context
from backend.spatial_reasoning.building_resolution import filter_and_rank_buildings, surviving_candidates
from backend.spatial_reasoning.consistency import check_physical_consistency
from backend.spatial_reasoning.dimension_classification import (
    ClassifiedDimension,
    DimensionSemanticType,
    classify_dimensions,
    vision_only_dimensions,
)
from backend.spatial_reasoning.evidence_reconciliation import EvidenceCandidate, to_value_field
from backend.spatial_reasoning.front_side import FrontSideResolution, resolve_front_side
from backend.spatial_reasoning.plot_resolution import plot_confidence, score_plot_candidates
from backend.spatial_reasoning.road_access import best_road_candidate, collect_access_evidence
from backend.spatial_reasoning.scale import dimension_length_metres, estimate_scale
from backend.spatial_reasoning.setbacks import compute_setbacks, flag_if_setback_implausible
from backend.spatial_reasoning.final_fusion import apply_final_agreement_to_plan
from backend.spatial_reasoning.vision_semantics import (
    floor_count_from_vision,
    preferred_vision_area,
    preferred_vision_value,
    semantic_frame_from_dimensions,
    site_small_dimension_candidates,
    vision_setbacks,
    setback_values_from_native_site,
)

# Notes on the alternatives:
#   `g(?:round|f)?\s*\+\s*N`  "G+2", "GROUND +2 FLOORS" and the BBMP "GF+2UF"
#                             (ground floor + N upper floors); a stilt level
#                             in "STILT, GF+2UF" is not part of the count.
#   `N floors` (fallback)     must NOT be followed by "PLAN": "2 FLOOR PLAN" is
#                             the caption of the SECOND floor's drawing, and
#                             reading it as "2 floors" shipped a wrong count at
#                             MEDIUM confidence on a real sheet whose true
#                             count is 3.
_FLOOR_COUNT_RE = re.compile(
    r"\b(?:g(?:round|f)?\s*\+\s*(?P<plus>\d+)|no\.?\s*of\s*floors?\s*[:\-]?\s*(?P<explicit>\d+)|"
    r"(?P<stories>\d+)\s*(?:storey|story|floors?)\b(?!\s*plans?\b))",
    re.I,
)


def _empty_plan(plan_id: str, document_id: str, reason: str) -> NormalizedPlan:
    missing = lambda r=reason: ValueField.missing(r)  # noqa: E731
    return NormalizedPlan(
        plan_id=plan_id,
        source_document_id=document_id,
        plot=PlotSection(width=missing(), depth=missing(), area=missing()),
        building=BuildingSection(width=missing(), depth=missing(), footprint_area=missing()),
        road=RoadSection(width=missing()),
        setbacks=SetbackSection(front=missing(), rear=missing(), left=missing(), right=missing()),
        coverage=missing(),
        far=missing(),
        overall_confidence_note=reason,
    )


def _cap_field_confidence(vf: ValueField, cap: ConfidenceLevel, cap_reason: str) -> ValueField:
    """Return `vf` unchanged if its confidence is already at or below `cap`'s
    severity (see `enums.cap_confidence_level`); otherwise a copy with the
    level lowered to `cap` and the reason annotated with why. Never mutates
    `vf` in place -- `ValueField` instances are shared with the fields this
    was built from, and other callers must not see it change out from under
    them."""
    capped_level = cap_confidence_level(vf.confidence.level, cap)
    if capped_level == vf.confidence.level:
        return vf
    original_reason = vf.confidence.reason or "no reason recorded"
    return vf.model_copy(update={
        "confidence": Confidence(
            level=capped_level, score=vf.confidence.score,
            reason=f"{original_reason} (confidence capped: {cap_reason})",
        )
    })


def _building_polygon(bc: BuildingCandidate) -> Optional[Polygon]:
    if bc.geometry is None:
        return None
    if bc.geometry.polygon is not None:
        return bc.geometry.polygon
    if bc.geometry.bounding_box is not None:
        return geo.bbox_to_polygon(bc.geometry.bounding_box)
    return None


def _value_metres_lookup(points_per_metre: float):
    def _lookup(dim) -> Optional[float]:
        text_value = dimension_length_metres(dim)
        if text_value is not None:
            return text_value
        if dim.geometry is not None:
            return dim.geometry.length / points_per_metre
        return None

    return _lookup


def _scale_front_side(fsr: FrontSideResolution, points_per_metre: float) -> FrontSideResolution:
    return FrontSideResolution(
        front_edges=[geo.scale_line(e, points_per_metre) for e in fsr.front_edges],
        rear_edges=[geo.scale_line(e, points_per_metre) for e in fsr.rear_edges],
        left_edges=[geo.scale_line(e, points_per_metre) for e in fsr.left_edges],
        right_edges=[geo.scale_line(e, points_per_metre) for e in fsr.right_edges],
        confidence=fsr.confidence,
        reasoning=fsr.reasoning,
        evidence_level=fsr.evidence_level,
    )


def _detect_floor_count(text_evidence) -> Optional[ValueField[int]]:
    for t in text_evidence:
        m = _FLOOR_COUNT_RE.search(t.raw_text or "")
        if not m:
            continue
        if m.group("plus") is not None:
            value = int(m.group("plus")) + 1  # "G+1" = ground + 1 upper floor = 2 floors
            note = f"Floor count parsed from 'G+{m.group('plus')}' label."
        elif m.group("explicit") is not None:
            value = int(m.group("explicit"))
            note = "Floor count parsed from an explicit 'No. of floors' label."
        else:
            value = int(m.group("stories"))
            note = "Floor count parsed from a '<N> storey/floors' label."
        return ValueField[int](
            value=value,
            confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason=note),
            source="text label",
        )
    # No numeric phrasing anywhere: fall back to counting the floors the
    # sheet NAMES (title enumeration, one caption per floor plan, one area-
    # table row per floor). Corroboration between independent forms is what
    # earns MEDIUM; a single form is only LOW.
    from backend.spatial_reasoning.floor_levels import floor_count_from_named_levels

    named = floor_count_from_named_levels(t.raw_text or "" for t in text_evidence)
    if named is not None:
        return ValueField[int](
            value=named.count,
            confidence=Confidence(
                level=ConfidenceLevel.MEDIUM if len(named.forms) >= 2 else ConfidenceLevel.LOW,
                reason=named.note,
            ),
            source="named floor levels",
        )
    return None


def _vision_document(extraction: ExtractionResult):
    if not extraction.vision_pages:
        return None
    from backend.schemas.vision import VisionDocumentResult
    return VisionDocumentResult(pages=extraction.vision_pages, model_name="pipeline", enabled=True)


def _apply_phase8_bridge(plan: NormalizedPlan, extraction: ExtractionResult, vision_doc) -> NormalizedPlan:
    """Architecture V2, Phase 8's shared tail (see ARCHITECTURE_V2.md's phased
    order): `evidence_decision.py`'s field-level decisions are ALWAYS also
    computed and logged for comparison here, regardless of the feature flag --
    only which plan is RETURNED depends on it. This lets real discrepancies
    accumulate in the logs (the raw material for Phase 9's old/new comparison
    across all real plans) before the flag's default is ever considered for a
    flip. A failure computing the new path must never break the old,
    always-working one.

    Factored out of `build_normalized_plan`'s ordinary tail (Phase DXF-0) so
    the SAME logic also runs for the DXF/no-legacy-winner early-return branch
    above, which previously returned before this bridge was ever reached --
    `use_evidence_decision_engine=True` had zero effect on any DXF plan until
    this fix, confirmed by tracing the control flow directly: DXF never
    populates the legacy `plot_candidates` pool, so `winner` above is always
    `None`, and every DXF plan took the early-return path.
    """
    try:
        from backend.spatial_reasoning.evidence_decision_bridge import (
            apply_evidence_decision_fields_to_plan,
            compute_evidence_decision_fields,
            log_comparison,
        )

        field_decisions = compute_evidence_decision_fields(extraction.independent_cv, vision_doc, legacy_plan=plan)
        log_comparison(field_decisions, plan)
    except Exception as exc:  # noqa: BLE001 -- shadow computation must never break the shipping path
        from backend.tools.logging_config import get_logger

        get_logger(__name__).warning("evidence_decision Phase 8 shadow computation failed: %s", exc)
        field_decisions = None

    if get_settings().use_evidence_decision_engine and field_decisions is not None:
        return apply_evidence_decision_fields_to_plan(plan, field_decisions)
    return plan


def build_normalized_plan(extraction: ExtractionResult, plan_id: Optional[str] = None) -> NormalizedPlan:
    """Resolve an extraction into a plan, then withdraw physically impossible values."""
    from backend.spatial_reasoning.physical_bounds import enforce_physical_bounds

    return enforce_physical_bounds(_build_normalized_plan_unchecked(extraction, plan_id))


def _build_normalized_plan_unchecked(extraction: ExtractionResult, plan_id: Optional[str] = None) -> NormalizedPlan:
    plan_id = plan_id or f"plan-{extraction.document_id}"

    winner, all_scored, plot_note = score_plot_candidates(extraction)
    if winner is None:
        # Independent CV is the Phase-3.3 authority. If the legacy/global
        # candidate pool cannot identify a plot, still build the plan from
        # the independent site-plan evidence instead of returning all fields
        # as MISSING.
        if extraction.independent_cv is not None and extraction.independent_cv.measurements:
            from backend.spatial_reasoning.final_fusion import build_final_agreement
            from backend.schemas.vision import VisionDocumentResult
            empty_vision = VisionDocumentResult(model_name="disabled", pages=[], enabled=False)
            fusion = build_final_agreement(extraction.independent_cv, _vision_document(extraction) or empty_vision)
            missing = lambda reason: ValueField.missing(reason)
            def fv(name, unit):
                item = fusion["values"].get(name)
                if item and item["value"] is not None and item["status"] != "CONFLICT":
                    # `item["confidence"]` already carries whatever
                    # `build_final_agreement`/`_value_field` correctly
                    # derived from the SOURCE measurement's own confidence
                    # (see `_value_field`'s own docstring for the real bug
                    # this replaced: CV-only/Vision-only values used to be
                    # hardcoded MEDIUM here regardless of status,
                    # independent of how much the extractor itself trusted
                    # the number -- silently defeating every downstream
                    # LOW-confidence gate, including `backend/compliance/
                    # engine.py`'s own REQUIRES_REVIEW check).
                    return ValueField(value=item["value"], normalized_value=UnitValue(magnitude=item["value"], unit=unit),
                        confidence=Confidence(level=ConfidenceLevel(item["confidence"]), reason=item["reason"]), source=item["source"])
                return missing(f"No non-conflicting independent evidence for {name}.")
            plot_area_fv = fv("plot.area", "m2")
            footprint_area_fv = fv("building.footprint_area", "m2")
            # coverage/far are direct-evidence fields in `fusion["values"]`
            # (a literal printed "Coverage %"/"FAR" reading, or a Vision
            # COVERAGE_PERCENT/FAR_RATIO area) -- most independent-CV-only
            # documents (e.g. a DXF with no such printed statement) never
            # have one, even though footprint_area/plot_area above are
            # both already resolved and are enough to compute both
            # deterministically. Without this fallback, coverage/far
            # stayed MISSING here even when perfectly computable -- the
            # same gap `apply_final_agreement_to_plan` (this function's
            # other, more common return path, below) already closes for
            # the ordinary case; this mirrors that fix for the no-winner/
            # independent-CV-only path.
            coverage_fv = fv("coverage", "%")
            if coverage_fv.value is None:
                computed_coverage = coverage_field(footprint_area_fv, plot_area_fv)
                if computed_coverage.value is not None:
                    coverage_fv = computed_coverage
            far_fv = fv("far", "ratio")
            if far_fv.value is None:
                computed_far = far_field(footprint_area_fv, plot_area_fv, None)
                if computed_far.value is not None:
                    far_fv = computed_far
            plan = NormalizedPlan(
                plan_id=plan_id, source_document_id=extraction.document_id,
                plot=PlotSection(width=fv("plot.width","m"), depth=fv("plot.depth","m"), area=plot_area_fv),
                building=BuildingSection(width=fv("building.width","m"), depth=fv("building.depth","m"), footprint_area=footprint_area_fv),
                road=RoadSection(width=fv("road.width","m")),
                setbacks=SetbackSection(front=fv("setbacks.front","m"), rear=fv("setbacks.rear","m"), left=fv("setbacks.left","m"), right=fv("setbacks.right","m")),
                coverage=coverage_fv, far=far_fv,
                overall_confidence_note="Legacy global plot resolver had no winner; final values came from independent CV + Vision agreement layer.",
            )
            # Same physical-plausibility flag as `compute_setbacks`/
            # `pipeline.py`'s main path and `apply_final_agreement_to_plan`
            # (final_fusion.py) -- see `flag_if_setback_implausible`'s own
            # docstring for why this only ever attaches `.conflict` and
            # never touches `.value`/`.confidence`. This branch has its own
            # independent path to a NormalizedPlan, so it needs its own copy
            # of the same guard rather than relying on one of the others to
            # have already run.
            for side in ("front", "rear", "left", "right"):
                setattr(plan.setbacks, side, flag_if_setback_implausible(
                    side, getattr(plan.setbacks, side), f"setbacks.{side}",
                    plan.plot.width, plan.plot.depth, plan.building.width, plan.building.depth,
                    frontage_relative_axes=False,  # values come straight from the sheet, not the classifier
                ))
            return _apply_phase8_bridge(plan, extraction, _vision_document(extraction))
        return _empty_plan(
            plan_id, extraction.document_id, f"Cannot resolve a NormalizedPlan: {plot_note}"
        )

    plot_candidate = winner.candidate
    page = plot_candidate.geometry.source_page
    plot_conf_level, plot_conf_reason = plot_confidence(winner, all_scored)

    # The sheet's own printed "AREA STATEMENT" table (Site Area + one row per
    # floor) is the most authoritative source available for plot.area and
    # building.footprint_area where it exists: it's the figure the architect
    # and sanctioning authority actually signed off on, and it's immune to
    # the two failure modes that hurt CV-derived geometry on a composite
    # sheet -- (a) the plot/building rectangle picked up from the wrong
    # sub-drawing, and (b) a non-rectangular (trapezoidal/irregular) plot
    # where width x depth is not a valid area formula at all. It is used
    # below only where present and physically plausible; when the sheet has
    # no such table this is simply empty and every field falls back to the
    # existing geometry-based resolution untouched.
    area_statement = find_area_statement(extraction.text_evidence)

    # Vision is allowed to supply semantics, while native dimension lines
    # supply the actual page-space anchors. This is the key distinction that
    # prevents a bad OpenCV contour from turning into a bad plot/building
    # measurement.
    vision_plot_width = preferred_vision_value(extraction, page, "PLOT_WIDTH")
    vision_plot_depth = preferred_vision_value(extraction, page, "PLOT_DEPTH")
    vision_building_width = preferred_vision_value(extraction, page, "BUILDING_WIDTH")
    vision_building_depth = preferred_vision_value(extraction, page, "BUILDING_DEPTH")

    semantic_plot_bbox = None
    if vision_plot_width is not None and vision_plot_depth is not None:
        semantic_plot_bbox = semantic_frame_from_dimensions(
            extraction, page, vision_plot_width, vision_plot_depth
        )

    semantic_building_bbox = None
    if semantic_plot_bbox is not None and vision_building_width is not None and vision_building_depth is not None:
        semantic_building_bbox = semantic_frame_from_dimensions(
            extraction, page, vision_building_width, vision_building_depth, semantic_plot_bbox
        )

    plot_polygon_page = (
        geo.bbox_to_polygon(semantic_plot_bbox)
        if semantic_plot_bbox is not None
        else plot_candidate.geometry.polygon
    )

    # --- scale --------------------------------------------------------
    plot_bbox_page = semantic_plot_bbox or (plot_candidate.geometry.bounding_box if plot_candidate.geometry else None)
    scale_est = estimate_scale(
        extraction.dimensions,
        page=page,
        region_bbox=plot_bbox_page,
        region_padding_factor=get_settings().local_scale_region_padding_factor,
    )
    # If semantic plot dimensions are explicit metres, derive the page scale
    # directly from the semantic frame. This prevents unrelated floor/detail
    # dimensions from contaminating scale estimation.
    if semantic_plot_bbox is not None and vision_plot_width and vision_plot_depth:
        x_ppm = semantic_plot_bbox.width / vision_plot_width
        y_ppm = semantic_plot_bbox.height / vision_plot_depth
        if x_ppm > 1e-6 and y_ppm > 1e-6:
            ppm = (x_ppm + y_ppm) / 2.0
            scale_est.reason = (
                f"Semantic plot frame scale: {x_ppm:.3f} pt/m horizontally and "
                f"{y_ppm:.3f} pt/m vertically; unrelated sheet dimensions excluded."
            )
        else:
            ppm = scale_est.points_per_metre
    else:
        ppm = scale_est.points_per_metre

    # --- building -------------------------------------------------------
    filtered_buildings = filter_and_rank_buildings(extraction, plot_candidate)
    survivors = surviving_candidates(filtered_buildings)
    survivor_polygons_page = [p for p in (_building_polygon(b) for b in survivors) if p is not None]
    primary_building = (
        max(survivors, key=lambda b: (_building_polygon(b).area if _building_polygon(b) else 0))
        if survivors
        else None
    )
    if semantic_building_bbox is not None:
        semantic_building_polygon = geo.bbox_to_polygon(semantic_building_bbox)
    else:
        semantic_building_polygon = _building_polygon(primary_building) if primary_building else None

    # --- road / access ---------------------------------------------------
    road_candidate = best_road_candidate(extraction, page)
    road_bbox_page = road_candidate.geometry.bounding_box if (road_candidate and road_candidate.geometry) else None
    road_bboxes_page = [road_bbox_page] if road_bbox_page is not None else []
    access_evidence = collect_access_evidence(extraction.text_evidence, page)

    # --- front-side reasoning (page space, then scaled) -------------------
    fsr_page = resolve_front_side(plot_polygon_page, road_bbox_page, access_evidence)
    fsr_metric = _scale_front_side(fsr_page, ppm)

    # --- dimension classification (page space) ----------------------------
    classified: list[ClassifiedDimension] = classify_dimensions(
        extraction.dimensions,
        plot_polygon_page,
        survivor_polygons_page,
        fsr_page.front_edges,
        fsr_page.rear_edges,
        fsr_page.left_edges,
        fsr_page.right_edges,
        road_bboxes_page,
        _value_metres_lookup(ppm),
        extraction.vision_pages,
    )
    # Vision <-> CV fusion, part 2: dimensions the vision model read that
    # the deterministic native/OCR pipeline never produced a candidate for
    # at all (as opposed to the vision-as-hint fusion already wired into
    # classify_dimensions() above, which only re-labels dimensions that
    # already existed). See vision_only_dimensions()'s docstring.
    classified = classified + vision_only_dimensions(extraction.vision_pages, extraction.dimensions, page)

    def _classified_of(t: DimensionSemanticType) -> list[ClassifiedDimension]:
        return [c for c in classified if c.semantic_type is t and c.value_metres is not None]

    # --- metric-space geometry --------------------------------------------
    plot_polygon_metric = geo.scale_polygon(plot_polygon_page, ppm)
    building_polygon_metric = (
        geo.scale_polygon(semantic_building_polygon, ppm) if semantic_building_polygon is not None else
        (geo.scale_polygon(_building_polygon(primary_building), ppm) if primary_building else None)
    )

    # --- plot width/depth (front/rear edge length vs classified dims) -----
    def _edge_group_length_candidates(edges_page, label: str) -> list[EvidenceCandidate]:
        out = []
        for e in edges_page:
            out.append(
                EvidenceCandidate(
                    value=round(e.length / ppm, 4),
                    source=f"geometry ({label} edge)",
                    evidence=GeometryEvidence(description=f"{label} plot boundary edge length, scaled to metres."),
                )
            )
        return out

    # Prefer semantically grounded VLM values for the core dimensions. The
    # native dimension layer is the measurement evidence; CV edge lengths are
    # only a fallback because this sheet contains many unrelated rectangles.
    if vision_plot_width is not None:
        plot_width = to_value_field(
            [EvidenceCandidate(value=round(vision_plot_width, 4), source="vision semantic PLOT_WIDTH + native text grounding")],
            "plot.width",
        )
    else:
        width_candidates = _edge_group_length_candidates(fsr_page.front_edges, "front") + _edge_group_length_candidates(
            fsr_page.rear_edges, "rear"
        )
        for c in _classified_of(DimensionSemanticType.PLOT_WIDTH):
            width_candidates.append(
                EvidenceCandidate(value=round(c.value_metres, 4), source=f"dimension label ('{c.dimension.label}')")
            )
        plot_width = to_value_field(width_candidates, "plot.width", "No frontage edge or PLOT_WIDTH dimension found.")

    if vision_plot_depth is not None:
        plot_depth = to_value_field(
            [EvidenceCandidate(value=round(vision_plot_depth, 4), source="vision semantic PLOT_DEPTH + native text grounding")],
            "plot.depth",
        )
    else:
        depth_candidates = _edge_group_length_candidates(fsr_page.left_edges, "left") + _edge_group_length_candidates(
            fsr_page.right_edges, "right"
        )
        for c in _classified_of(DimensionSemanticType.PLOT_DEPTH):
            depth_candidates.append(
                EvidenceCandidate(value=round(c.value_metres, 4), source=f"dimension label ('{c.dimension.label}')")
            )
        plot_depth = to_value_field(depth_candidates, "plot.depth", "No lateral edge or PLOT_DEPTH dimension found.")

    plot_rectangularity = geo.rectangularity(plot_polygon_page)
    # Always route through polygon_area_field's own rectangularity gate
    # (>= 0.9 -> width x depth at HIGH; otherwise the polygon's own
    # shoelace area at MEDIUM) instead of unconditionally trusting
    # width x depth here whenever both happened to resolve. Audit finding
    # (BUILDCHECK_FORENSIC_AUDIT.md Sec 4.5/5.7): this branch previously
    # skipped that same gate -- which the "else" fallback right below it
    # already applied -- so a confidently-resolved width/depth pair on a
    # non-rectangular plot (where width*depth is not a valid area formula
    # at all) still shipped plot.area at HIGH confidence.
    plot_area = polygon_area_field(plot_polygon_metric, plot_width, plot_depth, plot_rectangularity, "plot.area")

    site_area_stmt = area_statement.get("site_area")
    if site_area_stmt is not None and site_area_stmt.value > 0:
        note = f"Printed Area Statement 'Site Area' label ({site_area_stmt.value:.3f} m²)."
        if plot_area.value is not None and plot_area.value > 0:
            rel_diff = abs(site_area_stmt.value - plot_area.value) / plot_area.value
            note += (
                f" Geometry-derived plot.area was {plot_area.value:.3f} m² (relative diff {rel_diff:.1%})"
                + (
                    " -- consistent with a non-rectangular/irregular plot, where width x depth "
                    "over/under-states the true area; the printed figure is preferred."
                    if rel_diff > 0.03
                    else "; both agree."
                )
            )
        plot_area = ValueField[float](
            value=round(site_area_stmt.value, 4),
            normalized_value=UnitValue(magnitude=round(site_area_stmt.value, 4), unit=CanonicalUnit.SQUARE_METRE.value),
            confidence=Confidence(level=ConfidenceLevel.HIGH, reason=note),
            source="plot.area = printed Area Statement 'Site Area'",
        )

    plot_section = PlotSection(
        geometry=NormalizedGeometry(
            coordinate_space=CoordinateSpace.METRIC_PLAN,
            polygon=plot_polygon_metric,
            bounding_box=plot_polygon_metric.bounding_box,
            points_per_metre=ppm,
            rotation_degrees=plot_candidate.geometry.rotation_degrees,
            source_page=page,
        ),
        width=plot_width,
        depth=plot_depth,
        area=plot_area,
    )

    # --- building width/depth/footprint ------------------------------------
    if (primary_building is not None and building_polygon_metric is not None) or semantic_building_bbox is not None:
        frontage_deg = (
            geo.line_orientation_degrees(fsr_page.front_edges[0]) if fsr_page.front_edges else 0.0
        )
        bbox_page = semantic_building_bbox or (_building_polygon(primary_building).bounding_box if primary_building else None)
        horiz_deg = 0.0
        align_horizontal = geo.orientation_alignment(frontage_deg, horiz_deg)
        if bbox_page is not None and semantic_building_bbox is None:
            if align_horizontal >= 0.5:
                width_px, depth_px = bbox_page.width, bbox_page.height
            else:
                width_px, depth_px = bbox_page.height, bbox_page.width
        else:
            width_px = bbox_page.width if bbox_page is not None else 0.0
            depth_px = bbox_page.height if bbox_page is not None else 0.0

        if vision_building_width is not None:
            building_width = to_value_field(
                [EvidenceCandidate(value=round(vision_building_width, 4), source="vision semantic BUILDING_WIDTH + native text grounding")],
                "building.width",
            )
        else:
            b_width_candidates = [EvidenceCandidate(value=round(width_px / ppm, 4), source="geometry (building bounding box, frontage-aligned side)", evidence=GeometryEvidence(description="Building footprint extent aligned with the plot frontage axis."))]
            for c in _classified_of(DimensionSemanticType.BUILDING_WIDTH):
                b_width_candidates.append(EvidenceCandidate(value=round(c.value_metres, 4), source=f"dimension label ('{c.dimension.label}')"))
            building_width = to_value_field(b_width_candidates, "building.width")

        if vision_building_depth is not None:
            building_depth = to_value_field(
                [EvidenceCandidate(value=round(vision_building_depth, 4), source="vision semantic BUILDING_DEPTH + native text grounding")],
                "building.depth",
            )
        else:
            b_depth_candidates = [EvidenceCandidate(value=round(depth_px / ppm, 4), source="geometry (building bounding box, depth-aligned side)", evidence=GeometryEvidence(description="Building footprint extent aligned with the plot depth axis."))]
            for c in _classified_of(DimensionSemanticType.BUILDING_DEPTH):
                b_depth_candidates.append(EvidenceCandidate(value=round(c.value_metres, 4), source=f"dimension label ('{c.dimension.label}')"))
            building_depth = to_value_field(b_depth_candidates, "building.depth")

        explicit_plinth = preferred_vision_area(extraction, page, "PLINTH_AREA")
        # Ground Floor is the byelaw-relevant footprint for coverage in the
        # overwhelming majority of plans (the building's largest/base
        # extent); Stilt Floor is used only when there is no Ground Floor
        # row at all (some sheets label the ground-level floor "Stilt").
        floor_area_stmt = area_statement.get("ground_floor") or area_statement.get("stilt_floor")
        footprint_source = "building.footprint_area (sum of validated building block polygons)"
        if explicit_plinth is not None:
            footprint_area_m2 = explicit_plinth
            footprint_note = "Explicit PLINTH_AREA from vision, grounded to native page text."
            footprint_level = ConfidenceLevel.HIGH
            footprint_source = "building.footprint_area = explicit PLINTH_AREA"
        elif floor_area_stmt is not None and floor_area_stmt.value > 0:
            footprint_area_m2 = floor_area_stmt.value
            geometry_cross_check = (
                building_width.value * building_depth.value
                if building_width.value is not None and building_depth.value is not None
                else None
            )
            footprint_note = f"Printed Area Statement '{floor_area_stmt.label}' area ({footprint_area_m2:.3f} m²)."
            if geometry_cross_check is not None:
                rel_diff = abs(footprint_area_m2 - geometry_cross_check) / footprint_area_m2 if footprint_area_m2 else 0.0
                footprint_note += (
                    f" CV/geometry-derived footprint was {geometry_cross_check:.3f} m² (relative diff "
                    f"{rel_diff:.1%}) -- likely picked up a rectangle from an unrelated drawing on the "
                    "same sheet; the printed figure is preferred."
                    if rel_diff > 0.15
                    else "; consistent with the CV/geometry-derived footprint."
                )
            footprint_level = ConfidenceLevel.HIGH
            footprint_source = f"building.footprint_area = printed Area Statement '{floor_area_stmt.label}'"
        elif building_width.value is not None and building_depth.value is not None:
            footprint_area_m2 = building_width.value * building_depth.value
            footprint_note = "Building footprint computed from semantically resolved building width x depth."
            footprint_level = ConfidenceLevel.HIGH
            footprint_source = "building.footprint_area = building.width x building.depth"
        else:
            total_footprint_pts2 = sum(p.area for p in survivor_polygons_page)
            footprint_area_m2 = total_footprint_pts2 / (ppm**2)
            footprint_note = f"Sum of {len(survivors)} validated building block(s)' footprint polygon area(s), scaled to square metres."
            footprint_level = ConfidenceLevel.HIGH if len(survivors) <= 2 else ConfidenceLevel.MEDIUM
        if plot_area.value is not None and plot_area.value > 0 and (
            footprint_area_m2 / plot_area.value > _MAX_PHYSICALLY_PLAUSIBLE_COVERAGE_RATIO
        ):
            # FIX #9 (phase3.1): a building footprint that exceeds the
            # resolved plot area is physically impossible, not just
            # "high coverage" — almost always a sign the plot/building
            # candidates that got paired together don't actually refer to
            # the same real-world site (mismatched drawing regions/scale
            # rather than a true measurement). Mark the value itself
            # CONFLICTING (not a bogus number) so it doesn't silently
            # propagate into coverage/FAR either.
            footprint_area = ValueField[float].conflicting(
                Conflict(
                    description=(
                        f"Computed building.footprint_area ({footprint_area_m2:.3f} m²) exceeds "
                        f"the resolved plot.area ({plot_area.value:.3f} m²), which is physically "
                        "impossible. This most likely means the selected building candidate does "
                        "not actually belong to the same drawing region as the selected plot "
                        "candidate."
                    ),
                    conflicting_raw_values=[
                        UnitValue(magnitude=round(footprint_area_m2, 4), unit=CanonicalUnit.SQUARE_METRE.value),
                        UnitValue(magnitude=plot_area.value, unit=CanonicalUnit.SQUARE_METRE.value),
                    ],
                    conflicting_sources=["building.footprint_area (computed)", "plot.area"],
                )
            )
        else:
            footprint_area = ValueField[float](
                value=round(footprint_area_m2, 4),
                normalized_value=UnitValue(
                    magnitude=round(footprint_area_m2, 4), unit=CanonicalUnit.SQUARE_METRE.value
                ),
                confidence=Confidence(level=footprint_level, reason=footprint_note),
                source=footprint_source,
            )

        # An explicit printed declaration ("G+2", "No. of floors: 3", "3
        # storey") is stronger evidence than counting distinct floor-plan
        # regions on the sheet: the latter conflates a genuine additional
        # floor with a TERRACE/STILT/BASEMENT region (region_detection.py's
        # OTHER_FLOOR_PLAN type matches all of these) that a printed floor
        # count deliberately does not include -- confirmed live on
        # PLAN5.pdf, whose sheet states "(GROUND +2 FLOORS)" (G+2 = 3
        # floors total) but whose vision-region count came to 4 by also
        # counting its drawn TERRACE_FLOOR_PLAN region as if it were a
        # fourth habitable floor. Try the explicit text first; fall back to
        # the region count only when the sheet states no explicit number.
        floor_count = _detect_floor_count(extraction.text_evidence)
        if floor_count is None:
            floor_count_value = floor_count_from_vision(extraction, page)
            if floor_count_value is not None:
                floor_count = ValueField[int](
                    value=floor_count_value,
                    confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="Distinct floor-plan regions identified by the vision semantic layer (no explicit floor-count text found on the sheet; this may include non-floor regions such as a terrace)."),
                    source="vision floor-plan region count",
                )

        building_section = BuildingSection(
            geometry=NormalizedGeometry(
                coordinate_space=CoordinateSpace.METRIC_PLAN,
                polygon=building_polygon_metric,
                bounding_box=building_polygon_metric.bounding_box,
                points_per_metre=ppm,
                rotation_degrees=(primary_building.geometry.rotation_degrees if primary_building is not None else 0.0),
                source_page=(primary_building.geometry.source_page if primary_building is not None else page),
            ),
            width=building_width,
            depth=building_depth,
            footprint_area=footprint_area,
            floor_count=floor_count,
        )
    else:
        building_section = BuildingSection(
            width=ValueField.missing("No building candidate survived filtering (rooms/furniture/compound-wall exclusion)."),
            depth=ValueField.missing("No building candidate survived filtering."),
            footprint_area=ValueField.missing("No building candidate survived filtering."),
        )

    # --- road ---------------------------------------------------------------
    road_candidates_for_width: list[EvidenceCandidate] = []
    road_geometry = None
    if (
        road_candidate is not None
        and road_candidate.geometry is not None
        and not road_candidate.id.startswith("text-road-")
        and re.search(r"\d", road_candidate.name_or_label or "")
    ):
        bbox = road_candidate.geometry.bounding_box
        if bbox is not None:
            road_width_px = min(bbox.width, bbox.height)
            road_candidates_for_width.append(
                EvidenceCandidate(
                    value=round(road_width_px / ppm, 4),
                    source="geometry (road candidate short-side width; explicitly labeled road candidate)",
                    evidence=GeometryEvidence(description="Road candidate with an explicit numeric road label."),
                )
            )
            road_geometry = NormalizedGeometry(
                coordinate_space=CoordinateSpace.METRIC_PLAN,
                bounding_box=geo.scale_bbox(bbox, ppm),
                points_per_metre=ppm,
                rotation_degrees=road_candidate.geometry.rotation_degrees,
                source_page=road_candidate.geometry.source_page,
            )
    vision_road_width = preferred_vision_value(extraction, page, "ROAD_WIDTH")
    if vision_road_width is not None:
        road_candidates_for_width.append(
            EvidenceCandidate(value=round(vision_road_width, 4), source="vision semantic ROAD_WIDTH + native text grounding")
        )
    else:
        # Geometry-only proximity to a road is not enough to call a number a
        # road width; floor/parking dimensions frequently sit beside the road
        # region. Require an explicit textual road association here.
        for c in _classified_of(DimensionSemanticType.ROAD_WIDTH):
            if re.search(r"\broad\b", c.dimension.label or "", re.I):
                road_candidates_for_width.append(
                    EvidenceCandidate(value=round(c.value_metres, 4), source=f"dimension label ('{c.dimension.label}')")
                )
    road_width = to_value_field(
        road_candidates_for_width, "road.width", "No explicitly dimensioned road width found."
    )
    road_section = RoadSection(geometry=road_geometry, width=road_width)

    # --- setbacks -------------------------------------------------------------
        # --- setbacks -------------------------------------------------------------
    setbacks_section = compute_setbacks(
        building_polygon_metric,
        fsr_metric,
        classified,
        ppm,
        plot_width=plot_width,
        plot_depth=plot_depth,
        building_width=building_section.width,
        building_depth=building_section.depth,
    )

    # Replace noisy geometry-derived setbacks with explicit small site-plan
    # dimensions when the semantic site/building frames are available. This
    # is still evidence-first: values come from printed dimension labels and
    # their spatial orientation, never from a regulation-specific default.
    explicit_setbacks = vision_setbacks(extraction, page)
    native_site_setbacks = setback_values_from_native_site(
        extraction, page, semantic_plot_bbox, road_bbox_page, ppm
    )
    explicit_setbacks = {**native_site_setbacks, **explicit_setbacks}
    if semantic_plot_bbox is not None:
        # Once a semantic site frame has been established, do not retain a
        # noisy CV distance for a side that has no explicit setback label.
        # Missing is preferable to an apparently precise but unsupported
        # number.
        for side in ("front", "rear", "left", "right"):
            value = explicit_setbacks.get(side)
            if value is None:
                setattr(setbacks_section, side, ValueField[float].missing(
                    "No explicit setback dimension was found for this plot edge."
                ))
            else:
                setattr(
                    setbacks_section,
                    side,
                    ValueField[float](
                        value=round(value, 4),
                        normalized_value=UnitValue(magnitude=round(value, 4), unit=CanonicalUnit.METRE.value),
                        confidence=Confidence(level=ConfidenceLevel.HIGH, reason="Explicit site-plan setback dimension, spatially assigned to the plot edge."),
                        source="site-plan setback dimension",
                    ),
                )
    else:
        for side, value in explicit_setbacks.items():
            setattr(
                setbacks_section,
                side,
                ValueField[float](
                    value=round(value, 4),
                    normalized_value=UnitValue(magnitude=round(value, 4), unit=CanonicalUnit.METRE.value),
                    confidence=Confidence(level=ConfidenceLevel.HIGH, reason="Explicit site-plan setback dimension, spatially assigned to the plot edge."),
                    source="site-plan setback dimension",
                ),
            )

    # A setback value -- whether from `compute_setbacks` above, a vision
    # reading, or a native-site label -- is still just one candidate, not
    # ground truth: it can come from a misread caption or a label anchored
    # to the wrong sub-drawing. Flag (never silently reject or null out) any
    # value that cannot physically fit the plot/building extent already
    # resolved for this side, the same way `pdf_dxf_reconciliation.py`
    # flags a PDF/DXF disagreement via `.conflict` alone -- confirmed live
    # on plans where building.width came out >= plot.width and the shipped
    # left/right setbacks didn't sum anywhere near the available width.
    for side in ("front", "rear", "left", "right"):
        setattr(setbacks_section, side, flag_if_setback_implausible(
            side, getattr(setbacks_section, side), f"setbacks.{side}",
            plot_width, plot_depth, building_section.width, building_section.depth,
        ))

    # --- coverage / FAR ---------------------------------------------------------
    coverage = coverage_field(building_section.footprint_area, plot_area)
    far = far_field(building_section.footprint_area, plot_area, building_section.floor_count)

    # --- confidence cap: a field must never ship MORE confident than the
    # plot-candidate-selection step itself was in its own winner. --------------
    # `plot_conf_level` (computed above, via `plot_confidence`) is a real,
    # margin-aware judgment of how much this whole resolution should be
    # trusted -- but until this fix it only ever reached a human-readable
    # note (`overall_confidence_note`), never actually constraining any
    # field's own shipped `ValueField.confidence`. A weak or spurious
    # winner (e.g. a single geometric "candidate" detected on a document
    # with no real site plan at all) could still ship plot.width/
    # building.width/building.depth at HIGH purely because two readings of
    # that SAME weak candidate happened to numerically agree with each
    # other (`evidence_reconciliation.to_value_field`'s "do these values
    # agree" logic has no notion of whether the candidate itself was
    # trustworthy). Confirmed live on PLAN7 (a photograph of a blueprint
    # with no vector geometry, where every geometric field should stay
    # unresolved) via `backend/tools/run_evidence_decision_shadow.py`'s
    # shadow-mode comparison -- see ARCHITECTURE_V2.md's Implementation
    # log. This does not touch `score_plot_candidates` itself (which
    # deliberately always returns a winner when candidates exist, even a
    # weak lone one -- see `tests/test_plot_resolution.py::
    # test_plot_confidence_high_with_clear_margin`), only makes sure its
    # own already-computed confidence judgment actually propagates.
    _plot_cap_reason = f"plot-candidate selection scored {plot_conf_level.value.lower()}: {plot_conf_reason}"
    plot_width = _cap_field_confidence(plot_width, plot_conf_level, _plot_cap_reason)
    plot_depth = _cap_field_confidence(plot_depth, plot_conf_level, _plot_cap_reason)
    plot_area = _cap_field_confidence(plot_area, plot_conf_level, _plot_cap_reason)
    building_section.width = _cap_field_confidence(building_section.width, plot_conf_level, _plot_cap_reason)
    building_section.depth = _cap_field_confidence(building_section.depth, plot_conf_level, _plot_cap_reason)
    building_section.footprint_area = _cap_field_confidence(building_section.footprint_area, plot_conf_level, _plot_cap_reason)
    plot_section.width = plot_width
    plot_section.depth = plot_depth
    plot_section.area = plot_area

    # --- evidence reconciliation conflicts + physical consistency ---------------
    conflicts = []
    for vf in (
        plot_width,
        plot_depth,
        plot_area,
        building_section.width,
        building_section.depth,
        building_section.footprint_area,
        coverage,
        far,
        road_width,
        setbacks_section.front,
        setbacks_section.rear,
        setbacks_section.left,
        setbacks_section.right,
    ):
        if vf.conflict is not None:
            conflicts.append(vf.conflict)
    conflicts.extend(check_physical_consistency(plot_section, building_section, setbacks_section))

    # --- overall confidence note ------------------------------------------------
    labeled_areas = find_labeled_areas(extraction.text_evidence)
    labeled_area_note = (
        "; ".join(f"{a.label}={a.value}{a.unit or ''}" for a in labeled_areas)
        if labeled_areas
        else "none found"
    )
    rejected_buildings = [f for f in filtered_buildings if not f.kept]
    note_parts = [
        f"Plot: {plot_note} {plot_conf_reason}",
        f"Scale: {scale_est.reason}",
        f"Front-side: {fsr_page.reasoning}",
        f"Building blocks kept: {len(survivors)}, rejected: {len(rejected_buildings)}.",
        f"Distinct labeled areas in source text (kept separate from footprint_area): {labeled_area_note}.",
    ]
    if len({p for p in (page,) }) and page is not None:
        note_parts.append(f"Resolved from page {page}.")
    overall_note = " | ".join(note_parts)

    plan = NormalizedPlan(
        plan_id=plan_id,
        source_document_id=extraction.document_id,
        plot=plot_section,
        building=building_section,
        road=road_section,
        setbacks=setbacks_section,
        coverage=coverage,
        far=far,
        conflicts=conflicts,
        overall_confidence_note=overall_note,
    )

    # Building use / development-area class / height-excluding-stilt: a
    # deterministic text-based read of the plan's own title block/property
    # table, not a geometric measurement. Works identically for PDF native/
    # OCR text and DXF TEXT/MTEXT entities (both land in
    # `extraction.text_evidence`). Set here, ahead of the Vision-metadata
    # block below, so a real printed label always wins over metadata; an
    # unprinted value is left for the Vision-metadata fallback (harmless if
    # it never fires) and otherwise correctly stays MISSING -- development
    # area in particular is often a location-based zoning classification a
    # drawing does not always carry at all.
    ctx = extract_plan_context(extraction.text_evidence)
    if ctx.building_use_raw is not None:
        normalized_use = NormalizedPlan._normalize_building_use(ctx.building_use_raw)
        if normalized_use:
            plan.building_use = ValueField[str](
                value=normalized_use,
                confidence=Confidence(
                    level=ConfidenceLevel.HIGH,
                    reason=f"Building-use label printed on the plan: '{ctx.building_use_source_text}'.",
                ),
                source="plan text (title block / property table)",
            )
    if ctx.development_area is not None:
        plan.development_area = ValueField[str](
            value=ctx.development_area,
            confidence=Confidence(
                level=ConfidenceLevel.HIGH,
                reason=f"Development-area class printed on the plan: '{ctx.development_area_source_text}'.",
            ),
            source="plan text (title block / property table)",
        )
    if ctx.height_excluding_stilt_m is not None:
        plan.building_height_excluding_stilt = ValueField[float](
            value=round(ctx.height_excluding_stilt_m, 4),
            normalized_value=UnitValue(magnitude=round(ctx.height_excluding_stilt_m, 4), unit="m"),
            confidence=Confidence(
                level=ConfidenceLevel.HIGH,
                reason=f"Explicit 'height excluding stilt' label printed on the plan: "
                f"'{ctx.height_excluding_stilt_source_text}'.",
            ),
            source="plan text (explicit label)",
        )

    # Extended semantic information from the Vision layer. It is kept separate
    # from core rule-engine fields so metadata can never affect compliance.
    vision_doc = _vision_document(extraction)
    if vision_doc is not None:
        meta = {}
        for vp in vision_doc.pages:
            for k, val in (vp.metadata or {}).items():
                if val not in (None, "", [], {}):
                    meta.setdefault(k, val)
        plan.metadata = meta
        # Promote explicitly emitted vision metadata into rule-context fields.
        # This is extraction, not compliance logic: the deterministic engine
        # only consumes the resulting ValueFields and never asks the vision/LLM
        # layer to choose a legal threshold.
        raw_use = meta.get("building_use", meta.get("building_type"))
        normalized_use = NormalizedPlan._normalize_building_use(raw_use) if isinstance(raw_use, str) else None
        if plan.building_use is None and normalized_use:
            plan.building_use = ValueField[str](
                value=normalized_use,
                confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="Building-use classification from vision metadata."),
                source="vision metadata",
            )
        raw_area = meta.get("development_area")
        if plan.development_area is None and isinstance(raw_area, str) and raw_area.strip().upper() in {"A", "B", "C"}:
            plan.development_area = ValueField[str](
                value=raw_area.strip().upper(),
                confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="Development-area class from vision metadata."),
                source="vision metadata",
            )
        raw_height_ex_stilt = meta.get("building_height_excluding_stilt")
        if plan.building_height_excluding_stilt is None and isinstance(raw_height_ex_stilt, (int, float)):
            plan.building_height_excluding_stilt = ValueField[float](
                value=float(raw_height_ex_stilt),
                normalized_value=UnitValue(magnitude=float(raw_height_ex_stilt), unit="m"),
                confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="Height excluding stilt from vision metadata."),
                source="vision metadata",
            )
        floor_to_floor_height_m = None
        floor_to_floor_height_conf = -1.0
        stilt_height_m = None
        stilt_height_conf = -1.0
        plinth_height_m = None
        plinth_height_conf = -1.0
        parapet_height_m = None
        parapet_height_conf = -1.0
        height_excluding_stilt_m = None
        height_excluding_stilt_conf = -1.0
        for vp in vision_doc.pages:
            for area in vp.areas:
                typ = (area.type or "").upper()
                if area.value is None or "FLOOR" not in typ:
                    continue
                key = area.region_id or f"page_{vp.page_number}"
                unit = (area.unit or "m2").lower().replace("sqm", "m2").replace("sq.m", "m2")
                if unit == "ft2":
                    value = float(area.value) * 0.09290304
                    unit = "m2"
                else:
                    value = float(area.value)
                plan.floor_areas[key] = ValueField[float](
                    value=round(value, 4),
                    normalized_value=UnitValue(magnitude=round(value, 4), unit=unit),
                    confidence=Confidence(level=confidence_level_from_score(area.confidence), reason=f"Vision area evidence: {area.evidence or typ}"),
                    source="vision semantic area",
                )
            for dim in vp.dimensions:
                if dim.value is None or "HEIGHT" not in (dim.type or "").upper():
                    continue
                unit = (dim.unit or "m").lower()
                factor = {"m": 1, "meter": 1, "metre": 1, "mm": .001, "cm": .01, "ft": .3048, "feet": .3048}.get(unit)
                if not factor:
                    continue
                val = float(dim.value) * factor
                dim_type = (dim.type or "").upper()
                if dim_type in ("BUILDING_HEIGHT", "PROPOSED_BUILDING_HEIGHT"):
                    # An explicit, directly-stated overall/proposed building
                    # height is the strongest possible evidence -- always
                    # prefer it over anything derived from floor count.
                    # PROPOSED_BUILDING_HEIGHT is the same real-world
                    # quantity as BUILDING_HEIGHT, just printed under a
                    # different label on the sheet -- never conflated with
                    # REGULATORY_MAX_HEIGHT below, which is a legal ceiling,
                    # not a measurement of this building.
                    if plan.building_height_estimated is None or (dim.confidence > .8):
                        plan.building_height_estimated = ValueField[float](value=round(val, 4), normalized_value=UnitValue(magnitude=round(val, 4), unit="m"), confidence=Confidence(level=confidence_level_from_score(dim.confidence), reason=f"Vision height dimension evidence (explicit {dim_type.lower()})."), source="vision semantic height")
                elif dim_type == "FLOOR_HEIGHT":
                    if floor_to_floor_height_m is None or dim.confidence > floor_to_floor_height_conf:
                        floor_to_floor_height_m = val
                        floor_to_floor_height_conf = dim.confidence
                elif dim_type == "STILT_HEIGHT":
                    if stilt_height_m is None or dim.confidence > stilt_height_conf:
                        stilt_height_m = val
                        stilt_height_conf = dim.confidence
                elif dim_type == "PLINTH_HEIGHT":
                    if plinth_height_m is None or dim.confidence > plinth_height_conf:
                        plinth_height_m = val
                        plinth_height_conf = dim.confidence
                elif dim_type == "PARAPET_HEIGHT":
                    if parapet_height_m is None or dim.confidence > parapet_height_conf:
                        parapet_height_m = val
                        parapet_height_conf = dim.confidence
                elif dim_type == "HEIGHT_EXCLUDING_STILT":
                    if height_excluding_stilt_m is None or dim.confidence > height_excluding_stilt_conf:
                        height_excluding_stilt_m = val
                        height_excluding_stilt_conf = dim.confidence
                # REGULATORY_MAX_HEIGHT / REGULATORY_TEXT are intentionally
                # NOT handled by any branch above: a regulatory ceiling must
                # never feed building_height_estimated (or any other
                # measurement field) no matter how it's labeled -- see
                # HEIGHT_FOCUS_PROMPT's critical rule #2. They are still
                # collected below purely for evidence/audit purposes.
                elif dim_type in ("REGULATORY_MAX_HEIGHT", "REGULATORY_TEXT"):
                    plan.overall_confidence_note = (
                        (plan.overall_confidence_note + " " if plan.overall_confidence_note else "")
                        + f"Regulatory height text found on the plan (not used as a measurement): "
                        f"'{dim.evidence or dim.value}'."
                    )

        # Derive an estimated overall height ONLY when no explicit
        # BUILDING_HEIGHT annotation was found, and ONLY from evidence that
        # was actually read off the plan (an explicit FLOOR_HEIGHT dimension)
        # -- never a fabricated generic storey height (e.g. an assumed 3m).
        # floor_count must also be independently resolved (from vision floor
        # regions or a text label like "G+1"), not itself guessed.
        if (
            plan.building_height_estimated is None
            and floor_to_floor_height_m is not None
            and plan.building.floor_count is not None
            and plan.building.floor_count.value is not None
            and plan.building.floor_count.value > 0
        ):
            n_floors = plan.building.floor_count.value
            estimated = floor_to_floor_height_m * n_floors
            parts = [f"{n_floors} floor(s) x {floor_to_floor_height_m:.3f} m floor-to-floor height"]
            if stilt_height_m is not None:
                # A stilt level is a distinct labeled storey the sheet
                # explicitly called out (see HEIGHT_FOCUS_PROMPT), not one of
                # the generic repeated FLOOR_HEIGHT storeys -- added as its
                # own explicit component, never assumed present.
                estimated += stilt_height_m
                parts.append(f"+ {stilt_height_m:.3f} m stilt height")
            if plinth_height_m is not None:
                estimated += plinth_height_m
                parts.append(f"+ {plinth_height_m:.3f} m plinth height")
            if parapet_height_m is not None:
                estimated += parapet_height_m
                parts.append(f"+ {parapet_height_m:.3f} m parapet height")
            reason = (
                "Derived building height (no explicit overall height annotation found): "
                + " ".join(parts)
                + ". All components read directly from the plan's own vision-extracted "
                "dimensions/floor count -- no generic/assumed floor height was used."
            )
            plan.building_height_estimated = ValueField[float](
                value=round(estimated, 4),
                normalized_value=UnitValue(magnitude=round(estimated, 4), unit="m"),
                confidence=Confidence(
                    level=ConfidenceLevel.MEDIUM if floor_to_floor_height_conf >= .8 else ConfidenceLevel.LOW,
                    reason=reason,
                ),
                source="derived: floor_count x floor_to_floor_height (+ plinth/parapet if present)",
            )

        # An explicit HEIGHT_EXCLUDING_STILT reading is a DIFFERENT quantity
        # from building_height_estimated (see HEIGHT_FOCUS_PROMPT) -- only
        # ever set from a dimension the sheet itself explicitly labeled that
        # way, never derived/guessed. Vision evidence takes priority over the
        # plan-metadata fallback already handled above since it is grounded
        # in this specific sheet's own printed figure.
        if height_excluding_stilt_m is not None and (
            plan.building_height_excluding_stilt is None or height_excluding_stilt_conf > 0.8
        ):
            plan.building_height_excluding_stilt = ValueField[float](
                value=round(height_excluding_stilt_m, 4),
                normalized_value=UnitValue(magnitude=round(height_excluding_stilt_m, 4), unit="m"),
                confidence=Confidence(
                    level=confidence_level_from_score(height_excluding_stilt_conf),
                    reason="Vision height dimension evidence (explicit 'excluding stilt' label).",
                ),
                source="vision semantic height",
            )

    # Authoritative Phase-3.3+ output: if the independent CV result exists,
    # reconcile it directly with independent Vision evidence. This prevents
    # the legacy 400+ candidate pool from contaminating final values.
    old_plan = apply_final_agreement_to_plan(plan, extraction.independent_cv, vision_doc)
    return _apply_phase8_bridge(old_plan, extraction, vision_doc)


__all__ = ["build_normalized_plan"]
