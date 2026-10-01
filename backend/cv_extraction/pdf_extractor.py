"""
PDFHybridExtractor — Phase 2 implementation of `GeometryExtractor`.

Pipeline (see ARCHITECTURE.md's "PLAN PDF -> ... -> RAW EXTRACTION" stage):

    1. PDF-native vector + text extraction        (pdf_native.py)
    2. OCR fallback for scanned/insufficient pages (ocr_fallback.py)
    3. OpenCV raster geometry evidence             (opencv_geometry.py)
    4. Coordinate bookkeeping                       (coordinates.py)
    5. Dimension-candidate detection                (dimension_candidates.py)
    6. First-pass Plot/Building/Road candidates     (candidate_geometry.py)

Steps 1-5 build a `RawExtractionBundle` (raw_types.py) — the fully
detailed, page-accurate intermediate representation. Step 6 plus a final
folding step turn that into the Phase 1 `ExtractionResult`, which is the
ONLY thing `extract()` (the contract method) returns.

`extract_with_debug()` additionally returns the `RawExtractionBundle`,
for tests and `debug_overlay.py` — never for anything downstream of
extraction.

No filename/page/value hard-coding anywhere in this module: every
decision is a function of what was actually detected on the page.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.config import get_settings
from backend.cv_extraction import candidate_geometry, ocr_fallback, opencv_geometry, pdf_native
from backend.cv_extraction.coordinates import build_transform_record
from backend.cv_extraction.interfaces import GeometryExtractor
from backend.cv_extraction.raw_types import (
    CoordinateTransformRecord,
    DimensionCandidate,
    PageMetadata,
    RawExtractionBundle,
    RawLine,
    RawPolygon,
    RawRectangle,
    RawTextItem,
    SourceKind,
)
from backend.cv_extraction.dimension_candidates import detect_dimension_candidates
from backend.schemas.enums import DocumentType, SourceType
from backend.schemas.evidence import TextEvidence
from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import Dimension, SpatialRelation, SpatialRelationType
from backend.schemas.independent_measurements import IndependentCVResult
from backend.tools import bounded_execution
from backend.tools.logging_config import get_logger
from backend.vision_extraction import get_vision_extractor

logger = get_logger(__name__)

EXTRACTOR_NAME = "backend.cv_extraction.pdf_extractor.PDFHybridExtractor"
EXTRACTOR_VERSION = "0.1.0"

_SOURCE_KIND_TO_SOURCE_TYPE = {
    SourceKind.PDF_TEXT: SourceType.TEXT,
    SourceKind.OCR: SourceType.OCR,
}


_DEFAULT_UNIT_NOTE_RE = re.compile(
    r"ALL\s+DIMENSIONS?\s*(?:ARE|IS)?\s*IN\s+(METRE|METER|MILLIMETRE|MILLIMETER|MM|CENTIMETRE|CENTIMETER|CM|FEET|FOOT|FT)",
    re.IGNORECASE,
)
_DEFAULT_UNIT_NORMALIZATION = {
    "METRE": "m", "METER": "m",
    "MILLIMETRE": "mm", "MILLIMETER": "mm", "MM": "mm",
    "CENTIMETRE": "cm", "CENTIMETER": "cm", "CM": "cm",
    "FEET": "ft", "FOOT": "ft", "FT": "ft",
}
# A plausible single real-world length for a plot/room/door/setback
# dimension on an architectural drawing, in whatever the sheet's default
# unit turns out to be — wide enough to cover a door width up through a
# large plot side, narrow enough to exclude area figures (sq.m/sq.ft
# values routinely run into the hundreds/thousands) and unrelated
# integers (survey numbers, sheet counts) that happen to share a text
# run with a real dimension.
_PLAUSIBLE_DEFAULT_UNIT_RANGE = (0.05, 60.0)


def _detect_default_dimension_unit(text_items: list[RawTextItem]) -> str | None:
    """
    Real Indian architectural drawings very commonly state the unit ONCE
    ("ALL DIMENSIONS ARE IN METRE") rather than per-label, then give bare
    numbers everywhere else. Phase 2's per-candidate unit regex has no
    way to see that page-level note, so those bare numbers were being
    dropped as `unit=unknown` — which starved scale estimation of
    samples (observed directly on a real plan: 221/225 dimension
    candidates had no per-label unit). This scans the page text once for
    that convention and returns a normalized unit ("m"/"mm"/"cm"/"ft")
    to use as a FALLBACK — never overriding an explicit per-label unit.
    """
    for item in text_items:
        m = _DEFAULT_UNIT_NOTE_RE.search(item.text or "")
        if m:
            return _DEFAULT_UNIT_NORMALIZATION.get(m.group(1).upper())
    return None


def _fallback_unit_if_plausible(cand: DimensionCandidate, default_unit: str | None) -> str | None:
    """
    Apply `default_unit` to a candidate that had no explicit unit of its
    own, but ONLY when its raw text actually looks like a single
    real-world length reading (contains a decimal point — plot/room/door
    dimensions on these drawings are essentially always given to two
    decimal places, e.g. "3.35", "0.91" — and the parsed magnitude falls
    in a plausible length range). This deliberately excludes bare
    integers (survey/sheet numbers, door/window code labels like "D2"
    where a stray digit was picked up) and large bare numbers (area
    figures in sq.m/sq.ft, which share the same text blocks as real
    dimensions on these drawings) from being mistaken for lengths.
    """
    if default_unit is None:
        return None
    if "." not in (cand.raw_text or ""):
        return None
    low, high = _PLAUSIBLE_DEFAULT_UNIT_RANGE
    if cand.numeric_value is None or not (low <= cand.numeric_value <= high):
        return None
    return default_unit


def _point_to_line_distance_pts(px: float, py: float, line) -> float:
    ax, ay = line.start.x, line.start.y
    bx, by = line.end.x, line.end.y
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def _best_dimension_geometry(cand: DimensionCandidate, lines: list[RawLine], settings):
    """
    FIX #4 (phase3.1): select the strongest geometry-line association for
    a dimension candidate, not just `nearby_geometry_ids[0]` blindly.

    `cand.nearby_geometry_ids` is already sorted best-association-first
    (distance + orientation + length scoring — see
    `dimension_candidates._nearby_geometry_ids`). Here we additionally
    gate on a minimum confidence derived from actual distance: if even
    the best-scored candidate line is too far to be trustworthy, leave
    the association unresolved (`geometry = None`) rather than inventing
    one. Since the list is best-first, if the top candidate fails the
    gate, none of the rest would pass either.
    """
    if not cand.nearby_geometry_ids:
        return None
    window = settings.dimension_association_search_window_pts
    min_conf = settings.dimension_association_min_confidence
    idx = cand.nearby_geometry_ids[0]
    if idx < 0 or idx >= len(lines):
        return None
    line = lines[idx].line
    center = cand.bounding_box.center
    distance = _point_to_line_distance_pts(center.x, center.y, line)
    confidence = max(0.0, 1.0 - (distance / window)) if window > 0 else 0.0
    if confidence < min_conf:
        return None
    return line


class PDFHybridExtractor(GeometryExtractor):
    """
    Hybrid PDF extractor: vector-native where possible, OCR fallback where
    necessary, OpenCV for raster geometry evidence throughout.

    A single instance handles vector PDFs, raster/scanned PDFs, and mixed
    documents (some pages vector, some scanned) transparently — the
    per-page decision is made internally, never by the caller.
    """

    def __init__(
        self,
        ocr_dpi: float | None = None,
        opencv_dpi: float | None = None,
        *,
        enable_vision: bool | None = None,
    ) -> None:
        settings = get_settings()
        # Validation mode passes enable_vision=False so this extractor is a
        # genuinely independent CV/native/OCR evidence producer.
        self.enable_vision = settings.vision_enabled if enable_vision is None else enable_vision
        requested_ocr_dpi = ocr_dpi or ocr_fallback.DEFAULT_OCR_DPI
        requested_opencv_dpi = opencv_dpi or ocr_fallback.DEFAULT_OCR_DPI
        # FIX #10 (phase3.1): bounded processing — raster DPI is a direct
        # multiplier on OpenCV/OCR cost; cap it rather than trusting the
        # caller (or the default) blindly.
        self.ocr_dpi = min(requested_ocr_dpi, settings.max_raster_dpi)
        self.opencv_dpi = min(requested_opencv_dpi, settings.max_raster_dpi)

    def supported_document_type(self) -> DocumentType:
        # Declared primary type for registry purposes; this extractor also
        # transparently handles RASTER_PDF / mixed documents internally.
        return DocumentType.VECTOR_PDF

    # -- Contract method -----------------------------------------------------

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        settings = get_settings()
        try:
            result, _bundle = bounded_execution.run_with_timeout(
                lambda: self.extract_with_debug(document_path, document_id),
                timeout_seconds=settings.extraction_timeout_seconds,
                operation=f"extraction of document '{document_id}'",
            )
            # Keep the independent CV/native/OCR result alongside the legacy raw extraction.
            # The final resolver uses this field so the historical global candidate pool can
            # never overwrite the validated site-plan measurements.
            try:
                from backend.cv_extraction.site_plan import extract_independent_cv
                result.independent_cv = extract_independent_cv(document_path, document_id)
            except Exception as exc:
                # This must NOT be left as `None`: a `None` independent_cv is
                # indistinguishable from "never attempted" to the final-fusion
                # layer, and previously caused it to bypass reconciliation
                # entirely and ship the pre-fusion (Vision-first-pass) plan
                # untouched. Recording an explicit FAILED result lets fusion
                # still run against whatever other evidence exists, and
                # surfaces the failure instead of hiding it.
                logger.warning("independent CV validation path failed for %s: %s", document_id, exc)
                result.warnings.append(f"independent CV validation path failed: {exc}")
                result.independent_cv = IndependentCVResult(
                    document_id=document_id,
                    status="FAILED",
                    error=str(exc),
                    warnings=[f"independent CV validation path failed: {exc}"],
                )
            return result
        except bounded_execution.ExtractionTimeoutError as exc:
            logger.warning(str(exc))
            return ExtractionResult(
                document_id=document_id,
                document_type=DocumentType.RASTER_PDF,
                page_count=1,
                warnings=[
                    f"TIMEOUT: {exc}. Extraction was aborted with no candidates/dimensions "
                    "returned — this is an explicit failure, not a silently degraded result."
                ],
                extractor_name=EXTRACTOR_NAME,
                extractor_version=EXTRACTOR_VERSION,
                # Same reasoning as the FAILED case above: an explicit status
                # rather than a bare `None`, so downstream fusion knows CV
                # evidence was never actually produced for this document.
                independent_cv=IndependentCVResult(
                    document_id=document_id,
                    status="FAILED",
                    error=f"Extraction timed out: {exc}",
                ),
            )

    # -- Debug/testing entry point (not part of the Phase 1 contract) -------

    def extract_with_debug(
        self, document_path: Path, document_id: str
    ) -> tuple[ExtractionResult, RawExtractionBundle]:
        bundle = self._extract_raw(document_path, document_id)
        result = self._fold(document_id, bundle)
        return result, bundle

    # -- Raw extraction --------------------------------------------------

    def _extract_raw(self, document_path: Path, document_id: str) -> RawExtractionBundle:
        settings = get_settings()
        doc = pdf_native.open_document(document_path)
        warnings: list[str] = []

        page_metadata: list[PageMetadata] = []
        text_items: list[RawTextItem] = []
        ocr_items: list[RawTextItem] = []
        lines: list[RawLine] = []
        rectangles: list[RawRectangle] = []
        polygons: list[RawPolygon] = []
        transforms: list[CoordinateTransformRecord] = []

        try:
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                meta = pdf_native.extract_page_metadata(page, page_number)

                page_text_items = pdf_native.extract_text_items(page, page_number)
                page_lines, page_rects, page_polys = pdf_native.extract_vector_geometry(
                    page, page_number
                )
                has_native_text = pdf_native.has_sufficient_native_text(page_text_items)

                meta = meta.model_copy(
                    update={
                        "has_native_text": has_native_text,
                        "vector_object_count": len(page_lines) + len(page_rects) + len(page_polys),
                    }
                )

                page_ocr_items: list[RawTextItem] = []
                is_scanned = ocr_fallback.needs_ocr(
                    has_native_text, sum(len(t.text.strip()) for t in page_text_items)
                )

                # OpenCV pass always runs (evidence source), OCR only as fallback.
                raster_image = None
                try:
                    raster_image = ocr_fallback.rasterize_page(page, dpi=self.opencv_dpi)
                except Exception as exc:  # pragma: no cover - environment issue
                    warnings.append(f"page {page_number}: rasterization failed: {exc}")

                if is_scanned and raster_image is not None:
                    try:
                        # Orientation-aware: also reads the page sideways when
                        # its text runs vertically (a rotated/landscape sheet
                        # stored as an unrotated portrait page), same as
                        # site_plan.extract_independent_cv and
                        # region_detection.detect_regions already do -- this
                        # was previously the plain upright-only ocr_page here,
                        # which silently mis-OCR'd this exact case (audit
                        # finding: BUILDCHECK_FORENSIC_AUDIT.md Sec 4.1/5.4).
                        # Always returns at least what the upright pass found,
                        # so this is a safe drop-in replacement.
                        page_ocr_items = ocr_fallback.ocr_page_any_orientation(
                            raster_image, page_number, dpi=self.opencv_dpi
                        )
                    except Exception as exc:  # pragma: no cover
                        warnings.append(f"page {page_number}: OCR failed: {exc}")
                    if not page_ocr_items:
                        warnings.append(
                            f"page {page_number}: no native text and OCR produced no results"
                        )

                cv_lines: list[RawLine] = []
                cv_polys: list[RawPolygon] = []
                cv_rects: list[RawRectangle] = []
                if raster_image is not None:
                    try:
                        evidence = opencv_geometry.geometry_evidence_for_page(
                            raster_image, page_number, dpi=self.opencv_dpi
                        )
                        cv_lines = evidence["lines"]
                        cv_polys = evidence["polygons"]
                        cv_rects = evidence["rectangles"]
                        # FIX #10 (phase3.1): bound the number of raster
                        # primitives carried forward — real scanned sheets
                        # can produce thousands of tiny Hough segments /
                        # contours (hatching, noise). Keep the longest
                        # lines and largest contours, which are the most
                        # plausible dimension/boundary evidence anyway.
                        if len(cv_lines) > settings.max_opencv_lines:
                            cv_lines = sorted(
                                cv_lines, key=lambda ln: ln.line.length, reverse=True
                            )[: settings.max_opencv_lines]
                        if len(cv_polys) > settings.max_opencv_contours:
                            cv_polys = sorted(
                                cv_polys,
                                key=lambda p: (p.area_pts2 if p.area_pts2 is not None else p.polygon.area),
                                reverse=True,
                            )[: settings.max_opencv_contours]
                    except Exception as exc:  # pragma: no cover - environment issue
                        warnings.append(f"page {page_number}: OpenCV evidence extraction failed: {exc}")

                meta = meta.model_copy(update={"is_scanned": is_scanned})
                page_metadata.append(meta)

                text_items.extend(page_text_items)
                text_items.extend(page_ocr_items)
                ocr_items.extend(page_ocr_items)
                lines.extend(page_lines)
                lines.extend(cv_lines)
                rectangles.extend(page_rects)
                rectangles.extend(cv_rects)
                polygons.extend(page_polys)
                polygons.extend(cv_polys)

                transforms.append(
                    build_transform_record(
                        page=page_number,
                        dpi=self.opencv_dpi,
                        rotation_degrees=meta.rotation_degrees,
                        page_width_pts=meta.width_pts,
                        page_height_pts=meta.height_pts,
                        provisional_points_per_metre=settings.default_points_per_metre,
                    )
                )

                if not has_native_text and not page_ocr_items:
                    warnings.append(
                        f"page {page_number}: extraction incomplete — no vector text and no usable OCR text"
                    )
        finally:
            doc.close()

        vision_pages = []
        if self.enable_vision:
            try:
                vision_result = get_vision_extractor().analyze_pdf(document_path)
                vision_pages = vision_result.pages
                warnings.extend(vision_result.warnings)
            except Exception as exc:
                # Vision is an augmentation layer. Existing PDF/CV/OCR extraction
                # must continue to work if the optional model is unavailable.
                warnings.append(f"vision extraction unavailable: {exc}")

        dimension_candidates: list[DimensionCandidate] = detect_dimension_candidates(
            text_items, lines, proximity_pts=settings.dimension_association_search_window_pts,
            max_lines_considered=settings.max_dimension_association_lines,
        )

        geometry_evidence_combined: list = [*lines, *rectangles, *polygons]

        bundle = RawExtractionBundle(
            document_id=document_id,
            pages=page_metadata,
            text_evidence=text_items,
            geometry_evidence=geometry_evidence_combined,
            lines=lines,
            rectangles=rectangles,
            polygons=polygons,
            dimensions_candidates=dimension_candidates,
            ocr_evidence=ocr_items,
            page_metadata=page_metadata,
            coordinate_transform=transforms,
            warnings=warnings,
            vision_pages=vision_pages,
        )
        return bundle

    # -- Folding raw -> Phase 1 contract ---------------------------------

    def _fold(self, document_id: str, bundle: RawExtractionBundle) -> ExtractionResult:
        document_type = self._infer_document_type(bundle)

        text_evidence = [
            TextEvidence(
                source_type=_SOURCE_KIND_TO_SOURCE_TYPE.get(item.source, SourceType.TEXT),
                raw_text=item.text,
                page=item.page,
                bounding_box=item.bounding_box,
                ocr_confidence=item.ocr_confidence,
            )
            for item in bundle.text_evidence
        ]

        settings = get_settings()
        default_unit = _detect_default_dimension_unit(bundle.text_evidence)
        dimensions = [
            Dimension(
                label=cand.raw_text,
                magnitude=cand.numeric_value,
                unit=(cand.unit_hint or _fallback_unit_if_plausible(cand, default_unit) or "unknown"),
                geometry=_best_dimension_geometry(cand, bundle.lines, settings),
                text_bounding_box=cand.bounding_box,
                page=cand.page,
                # Preserve the candidate's own computed confidence instead
                # of discarding it here (audit finding:
                # BUILDCHECK_FORENSIC_AUDIT.md Sec 4.1/5.5 -- this score was
                # previously computed in dimension_candidates.py and then
                # lost at this exact fold, leaving every Dimension in
                # ExtractionResult.dimensions confidence-blind).
                confidence=cand.confidence,
            )
            for cand in bundle.dimensions_candidates
        ]

        page_areas = {p.page_number: p.width_pts * p.height_pts for p in bundle.page_metadata}
        page_dims = {p.page_number: (p.width_pts, p.height_pts) for p in bundle.page_metadata}
        transforms_by_page = {t.page: t for t in bundle.coordinate_transform}

        plot_candidates, building_candidates = candidate_geometry.build_plot_and_building_candidates(
            document_id, bundle.polygons, page_areas, transforms_by_page, page_dims
        )
        road_candidates = candidate_geometry.build_road_candidates(
            document_id, bundle.polygons, bundle.rectangles, bundle.text_evidence, transforms_by_page
        )

        spatial_relations = self._infer_spatial_relations(plot_candidates, building_candidates)

        return ExtractionResult(
            document_id=document_id,
            document_type=document_type,
            page_count=len(bundle.page_metadata) or 1,
            plot_candidates=plot_candidates,
            building_candidates=building_candidates,
            road_candidates=road_candidates,
            dimensions=dimensions,
            text_evidence=text_evidence,
            spatial_relations=spatial_relations,
            vision_pages=bundle.vision_pages,
            warnings=bundle.warnings,
            extractor_name=EXTRACTOR_NAME,
            extractor_version=EXTRACTOR_VERSION,
        )

    @staticmethod
    def _infer_document_type(bundle: RawExtractionBundle) -> DocumentType:
        if not bundle.page_metadata:
            return DocumentType.RASTER_PDF
        any_native = any(p.has_native_text for p in bundle.page_metadata)
        return DocumentType.VECTOR_PDF if any_native else DocumentType.RASTER_PDF

    @staticmethod
    def _infer_spatial_relations(plot_candidates, building_candidates) -> list[SpatialRelation]:
        relations: list[SpatialRelation] = []
        for plot in plot_candidates:
            if plot.geometry is None or plot.geometry.bounding_box is None:
                continue
            plot_page = plot.geometry.source_page
            for building in building_candidates:
                if building.geometry is None or building.geometry.bounding_box is None:
                    continue
                if building.geometry.source_page != plot_page:
                    continue
                if plot.geometry.bounding_box.intersects(building.geometry.bounding_box):
                    relations.append(
                        SpatialRelation(
                            subject_id=plot.id,
                            object_id=building.id,
                            relation=SpatialRelationType.CONTAINS,
                            notes="plot candidate bounding box contains/overlaps building candidate",
                        )
                    )
        return relations


__all__ = ["PDFHybridExtractor", "EXTRACTOR_NAME", "EXTRACTOR_VERSION"]
