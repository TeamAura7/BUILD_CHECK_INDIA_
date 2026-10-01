from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from backend.config import get_settings
from backend.tools.logging_config import get_logger
from backend.schemas.geometry import BoundingBox
from backend.vision_extraction.page_renderer import render_pdf_pages, render_pdf_region
from backend.vision_extraction.prompts import (
    AREA_STATEMENT_FOCUS_PROMPT,
    ARCHITECTURAL_PLAN_PROMPT,
    HEIGHT_FOCUS_PROMPT,
    SITE_PLAN_FOCUS_PROMPT,
)
from backend.vision_extraction.spatial import vision_bbox_to_page_points
from backend.schemas.vision import VisionDocumentResult, VisionPageResult

# Grounding tolerance (PDF points) between a vision-claimed dimension's bbox
# and the nearest matching native-text span with the same value. Generous,
# since VLM bboxes are approximate -- this only needs to reject "value exists
# somewhere totally unrelated on the sheet" (e.g. a floor-height label being
# reused to justify a fabricated setback), not demand pixel-perfect alignment.
_GROUNDING_PROXIMITY_TOLERANCE_PT = 250.0

logger = get_logger(__name__)


class BaseArchitecturalPlanExtractor(ABC):
    """
    Shared plumbing for optional local vision-language semantic
    extractors (page rendering, prompt formatting, JSON-from-model-text
    parsing, page-loop error handling). Concrete backends only need to
    implement `_load()` (lazy model/processor load) and
    `_generate_raw_response()` (one image + prompt -> raw model text).

    This class is intentionally lazy: importing the backend does not
    require the large vision-model dependencies unless vision extraction
    is enabled and a concrete backend is actually instantiated.
    """

    #: Overridden per-backend; used when settings.vision_model_name is
    #: not explicitly set, so switching backends doesn't require also
    #: remembering to update a hardcoded model name in config.
    DEFAULT_MODEL_NAME: str = ""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.vision_model_name or self.DEFAULT_MODEL_NAME

    @abstractmethod
    def _load(self) -> None:
        """Lazily import dependencies and load the model/processor. Must be idempotent."""

    @abstractmethod
    def _generate_raw_response(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        """Run the model on one rendered page image and return its raw text response."""

    @staticmethod
    def _native_number_spans_by_page(
        pdf_path: Path,
    ) -> dict[int, tuple[list[tuple[str, tuple[float, float]]], float, float]]:
        """
        Pull every numeric-looking text span out of the PDF's *native* text
        layer, keyed by 1-based page number, as (token, (center_x, center_y))
        in PDF point space, alongside that page's (width_pts, height_pts).
        Used as a grounding check against vision output. Returns an empty
        dict (grounding skipped everywhere) if the page can't be read
        natively (e.g. a scanned page with no text layer) -- this is a
        "don't trust an ungrounded number" safeguard, not a substitute for
        OCR, so it must never be used to wipe out a scanned page's results.
        """
        try:
            import fitz  # type: ignore
        except ImportError:
            return {}

        number_re = re.compile(r"\d+\.?\d*")
        spans_by_page: dict[int, tuple[list[tuple[str, tuple[float, float]]], float, float]] = {}
        try:
            with fitz.open(pdf_path) as doc:
                for page_index in range(doc.page_count):
                    page = doc.load_page(page_index)
                    spans: list[tuple[str, tuple[float, float]]] = []
                    for block in page.get_text("dict").get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                for token in number_re.findall(span.get("text", "")):
                                    x0, y0, x1, y1 = span["bbox"]
                                    spans.append((token, ((x0 + x1) / 2, (y0 + y1) / 2)))
                    spans_by_page[page_index + 1] = (spans, float(page.rect.width), float(page.rect.height))
        except Exception:
            logger.exception("native text extraction for vision grounding failed")
            return {}
        return spans_by_page

    @staticmethod
    def _value_tokens(value: float) -> set[str]:
        candidates = {f"{value:.2f}", f"{value:.1f}", f"{value:g}"}
        if 0 < value < 1:
            # Common "leading-dot" convention on these drawings (0.47 -> ".47").
            candidates.add(f"{value:.2f}"[1:])
            candidates.add(f"{value:.1f}"[1:])
        return candidates

    @staticmethod
    def _model_bbox_to_page_points(
        bbox: list[float] | None, page_width_pts: float, page_height_pts: float
    ) -> tuple[float, float] | None:
        """
        Convert a vision-model bbox to a (center_x, center_y) point in PDF
        page space. Thin wrapper around the single shared, auto-detecting
        conversion in `vision_extraction.spatial.vision_bbox_to_page_points`
        (previously this method had its own private copy of the same
        logic, which drifted from `spatial.py`'s -- see that function's
        docstring for the full explanation of the coordinate-space bug
        this guards against).
        """
        box = vision_bbox_to_page_points(bbox, page_width_pts, page_height_pts)
        if box is None:
            return None
        return box.center.x, box.center.y

    @classmethod
    def _is_item_grounded(
        cls,
        item: dict[str, Any],
        native_spans: list[tuple[str, tuple[float, float]]],
        page_width_pts: float,
        page_height_pts: float,
    ) -> bool:
        """
        Hard gate: does this value appear ANYWHERE in the page's real text
        layer at all? This alone is what reliably catches a fully invented
        value (nothing on the page prints "1.96" anywhere) without false-
        negatives on genuine values.

        NOTE: a stricter, bbox-proximity version was tried here (does the
        value appear *near* the location the model claims) and rejected --
        see `_model_bbox_to_page_points`'s docstring. The model's bbox
        coordinates are on a normalized 0-1000 grid (not raw pixels as the
        prompt requests), and even after correcting for that, the model's
        self-reported bbox positions aren't precise enough to use as a hard
        gate: it dropped genuinely correct dimensions while, by
        coincidence, keeping one of the fabricated setbacks (a real "3.00"
        floor-height label elsewhere on the sheet happened to fall inside
        the tolerance radius of the claimed setback bbox). Presence-anywhere
        is the reliable, low-false-negative check; true proximity grounding
        would need better-calibrated model bboxes or a second detector to
        anchor against, not a bigger tolerance.
        """
        value = item.get("value")
        evidence = str(item.get("evidence") or "")

        # Recover common SmolVLM failure mode: evidence contains the real
        # number (e.g. "9.14") but the numeric field is emitted as 0.0.
        if (value is None or float(value) == 0.0) and evidence:
            m = re.search(r"\d+(?:\.\d+)?", evidence)
            if m:
                value = float(m.group())
                item["value"] = value

        if value is None:
            # Not a numeric claim (e.g. a region record) -- grounding is
            # moot, and this is not a "confirmed independently" signal.
            item["grounded"] = None
            return True
        if not native_spans:
            # No native text layer exists on this page at all (scanned/
            # raster page), so this value's presence could not be checked
            # one way or the other -- keep it (Vision must remain usable on
            # scanned pages), but record that this specifically was NOT an
            # independent confirmation, only a trivial pass. Downstream
            # confidence assignment must not treat this the same as a value
            # that was actually found printed on the sheet (see
            # VisionDimension.grounded / VisionArea.grounded).
            item["grounded"] = False
            return True
        wanted = cls._value_tokens(float(value))
        found = any(token in wanted for token, _center in native_spans)
        item["grounded"] = found
        return found

    def _ground_against_native_text(
        self,
        payload: dict[str, Any],
        native_spans: list[tuple[str, tuple[float, float]]],
        page_width_pts: float = 0.0,
        page_height_pts: float = 0.0,
    ) -> dict[str, Any]:
        """
        Drop any dimension/area whose numeric value doesn't appear, near the
        bbox it's claimed at, in the PDF's native text layer -- this is what
        catches a VLM inventing a plausible-looking value (e.g. a "typical"
        3.00m setback) that either isn't printed anywhere on the sheet, or
        only exists as an unrelated label elsewhere (e.g. a floor-height
        dimension being repurposed to justify a fabricated setback).
        Regions are left alone (they aren't numeric claims in the same way).
        """
        warnings = list(payload.get("warnings") or [])
        for collection in ("dimensions", "areas"):
            kept = []
            for item in payload.get(collection, []) or []:
                if self._is_item_grounded(item, native_spans, page_width_pts, page_height_pts):
                    kept.append(item)
                else:
                    warnings.append(
                        f"Dropped ungrounded {collection[:-1]} "
                        f"{item.get('type')}={item.get('value')}: no matching text "
                        "found anywhere in the PDF's native text layer "
                        "(likely hallucinated)."
                    )
            payload[collection] = kept
        payload["warnings"] = warnings
        return payload

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()

        # Defensive stripping for reasoning/"thinking" models that inline
        # chain-of-thought in the response wrapped in <think>...</think>
        # (e.g. Groq's qwen/qwen3.6-27b in 'raw' reasoning_format, or any
        # other reasoning-capable model/provider that does this by
        # default). The 'api' backend requests 'hidden' reasoning_format
        # by default specifically to avoid this, but this stays as a
        # second line of defense for providers/overrides where a
        # <think> block still shows up in content.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def analyze_image(
        self,
        image_path: Path,
        page_number: int,
        native_spans: list[tuple[str, tuple[float, float]]] | None = None,
        page_width_pts: float = 0.0,
        page_height_pts: float = 0.0,
        *,
        ground_against_native_text: bool = True,
        prompt_override: str | None = None,
        max_new_tokens: int | None = None,
    ) -> VisionPageResult:
        self._load()
        prompt = (prompt_override or ARCHITECTURAL_PLAN_PROMPT).replace("PAGE_NUMBER", str(page_number))
        print(f"  [page {page_number}] generating response (this is the slow step on CPU)...", flush=True)
        response = self._generate_raw_response(image_path, prompt, max_new_tokens=max_new_tokens)
        print(f"  [page {page_number}] response received ({len(response)} chars), parsing JSON...", flush=True)
        payload = self._extract_json(response)
        # Grounding is an optional safety/production filter. Validation mode
        # deliberately disables it so Vision remains an independent evidence
        # source and CV/native text cannot silently delete a correct VLM value.
        if ground_against_native_text:
            payload = self._ground_against_native_text(
                payload, native_spans or [], page_width_pts, page_height_pts
            )
        return VisionPageResult.model_validate({
            **payload,
            "page_number": page_number,
            "raw_response": response,
            "page_width_pts": page_width_pts,
            "page_height_pts": page_height_pts,
        })

    @staticmethod
    def _crop_region_focus(
        pdf_path: Path,
        page_index: int,
        image_path: Path,
        region_bbox: BoundingBox,
        page_width_pts: float,
        page_height_pts: float,
        dpi: float,
        suffix: str,
    ) -> tuple[Path, float, float, float, float] | None:
        """Produce a focused crop of one semantic region, at `dpi`.

        Re-renders the region directly from the PDF's own vector page (see
        `page_renderer.render_pdf_region`) rather than cropping the already-
        rasterized full-page PNG -- a genuine resolution improvement, not
        an upscale of the same pixels, which matters for small printed text
        (setback decimals, area-statement table cells). Falls back to
        cropping the full-page render only if the direct re-render fails
        for some reason, so a focus pass still works, just without the
        resolution improvement, on whatever edge case caused that.
        """
        if page_width_pts <= 0 or page_height_pts <= 0:
            return None
        # Widened from 12% to 25% (min bumped 12pt -> 24pt). The region bbox
        # this crop is built from -- whether a VLM's first-pass guess or the
        # deterministic caption-anchored window -- is itself imprecise, and
        # a tight crop margin means a slightly-off bbox can crop a real
        # dimension label OUT of frame, or crop an unrelated adjacent label
        # IN -- and the focus pass then reads whatever pixels it was given
        # with full confidence, no way to tell from its output alone that
        # the crop itself was off.
        pad_x = max(24.0, region_bbox.width * 0.25)
        pad_y = max(24.0, region_bbox.height * 0.25)
        x0 = max(0.0, region_bbox.min_x - pad_x)
        y0 = max(0.0, region_bbox.min_y - pad_y)
        x1 = min(page_width_pts, region_bbox.max_x + pad_x)
        y1 = min(page_height_pts, region_bbox.max_y + pad_y)
        if x1 <= x0 or y1 <= y0:
            return None
        crop_path = image_path.with_name(f"{image_path.stem}_{suffix}.png")
        try:
            render_pdf_region(pdf_path, page_index, (x0, y0, x1, y1), crop_path, dpi=dpi)
            return crop_path, x0, y0, x1, y1
        except Exception:
            logger.warning(
                "high-resolution region re-render failed for %s (page %s); "
                "falling back to a crop of the full-page render", suffix, page_index,
                exc_info=True,
            )
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                w_px, h_px = img.size
                px0 = max(0, int(round(x0 / page_width_pts * w_px)))
                py0 = max(0, int(round(y0 / page_height_pts * h_px)))
                px1 = min(w_px, int(round(x1 / page_width_pts * w_px)))
                py1 = min(h_px, int(round(y1 / page_height_pts * h_px)))
                if px1 <= px0 or py1 <= py0:
                    return None
                img.crop((px0, py0, px1, py1)).save(crop_path, format="PNG")
                return crop_path, x0, y0, x1, y1
        except Exception:
            return None

    @staticmethod
    def _select_focus_bbox(
        first_pass_regions: list,
        detected_regions: list,
        region_type: str,
        page_width_pts: float,
        page_height_pts: float,
    ) -> tuple[BoundingBox, str] | None:
        """Pick the bbox to crop for a focused pass, cross-checking Vision against geometry.

        `detected_regions` comes from `backend.cv_extraction.region_detection`
        -- a deterministic, caption-anchored location that does not depend
        on a VLM correctly reading the whole page's layout. Confirmed live
        on `data/test_plans/PLAN5.pdf`: Vision's own first-pass SITE_PLAN
        bbox landed at the top-left of the page while the sheet's actual
        site plan is at the bottom-right (independently corroborated by
        `cv_extraction.site_plan.extract_independent_cv`'s own
        `site_plan_bbox_pts` and the sheet's area statement); the focused
        pass that used to blindly crop from Vision's bbox then read the
        ground-floor plan and reported it as the site plan. When the two
        sources disagree by more than a small overlap tolerance, the
        caption-anchored bbox wins; Vision's own bbox is used only when
        it's the sole source available (no printed caption to anchor to).
        """
        from backend.cv_extraction import region_detection

        vision_candidates = [
            vision_bbox_to_page_points(r.bbox, page_width_pts, page_height_pts)
            for r in first_pass_regions
            if (r.type or "").upper() == region_type
        ]
        vision_candidates = [b for b in vision_candidates if b is not None]
        vision_bbox = max(vision_candidates, key=lambda b: b.width * b.height) if vision_candidates else None

        detected = region_detection.best_match(detected_regions, region_type)
        det_bbox = detected.bbox_pts if detected is not None else None

        if det_bbox is not None and vision_bbox is not None:
            overlap = region_detection.iou(det_bbox, vision_bbox)
            if overlap < 0.10:
                return det_bbox, (
                    f"Vision's own first-pass {region_type} region bbox disagreed with the "
                    f"caption-anchored location (IoU={overlap:.2f}) -- using the caption-anchored "
                    "bbox, which does not depend on the VLM reading the whole page layout correctly."
                )
            return vision_bbox, f"Vision's {region_type} bbox agrees with the caption-anchored location (IoU={overlap:.2f})."
        if det_bbox is not None:
            return det_bbox, f"No Vision {region_type} region bbox available; using the caption-anchored location."
        if vision_bbox is not None:
            return vision_bbox, f"No caption-anchored {region_type} location found (caption unreadable/absent); using Vision's own region bbox."
        return None

    @staticmethod
    def _remap_focus_bbox(
        bbox: list[float] | None,
        crop_bbox_pts: tuple[float, float, float, float],
        page_width_pts: float,
        page_height_pts: float,
    ) -> list[float] | None:
        if not bbox or len(bbox) != 4:
            return None
        cx0, cy0, cx1, cy1 = crop_bbox_pts
        cw, ch = max(cx1 - cx0, 1e-6), max(cy1 - cy0, 1e-6)
        x0, y0, x1, y1 = [float(v) for v in bbox]
        px0 = cx0 + min(x0, x1) / 1000.0 * cw
        py0 = cy0 + min(y0, y1) / 1000.0 * ch
        px1 = cx0 + max(x0, x1) / 1000.0 * cw
        py1 = cy0 + max(y0, y1) / 1000.0 * ch
        return [
            max(0.0, min(1000.0, px0 / page_width_pts * 1000.0)),
            max(0.0, min(1000.0, py0 / page_height_pts * 1000.0)),
            max(0.0, min(1000.0, px1 / page_width_pts * 1000.0)),
            max(0.0, min(1000.0, py1 / page_height_pts * 1000.0)),
        ]

    def _focused_region_pass(
        self,
        first_pass: VisionPageResult,
        pdf_path: Path,
        page_index: int,
        image_path: Path,
        page_number: int,
        page_width_pts: float,
        page_height_pts: float,
        detected_regions: list,
        region_type: str,
        prompt: str,
        max_new_tokens: int,
        render_dpi: float,
    ) -> VisionPageResult | None:
        """Run one region-specific high-resolution focused Vision pass.

        Generalizes what used to be the SITE_PLAN-only independent focus
        pass to any semantic region type (SITE_PLAN, AREA_STATEMENT,
        ELEVATION, SECTION, ...): pick a grounded bbox for that region
        (`_select_focus_bbox`), re-render just that region at `render_dpi`
        (`_crop_region_focus`), then run `prompt` against the crop alone so
        the model reads that region's own pixels instead of the whole,
        busy, low-resolution sheet.
        """
        if page_width_pts <= 0 or page_height_pts <= 0:
            return None
        selection = self._select_focus_bbox(
            first_pass.regions, detected_regions, region_type, page_width_pts, page_height_pts
        )
        if selection is None:
            return None
        region_bbox, note = selection
        crop = self._crop_region_focus(
            pdf_path, page_index, image_path, region_bbox, page_width_pts, page_height_pts,
            render_dpi, region_type.lower(),
        )
        if crop is None:
            return None
        crop_path, x0, y0, x1, y1 = crop
        focus = self.analyze_image(
            crop_path,
            page_number,
            native_spans=[],
            page_width_pts=max(x1 - x0, 1e-6),
            page_height_pts=max(y1 - y0, 1e-6),
            ground_against_native_text=False,
            prompt_override=prompt,
            max_new_tokens=max_new_tokens,
        )
        payload = focus.model_dump(mode="python")
        crop_bbox = (x0, y0, x1, y1)
        for region in payload.get("regions", []):
            region["bbox"] = self._remap_focus_bbox(region.get("bbox"), crop_bbox, page_width_pts, page_height_pts)
        for dim in payload.get("dimensions", []):
            dim["bbox"] = self._remap_focus_bbox(dim.get("bbox"), crop_bbox, page_width_pts, page_height_pts)
        for area in payload.get("areas", []):
            # Area bboxes live in the same normalized crop coordinate system as dimensions.
            area["bbox"] = self._remap_focus_bbox(area.get("bbox"), crop_bbox, page_width_pts, page_height_pts)
        payload["raw_response"] = f"[{region_type}_FOCUS_PASS]\n" + (focus.raw_response or "")
        payload["page_number"] = page_number
        payload["page_width_pts"] = page_width_pts
        payload["page_height_pts"] = page_height_pts
        payload["warnings"] = list(payload.get("warnings") or []) + [f"[{region_type}_FOCUS] {note}"]
        return VisionPageResult.model_validate(payload)

    def _height_focus_pass(
        self,
        first_pass: VisionPageResult,
        pdf_path: Path,
        page_index: int,
        image_path: Path,
        page_number: int,
        page_width_pts: float,
        page_height_pts: float,
        detected_regions: list,
    ) -> VisionPageResult | None:
        """Try ELEVATION first, then SECTION -- either can carry the height dimensions."""
        settings = get_settings()
        for region_type in ("ELEVATION", "SECTION"):
            result = self._focused_region_pass(
                first_pass, pdf_path, page_index, image_path, page_number,
                page_width_pts, page_height_pts, detected_regions,
                region_type, HEIGHT_FOCUS_PROMPT,
                settings.vision_height_focus_max_new_tokens, settings.vision_height_focus_render_dpi,
            )
            if result is not None:
                return result
        return None

    def analyze_pdf(
        self,
        pdf_path: Path,
        *,
        ground_against_native_text: bool = True,
    ) -> VisionDocumentResult:
        settings = get_settings()
        render_dir = settings.resolve(settings.upload_dir) / "vision_rendered" / pdf_path.stem
        print(f"Rendering '{pdf_path.name}' at {settings.vision_render_dpi} DPI...", flush=True)
        page_files = render_pdf_pages(pdf_path, render_dir, dpi=settings.vision_render_dpi)
        print(f"Rendered {len(page_files)} page(s) to {render_dir}", flush=True)
        native_spans_by_page = self._native_number_spans_by_page(pdf_path)

        pages: list[VisionPageResult] = []
        warnings: list[str] = []
        for item in page_files:
            print(f"Analyzing page {item['page_number']}/{len(page_files)}...", flush=True)
            spans, native_page_w, native_page_h = native_spans_by_page.get(
                item["page_number"], ([], 0.0, 0.0)
            )
            # Prefer the page size derived from the actual rendered PNG's
            # own pixel dimensions over the native-text-layer page rect:
            # it's available even for scanned pages with no text layer at
            # all (native_spans_by_page returns 0.0/0.0 there), and it's
            # guaranteed consistent with the image the model actually saw.
            width_px = item.get("width_px")
            height_px = item.get("height_px")
            if width_px and height_px:
                page_w = float(width_px) * 72.0 / settings.vision_render_dpi
                page_h = float(height_px) * 72.0 / settings.vision_render_dpi
            else:
                page_w, page_h = native_page_w, native_page_h
            try:
                first_pass = self.analyze_image(
                    Path(item["image_path"]),
                    item["page_number"],
                    spans,
                    page_w,
                    page_h,
                    ground_against_native_text=ground_against_native_text,
                )
                # Independent validation gets region-specific focused
                # second passes (site plan, area statement, elevation/
                # section height).  This is intentionally disabled for
                # production/grounded mode so native-PDF grounding remains
                # the safety filter there.
                focus_passes: list[VisionPageResult] = []
                page_index = item["page_number"] - 1
                # Deterministic (caption-anchored, non-VLM) region detection is
                # needed by the height focus pass below regardless of
                # `ground_against_native_text`, so it is computed once here
                # rather than only inside the ungrounded branch.
                try:
                    from backend.cv_extraction import region_detection
                    detected_regions = region_detection.detect_regions(pdf_path, page_index)
                except Exception as det_exc:
                    detected_regions = []
                    warnings.append(
                        f"page {item['page_number']}: deterministic region detection failed: {det_exc}"
                    )
                if not ground_against_native_text:
                    focus_specs = [
                        ("SITE_PLAN", SITE_PLAN_FOCUS_PROMPT, settings.vision_focus_max_new_tokens, settings.vision_site_focus_render_dpi),
                        ("AREA_STATEMENT", AREA_STATEMENT_FOCUS_PROMPT, settings.vision_area_focus_max_new_tokens, settings.vision_area_focus_render_dpi),
                    ]
                    for region_type, prompt, max_new_tokens, render_dpi in focus_specs:
                        try:
                            result = self._focused_region_pass(
                                first_pass, pdf_path, page_index, Path(item["image_path"]),
                                item["page_number"], page_w, page_h, detected_regions,
                                region_type, prompt, max_new_tokens, render_dpi,
                            )
                            if result is not None:
                                focus_passes.append(result)
                        except Exception as focus_exc:
                            warnings.append(
                                f"page {item['page_number']}: {region_type} focus pass failed: {focus_exc}"
                            )
                # The height focus pass (ELEVATION/SECTION) runs regardless of
                # `ground_against_native_text`. It was previously gated behind
                # the same `if not ground_against_native_text:` branch as the
                # SITE_PLAN/AREA_STATEMENT passes above, which meant it never
                # ran in production (both real call sites -- pdf_extractor.py
                # and the /analyze route -- use the default
                # `ground_against_native_text=True`), leaving building-height
                # extraction dependent entirely on whatever the low-resolution
                # whole-sheet first pass happened to notice. The focused
                # crop's own dimension extraction never uses native-text
                # grounding either way (see `_focused_region_pass`, which
                # always calls `analyze_image(..., ground_against_native_text=
                # False, ...)` since a re-rendered crop has no native text
                # layer to ground against), so this does not weaken the
                # grounding safety net for the generic first-pass dimensions
                # -- it only stops silently skipping a dedicated, higher-
                # resolution, explicitly-prompted pass over the one region of
                # the sheet actually likely to carry the building height.
                # Height values found here still go through the same
                # semantic classification in `pipeline.py` (BUILDING_HEIGHT /
                # PROPOSED_BUILDING_HEIGHT vs REGULATORY_MAX_HEIGHT, etc.)
                # before ever reaching `NormalizedPlan` -- this only affects
                # whether the evidence is looked for, not whether it is
                # trusted uncritically once found.
                try:
                    height_result = self._height_focus_pass(
                        first_pass, pdf_path, page_index, Path(item["image_path"]),
                        item["page_number"], page_w, page_h, detected_regions,
                    )
                    if height_result is not None:
                        focus_passes.append(height_result)
                except Exception as focus_exc:
                    warnings.append(
                        f"page {item['page_number']}: height focus pass failed: {focus_exc}"
                    )
                raw_responses = [first_pass.raw_response or ""]
                for focus_pass in focus_passes:
                    first_pass.dimensions.extend(focus_pass.dimensions)
                    first_pass.regions.extend(focus_pass.regions)
                    first_pass.areas.extend(focus_pass.areas)
                    first_pass.warnings.extend(focus_pass.warnings)
                    raw_responses.append(focus_pass.raw_response or "")
                if focus_passes:
                    first_pass.raw_response = "\n\n".join(raw_responses)
                pages.append(first_pass)
            except Exception as exc:
                logger.exception("vision page analysis failed", extra={"page": item["page_number"]})
                warnings.append(f"page {item['page_number']}: vision analysis failed: {exc}")
                print(f"  [page {item['page_number']}] FAILED: {exc}", flush=True)

        result = VisionDocumentResult(
            pages=pages,
            model_name=self.model_name,
            warnings=warnings,
        )
        # Cross-region numeric consistency guard: applied once here so every
        # caller (production pdf_extractor.py fusion, run_vision.py,
        # run_validation.py, the eval harness) benefits without each needing
        # to remember to call it. See vision_consistency.py's docstring for
        # the concrete PLAN5 bug this catches -- a value pattern-completed
        # from one region into an unrelated region's different field.
        from backend.spatial_reasoning.vision_consistency import sanitize_vision_result
        return sanitize_vision_result(result)