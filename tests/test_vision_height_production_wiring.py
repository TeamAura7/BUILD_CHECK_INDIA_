"""
Regression test for a confirmed production bug: the focused ELEVATION/
SECTION height Vision pass (`HEIGHT_FOCUS_PROMPT` /
`BaseArchitecturalPlanExtractor._height_focus_pass`) was gated behind
`if not ground_against_native_text:` inside `analyze_pdf`. Both real
production call sites (`backend/cv_extraction/pdf_extractor.py` and
`backend/app/routes/analyze.py`) call `analyze_pdf` with the default
`ground_against_native_text=True`, so the dedicated height pipeline never
ran for a real user upload -- building height extraction depended entirely
on whatever the low-resolution, whole-sheet, generically-prompted first
pass happened to notice.

This exercises the real `analyze_pdf` orchestration end to end (page
rendering, deterministic caption-anchored region detection, focused-crop
re-rendering) with only the model call itself stubbed out, rather than unit
-testing the height-resolution math in isolation (see
`tests/test_height_resolution.py`, which never calls `analyze_pdf` at all
and therefore could not have caught this).
"""
from __future__ import annotations

import inspect
import json

import fitz  # PyMuPDF
import pytest

from backend.config import get_settings
from backend.vision_extraction.base import BaseArchitecturalPlanExtractor


def _build_elevation_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # A short, standalone caption -- exactly what
    # `backend.cv_extraction.region_detection`'s caption-anchor detector
    # looks for, independent of anything Vision reports.
    page.insert_text((260, 40), "ELEVATION", fontsize=10)
    page.draw_rect(fitz.Rect(60, 60, 550, 700), color=(0, 0, 0), width=1.0)
    doc.save(str(path))
    doc.close()
    return path


class _HeightFocusStubExtractor(BaseArchitecturalPlanExtractor):
    """A stub backend whose response depends only on which prompt it is given.

    This mirrors real backends' contract (`_load` + `_generate_raw_response`)
    without needing an actual VLM, so the test exercises `analyze_pdf`'s real
    orchestration logic -- including whether the height focus pass is even
    attempted -- rather than mocking `analyze_pdf`/`_focused_region_pass`
    themselves away.
    """

    DEFAULT_MODEL_NAME = "stub-height-wiring-test"

    def __init__(self):
        super().__init__(model_name=self.DEFAULT_MODEL_NAME)
        self.height_focus_prompt_calls = 0
        self.first_pass_calls = 0

    def _load(self) -> None:
        pass

    def _generate_raw_response(self, image_path, prompt, max_new_tokens=None) -> str:
        # HEIGHT_FOCUS_PROMPT's opening sentence is unique to it among the
        # prompts `analyze_pdf` can select.
        if "ELEVATION or SECTION drawing" in prompt:
            self.height_focus_prompt_calls += 1
            return json.dumps({
                "page_number": 1,
                "units": "m",
                "scale": None,
                "regions": [
                    {"id": "height_focus", "type": "ELEVATION", "bbox": [0, 0, 1000, 1000],
                     "confidence": 0.95, "label": "ELEVATION", "evidence": ""},
                ],
                "dimensions": [
                    {"value": 9.6, "unit": "m", "type": "BUILDING_HEIGHT", "region_id": "height_focus",
                     "bbox": [100, 50, 900, 70], "evidence": "9.60", "confidence": 0.9},
                ],
                "areas": [],
                "warnings": [],
            })
        # The generic whole-sheet first pass: deliberately reports nothing,
        # simulating a busy real sheet where a small elevation dimension is
        # missed at low resolution -- this is exactly the failure mode the
        # focused pass exists to recover from.
        self.first_pass_calls += 1
        return json.dumps({
            "page_number": 1, "units": "m", "scale": None,
            "regions": [], "dimensions": [], "areas": [], "warnings": [],
        })


def test_analyze_pdf_runs_focused_height_pass_in_production_grounded_mode(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")

    pdf_path = tmp_path / "plan_with_elevation.pdf"
    _build_elevation_pdf(pdf_path)

    extractor = _HeightFocusStubExtractor()
    # `ground_against_native_text=True` is the default every real production
    # call site (pdf_extractor.py, analyze.py) actually uses -- this is not
    # an artificial "best case" setting.
    result = extractor.analyze_pdf(pdf_path, ground_against_native_text=True)

    assert extractor.height_focus_prompt_calls >= 1, (
        "The focused ELEVATION/SECTION height pass must run even in "
        "production/grounded mode. It was previously gated behind "
        "`if not ground_against_native_text:`, which silently disabled "
        "building-height extraction for every real user upload."
    )

    heights = [
        d for page in result.pages for d in page.dimensions
        if d.type.upper() == "BUILDING_HEIGHT"
    ]
    assert heights, "Expected a BUILDING_HEIGHT dimension merged in from the focused height pass."
    assert heights[0].value == pytest.approx(9.6)


def test_production_call_sites_default_to_grounded_mode():
    """Pin the actual default so this test breaks loudly if it silently changes.

    The height-focus-pass fix only matters because production genuinely
    calls `analyze_pdf` with `ground_against_native_text=True` (the
    signature default) -- if that default were ever flipped, the reasoning
    behind decoupling the height pass from it should be revisited.
    """
    sig = inspect.signature(BaseArchitecturalPlanExtractor.analyze_pdf)
    assert sig.parameters["ground_against_native_text"].default is True
