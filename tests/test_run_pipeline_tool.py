"""
Smoke tests for `backend.tools.run_pipeline` -- the CLI that runs the
FULL extraction + spatial-reasoning pipeline (CV geometry, and vision
semantics fusion when enabled), as opposed to `run_vision.py` which only
exercises the isolated vision stage.

Not a test of resolution quality (that's covered by
tests/test_pipeline_integration.py and friends) -- just confirms the
tool's own glue code (`_summarize`, the extract -> resolve composition)
runs without crashing against the project's standard synthetic fixtures.
"""

from __future__ import annotations

from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.spatial_reasoning.pipeline import build_normalized_plan
from backend.tools.run_pipeline import _summarize
from tests.fixtures.pdf_builders import build_vector_plan_pdf


def test_full_pipeline_runs_end_to_end_without_vision(tmp_path):
    pdf_path = build_vector_plan_pdf(tmp_path / "plan.pdf")
    extraction = PDFHybridExtractor().extract(pdf_path, "smoke-test-doc")
    assert extraction.vision_pages == []  # vision disabled by default

    plan = build_normalized_plan(extraction, plan_id="smoke-test-plan")
    assert plan.plan_id == "smoke-test-plan"

    summary = _summarize(plan)
    assert "plot:" in summary
    assert "building:" in summary
    assert "road:" in summary
    assert "setbacks:" in summary
    assert "coverage:" in summary


def test_summarize_handles_missing_values_gracefully():
    from backend.schemas.evidence import ValueField
    from backend.schemas.normalized_plan import BuildingSection, NormalizedPlan, PlotSection, RoadSection, SetbackSection

    missing = ValueField[float].missing("no evidence")
    plan = NormalizedPlan(
        plan_id="p",
        source_document_id="d",
        plot=PlotSection(width=missing, depth=missing, area=missing),
        building=BuildingSection(width=missing, depth=missing, footprint_area=missing),
        road=RoadSection(width=missing),
        setbacks=SetbackSection(front=missing, rear=missing, left=missing, right=missing),
        coverage=missing,
        far=missing,
    )
    summary = _summarize(plan)
    assert "MISSING" in summary
