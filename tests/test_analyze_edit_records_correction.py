"""`POST /api/jobs/{id}/edit` records each edited field as a correction event
(backend/corpus/corrections.py) without changing the edit's own response, and
without letting a correction-capture failure break the edit."""

from __future__ import annotations

from fastapi.testclient import TestClient

import backend.app.routes.analyze as analyze_mod
from backend.app.main import app
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.normalized_plan import BuildingSection, NormalizedPlan, PlotSection, RoadSection, SetbackSection

client = TestClient(app)


def _vf(value, level=ConfidenceLevel.LOW, source="VECTOR_GEOMETRY"):
    if value is None:
        return ValueField.missing("absent")
    return ValueField(value=value, confidence=Confidence(level=level, reason="test"), source=source)


def _seed_job(job_id: str, *, source_paths=None) -> None:
    plan = NormalizedPlan(
        plan_id="t", source_document_id="doc-under-test",
        plot=PlotSection(width=_vf(4.7367), depth=_vf(3.5017), area=_vf(16.6)),
        building=BuildingSection(width=_vf(3.0), depth=_vf(4.0), footprint_area=_vf(12.0)),
        road=RoadSection(width=_vf(9.0)),
        setbacks=SetbackSection(front=_vf(3.0), rear=_vf(2.0), left=_vf(1.5), right=_vf(1.5)),
        coverage=_vf(26.7), far=_vf(1.2),
    )
    report = {
        "pipeline": {"municipality": "BBMP"},
        "plan": plan.model_dump(mode="json"),
        "plan_summary": {},
        "compliance": {"overall_status": "COMPLIANT", "rule_results": [], "counts": {}},
        "suggestions": [],
    }
    with analyze_mod._JOBS_LOCK:
        analyze_mod._JOBS[job_id] = {
            "id": job_id, "status": "done", "result": report,
            "source_paths": source_paths or {}, "log": [], "error": None,
        }


def _capture_record_correction(monkeypatch):
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        from backend.corpus.corrections import CorrectionEvent
        return CorrectionEvent(
            correction_id="fake", recorded_at="now", job_id=kwargs["job_id"],
            document_id=kwargs["document_id"], field=kwargs["field"], unit=kwargs["unit"],
            corrected_value=kwargs["corrected_value"],
        )

    monkeypatch.setattr("backend.corpus.corrections.record_correction", _fake)
    return calls


def test_editing_a_field_records_a_correction_with_the_before_and_after_value(monkeypatch):
    calls = _capture_record_correction(monkeypatch)
    _seed_job("job-edit-1")
    resp = client.post("/api/jobs/job-edit-1/edit", json={
        "updates": {"plot.width": {"value": 17.59, "unit": "m"}},
        "reviewer": "reviewer@example.com", "note": "measured from the printed dimension line",
    })
    assert resp.status_code == 200
    assert resp.json()["plan"]["plot"]["width"]["value"] == 17.59
    assert len(calls) == 1
    call = calls[0]
    assert call["field"] == "plot.width"
    assert call["predicted_value"] == 4.7367
    assert call["predicted_confidence"] == "LOW"
    assert call["corrected_value"] == 17.59
    assert call["corrected_by"] == "reviewer@example.com"
    assert call["document_id"] == "doc-under-test"


def test_editing_two_fields_records_two_corrections(monkeypatch):
    calls = _capture_record_correction(monkeypatch)
    _seed_job("job-edit-2")
    resp = client.post("/api/jobs/job-edit-2/edit", json={
        "updates": {
            "plot.width": {"value": 17.59, "unit": "m"},
            "setbacks.front": {"value": 3.5, "unit": "m"},
        },
    })
    assert resp.status_code == 200
    assert {c["field"] for c in calls} == {"plot.width", "setbacks.front"}


def test_a_correction_capture_failure_never_breaks_the_edit_response(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("backend.corpus.corrections.record_correction", _boom)
    _seed_job("job-edit-3")
    resp = client.post("/api/jobs/job-edit-3/edit", json={"updates": {"plot.width": {"value": 17.59, "unit": "m"}}})
    assert resp.status_code == 200
    assert resp.json()["plan"]["plot"]["width"]["value"] == 17.59


def test_an_invalid_edit_records_no_correction(monkeypatch):
    calls = _capture_record_correction(monkeypatch)
    _seed_job("job-edit-4")
    resp = client.post("/api/jobs/job-edit-4/edit", json={"updates": {"plot.width": {"value": -5.0, "unit": "m"}}})
    assert resp.status_code == 400
    assert calls == []
