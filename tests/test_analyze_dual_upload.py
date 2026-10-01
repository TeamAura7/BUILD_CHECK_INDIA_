"""
Tests for the PDF+DXF dual-upload wiring in `backend/app/routes/analyze.py`
(Phase 2 of the PDF+DXF cross-check). These test the ROUTING/validation
logic only -- `_run_pipeline` is monkeypatched to a capturing stub so real
extraction never runs (that's covered separately by
`tests/test_pdf_dxf_reconciliation_integration.py`), keeping these fast and
focused on "did the right paths/kwargs get passed to the pipeline."
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi.testclient import TestClient

import backend.app.routes.analyze as analyze_mod
from backend.app.main import app
from backend.config import get_settings

client = TestClient(app)


def _capture_calls(monkeypatch):
    calls: list[dict] = []

    def _fake_run_pipeline(job_id, pdf_path, municipality, vision, backend_choice, max_new_tokens, top_k, all_fields, mode, dxf_path=None):
        calls.append({
            "job_id": job_id, "pdf_path": pdf_path, "dxf_path": dxf_path,
            "municipality": municipality, "mode": mode,
        })
        with analyze_mod._JOBS_LOCK:
            analyze_mod._JOBS[job_id]["status"] = "done"
            analyze_mod._JOBS[job_id]["result"] = {"stub": True}

    monkeypatch.setattr(analyze_mod, "_run_pipeline", _fake_run_pipeline)
    return calls


def _wait_for_call(calls, timeout=2.0):
    deadline = time.time() + timeout
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    return calls


def test_single_file_upload_unchanged_when_second_file_omitted(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    calls = _capture_calls(monkeypatch)

    resp = client.post(
        "/api/analyze/upload",
        files={"file": ("plan.pdf", io.BytesIO(b"%PDF-1.4 stub"), "application/pdf")},
    )
    assert resp.status_code == 200
    _wait_for_call(calls)
    assert len(calls) == 1
    assert calls[0]["dxf_path"] is None
    assert calls[0]["pdf_path"].name.endswith("_plan.pdf")


def test_dual_upload_with_two_same_extension_files_is_rejected_400(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    calls = _capture_calls(monkeypatch)

    resp = client.post(
        "/api/analyze/upload",
        files={
            "file": ("plan_a.pdf", io.BytesIO(b"%PDF-1.4 stub a"), "application/pdf"),
            "second_file": ("plan_b.pdf", io.BytesIO(b"%PDF-1.4 stub b"), "application/pdf"),
        },
    )
    assert resp.status_code == 400
    assert "pdf" in resp.json()["detail"].lower()
    assert calls == []  # pipeline must never be invoked for a rejected request


def test_dual_upload_accepts_either_file_order(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)

    # PDF first, DXF second.
    calls_a = _capture_calls(monkeypatch)
    resp_a = client.post(
        "/api/analyze/upload",
        files={
            "file": ("plan.pdf", io.BytesIO(b"%PDF-1.4 stub"), "application/pdf"),
            "second_file": ("plan.dxf", io.BytesIO(b"0\nEOF\n"), "application/octet-stream"),
        },
    )
    assert resp_a.status_code == 200
    _wait_for_call(calls_a)
    assert len(calls_a) == 1
    assert calls_a[0]["pdf_path"].suffix == ".pdf"
    assert calls_a[0]["dxf_path"].suffix == ".dxf"

    # DXF first, PDF second -- must resolve identically regardless of order.
    calls_b = _capture_calls(monkeypatch)
    resp_b = client.post(
        "/api/analyze/upload",
        files={
            "file": ("plan.dxf", io.BytesIO(b"0\nEOF\n"), "application/octet-stream"),
            "second_file": ("plan.pdf", io.BytesIO(b"%PDF-1.4 stub"), "application/pdf"),
        },
    )
    assert resp_b.status_code == 200
    _wait_for_call(calls_b)
    assert len(calls_b) == 1
    assert calls_b[0]["pdf_path"].suffix == ".pdf"
    assert calls_b[0]["dxf_path"].suffix == ".dxf"


def test_analyze_sample_second_sample_plan_resolves_a_real_fixture_pair(monkeypatch):
    settings = get_settings()
    plan5_pdf = Path(settings.test_plans_dir) / "PLAN5.pdf"
    plan5_dxf = Path(settings.test_plans_dir) / "PLAN5.dxf"
    if not (plan5_pdf.exists() and plan5_dxf.exists()):
        import pytest
        pytest.skip("PLAN5.pdf/PLAN5.dxf fixtures not present in this checkout")

    calls = _capture_calls(monkeypatch)
    resp = client.post(
        "/api/analyze/sample",
        json={"sample_plan": "PLAN5.pdf", "second_sample_plan": "PLAN5.dxf"},
    )
    assert resp.status_code == 200
    _wait_for_call(calls)
    assert len(calls) == 1
    assert calls[0]["pdf_path"] == plan5_pdf
    assert calls[0]["dxf_path"] == plan5_dxf


def test_analyze_sample_rejects_two_same_format_samples():
    resp = client.post(
        "/api/analyze/sample",
        json={"sample_plan": "PLAN5.pdf", "second_sample_plan": "PLAN6.pdf"},
    )
    assert resp.status_code == 400
