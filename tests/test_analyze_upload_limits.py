"""
Tests for the `/api/analyze/upload` size gate.

A real user upload (a 45.7MB vectorized/traced DXF) was rejected with a
bare 413 against the previous 25MB default -- vectorized DXFs (many
thousands of tiny line/polygon fragments, see dxf_extractor.py's own
notes) are legitimately much larger than a PDF sheet. This pins the raised
default and the gate's actual accept/reject behavior so the limit can't
silently regress back down.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.config import get_settings

client = TestClient(app)


def test_default_upload_limit_has_real_headroom_above_observed_dxf_sizes():
    # A real vectorized/traced DXF upload that triggered this fix was
    # 45.7MB; the configured default must clear it with headroom, not just
    # barely exceed the one observed file.
    assert get_settings().max_upload_size_mb >= 60


def test_upload_under_the_configured_limit_is_accepted(monkeypatch, tmp_path):
    import ezdxf

    monkeypatch.setattr(get_settings(), "max_upload_size_mb", 1)
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    doc = ezdxf.new("R2018")
    buf = io.StringIO()
    doc.write(buf)
    small_dxf = buf.getvalue().encode("utf-8")  # a real, minimal, parseable DXF -- only the size gate is under test
    resp = client.post(
        "/api/analyze/upload",
        files={"file": ("small.dxf", io.BytesIO(small_dxf), "application/octet-stream")},
    )
    assert resp.status_code != 413


def test_upload_over_the_configured_limit_is_rejected_with_413(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "max_upload_size_mb", 1)
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)
    oversized = b"0" * (2 * 1024 * 1024)  # 2MB against a 1MB limit
    resp = client.post(
        "/api/analyze/upload",
        files={"file": ("big.dxf", io.BytesIO(oversized), "application/octet-stream")},
    )
    assert resp.status_code == 413
