"""
Architecture V2, Phase DXF-4 -- tests for the pure/deterministic helper
functions in `backend.tools.run_dxf_structural_hypothesis_shadow`. The
script's actual extraction/comparison logic is exercised directly against
the real DXF plans (see `PHASE_DXF_REPORT.md` for that run's results);
this file covers the small, deterministic pieces in isolation, mirroring
`tests/test_run_evidence_decision_shadow.py`'s own scope for the PDF-side
shadow script.
"""

from __future__ import annotations

import os

from backend.config import get_settings
from backend.schemas.enums import DocumentType
from backend.schemas.extraction import ExtractionResult
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.tools.run_dxf_structural_hypothesis_shadow import _run_dxf_with_flag, _score_dxf_result


def test_score_dxf_result_scores_matching_and_missing_fields():
    result = ExtractionResult(
        document_id="d", document_type=DocumentType.DXF, page_count=1,
        independent_cv=IndependentCVResult(document_id="d", measurements=[
            IndependentMeasurement(field="plot.width", value_m=12.19, source="NATIVE_TEXT", confidence=0.97),
        ]),
    )
    expected = {"plot.width": 12.19, "plot.depth": 18.28}
    scored = _score_dxf_result(result, expected)
    assert scored["plot.width"].verdict == "CORRECT"
    assert scored["plot.depth"].verdict == "MISSING"


def test_score_dxf_result_handles_none_independent_cv():
    result = ExtractionResult(document_id="d", document_type=DocumentType.DXF, page_count=1, independent_cv=None)
    expected = {"plot.width": 12.19}
    scored = _score_dxf_result(result, expected)
    assert scored["plot.width"].verdict == "MISSING"


def test_score_dxf_result_flags_wrong_values():
    result = ExtractionResult(
        document_id="d", document_type=DocumentType.DXF, page_count=1,
        independent_cv=IndependentCVResult(document_id="d", measurements=[
            IndependentMeasurement(field="plot.width", value_m=4.0, source="NATIVE_TEXT", confidence=0.4),
        ]),
    )
    expected = {"plot.width": 12.19}
    scored = _score_dxf_result(result, expected)
    assert scored["plot.width"].verdict == "WRONG"


def test_run_dxf_with_flag_restores_environment_after_success_and_failure(tmp_path):
    assert "USE_DXF_EVIDENCE_DECISION_ENGINE" not in os.environ

    # A path that doesn't exist raises during extraction -- the environment
    # variable this function sets must still be cleaned up afterward, not
    # leaked into whatever runs next (a real regression risk for a shadow
    # script that runs many extractions back-to-back in one process).
    missing_path = tmp_path / "does-not-exist.dxf"
    try:
        _run_dxf_with_flag(missing_path, "p", True)
    except Exception:
        pass
    assert "USE_DXF_EVIDENCE_DECISION_ENGINE" not in os.environ
    get_settings.cache_clear()
    assert get_settings().use_dxf_evidence_decision_engine is False
