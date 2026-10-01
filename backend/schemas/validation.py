from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


AgreementStatus = Literal[
    "AGREED",
    "CV_ONLY",
    "VISION_ONLY",
    "CONFLICT",
    "MISSING",
]


class ValidationField(BaseModel):
    field: str
    cv_value_m: Optional[float] = None
    vision_value_m: Optional[float] = None
    absolute_difference_m: Optional[float] = None
    tolerance_m: Optional[float] = None
    # Generic numeric representation for non-length fields such as m², %, and FAR ratio.
    cv_value: Optional[float] = None
    vision_value: Optional[float] = None
    absolute_difference: Optional[float] = None
    tolerance: Optional[float] = None
    unit: Optional[str] = None
    status: AgreementStatus
    cv_source: Optional[str] = None
    vision_evidence: Optional[str] = None
    vision_confidence: Optional[float] = None
    note: Optional[str] = None


class ExtractionValidationReport(BaseModel):
    document_id: str
    cv_warnings: list[str] = Field(default_factory=list)
    vision_warnings: list[str] = Field(default_factory=list)
    fields: list[ValidationField] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    # Independent evidence outputs.  `cv_result` is NOT the legacy
    # NormalizedPlan resolver; it is the CV/native-text validation path.
    cv_result: Optional[dict] = None
    # Kept for diagnostics/regression comparison only.
    legacy_cv_plan: Optional[dict] = None
    vision_result: Optional[dict] = None
    # Final deterministic CV+Vision agreement output. Conflicts remain null.
    final_agreed_values: Optional[dict] = None
    # PHASEE3NEW.md Validation/Fusion Engine: document-evidence-verified
    # fusion, with conflict resolution against the PDF itself (not just
    # CV/Vision agreement). See backend.spatial_reasoning.final_fusion
    # .build_document_verified_fusion for the full schema
    # (cv_values / vision_values / document_evidence / conflicts /
    # validation / final_agreed_values / reported_vs_calculated /
    # provenance).
    document_verified_fusion: Optional[dict] = None

    @property
    def has_conflicts(self) -> bool:
        return any(f.status == "CONFLICT" for f in self.fields)
