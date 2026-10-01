"""
PDF <-> DXF post-hoc cross-check report contract.

See `backend/spatial_reasoning/pdf_dxf_reconciliation.py`'s module docstring
for the full rationale. This schema mirrors the shape of `backend/schemas/
validation.py`'s `ValidationField`/`ExtractionValidationReport` (the
existing CV-vs-Vision precedent), field-relabeled for PDF-vs-DXF rather than
copied verbatim, since the two comparisons mean different things (CV/Vision
read the SAME PDF page; PDF/DXF are two independent source DOCUMENTS of the
same real building).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

PdfDxfFieldStatus = Literal["AGREED", "CONFLICT", "PDF_ONLY", "DXF_ONLY", "BOTH_MISSING"]


class PdfDxfFieldComparison(BaseModel):
    """One field's independent PDF vs. DXF reading, and what the reconciled
    plan ended up shipping for it."""

    field: str
    unit: str
    pdf_value: Optional[float] = None
    dxf_value: Optional[float] = None
    pdf_confidence: Optional[str] = None
    dxf_confidence: Optional[str] = None
    absolute_difference: Optional[float] = None
    tolerance: Optional[float] = None
    status: PdfDxfFieldStatus
    shipped_value: Optional[float] = None
    note: str


class PdfDxfReconciliationReport(BaseModel):
    """Full, both-sides-visible record of a PDF+DXF cross-check -- returned
    alongside (never instead of) the reconciled `NormalizedPlan`, so a
    disagreement is never visible only as a single flag on the shipped
    field; the raw numbers on both sides are always here too."""

    pdf_document_id: str
    dxf_document_id: str
    fields: list[PdfDxfFieldComparison] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    has_conflicts: bool = False
    note: str = (
        "PDF extraction is documented reliable; DXF has multiple root-caused "
        "failure modes (see DXF_FAILURE_TAXONOMY.md). On CONFLICT, the PDF "
        "value ships as authoritative -- an intentional, narrow exception to "
        "this project's usual 'never silently pick a winner on a genuine "
        "conflict' rule, scoped ONLY to this PDF-vs-DXF cross-check. The "
        "disagreement itself is never hidden: it survives on the shipped "
        "field's own `.conflict` and in this report's own field rows."
    )


__all__ = ["PdfDxfFieldStatus", "PdfDxfFieldComparison", "PdfDxfReconciliationReport"]
