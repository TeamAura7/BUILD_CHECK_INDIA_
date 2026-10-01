"""
Extraction-layer interfaces.

The point of this file: nothing downstream of extraction should ever
import a PDF library, an image library, or a DXF/BIM library. Every
extractor implements GeometryExtractor and returns the same
ExtractionResult shape, regardless of source format.

Phase 1 defines the interface only. Concrete PDF/vector/OCR extraction
logic is implemented by later phases/teammates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.schemas.enums import DocumentType
from backend.schemas.extraction import ExtractionResult


class GeometryExtractor(ABC):
    """
    Interface every source-format extractor must implement.

    supported_document_type() lets a registry pick the right extractor
    without any downstream code branching on file extension.
    """

    @abstractmethod
    def supported_document_type(self) -> DocumentType: ...

    @abstractmethod
    def extract(self, document_path: Path, document_id: str) -> ExtractionResult: ...


class UnsupportedDocumentTypeError(Exception):
    """Raised by an extractor registry when no extractor matches a document."""


class DXFGeometryExtractor(GeometryExtractor):
    """
    Placeholder interface for future DXF ingestion.

    NOT implemented in Phase 1 — exists only so RuleEngine/NormalizedPlan
    consumers never need to change when DXF support is added.
    """

    def supported_document_type(self) -> DocumentType:
        return DocumentType.DXF

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        raise NotImplementedError("DXF extraction is out of scope for Phase 1.")


class BIMGeometryExtractor(GeometryExtractor):
    """Placeholder interface for future BIM ingestion. NOT implemented in Phase 1."""

    def supported_document_type(self) -> DocumentType:
        return DocumentType.BIM

    def extract(self, document_path: Path, document_id: str) -> ExtractionResult:
        raise NotImplementedError("BIM extraction is out of scope for Phase 1.")
