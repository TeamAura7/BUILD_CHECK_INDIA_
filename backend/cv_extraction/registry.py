"""
Minimal extractor registry.

Downstream code should never branch on file extension — it asks the
registry for an extractor by `DocumentType` (or lets the registry sniff
a path's extension once, at the edge).
"""

from __future__ import annotations

from pathlib import Path

from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
from backend.cv_extraction.interfaces import GeometryExtractor, UnsupportedDocumentTypeError
from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.schemas.enums import DocumentType

_EXTENSION_TO_TYPE = {
    ".pdf": DocumentType.VECTOR_PDF,  # PDFHybridExtractor handles vector/raster/mixed internally
    ".dxf": DocumentType.DXF,          # DXFHybridExtractor -- exact vector geometry, no rasterization
}


class ExtractorRegistry:
    def __init__(self) -> None:
        self._extractors: dict[DocumentType, GeometryExtractor] = {}

    def register(self, extractor: GeometryExtractor) -> None:
        self._extractors[extractor.supported_document_type()] = extractor

    def get_for_document_type(self, document_type: DocumentType) -> GeometryExtractor:
        try:
            return self._extractors[document_type]
        except KeyError as exc:
            raise UnsupportedDocumentTypeError(
                f"No extractor registered for {document_type}"
            ) from exc

    def get_for_path(self, path: Path) -> GeometryExtractor:
        document_type = _EXTENSION_TO_TYPE.get(Path(path).suffix.lower())
        if document_type is None:
            raise UnsupportedDocumentTypeError(f"No extractor registered for file type {path}")
        return self.get_for_document_type(document_type)


def default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register(PDFHybridExtractor())
    registry.register(DXFHybridExtractor())
    return registry


__all__ = ["ExtractorRegistry", "default_registry"]
