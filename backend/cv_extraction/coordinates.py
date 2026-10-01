"""
Coordinate handling for the hybrid extraction pipeline.

Everything downstream of extraction (see raw_types.py) is expressed in a
single page's PAGE_POINTS space (origin top-left, X right, Y down,
matching PDF page conventions — see backend/schemas/geometry.py).

Two other spaces feed into that:

- Raster pixel space: whenever a page is rasterized (for OCR or OpenCV),
  pixel coordinates must be scaled back to page points using the DPI the
  page was rasterized at.
- Rotated page space: a PDF page may declare a /Rotate of 0/90/180/270.
  PyMuPDF's high-level APIs (get_text, get_pixmap) already return
  content in the *rotated* (i.e. as-displayed) page space, so this
  module treats `page.rect` (post-rotation) as ground truth and does
  NOT apply a second rotation on top of PyMuPDF's own output. What it
  DOES do is normalize/record the rotation for auditability and for the
  handful of raw OpenCV pixel coordinates that must be mapped back.

This module deliberately has zero PDF-library imports, so it can be unit
tested without a real PDF file.
"""

from __future__ import annotations

from backend.config import get_settings
from backend.cv_extraction.raw_types import CoordinateTransformRecord
from backend.schemas.geometry import BoundingBox, Point

POINTS_PER_INCH = 72.0


def points_per_pixel(dpi: float) -> float:
    """Scale factor to go from a raster pixel (at `dpi`) to page points."""
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    return POINTS_PER_INCH / dpi


def pixel_to_page_point(x_px: float, y_px: float, dpi: float) -> Point:
    """Convert a single raster pixel coordinate to page-point space."""
    scale = points_per_pixel(dpi)
    return Point(x=x_px * scale, y=y_px * scale)


def pixel_bbox_to_page_bbox(
    x0: float, y0: float, x1: float, y1: float, dpi: float
) -> BoundingBox:
    """Convert a raster pixel bounding box to a page-point BoundingBox."""
    scale = points_per_pixel(dpi)
    return BoundingBox(
        min_x=min(x0, x1) * scale,
        min_y=min(y0, y1) * scale,
        max_x=max(x0, x1) * scale,
        max_y=max(y0, y1) * scale,
    )


def normalize_rotation(rotation_degrees: int) -> int:
    """Fold any rotation value onto {0, 90, 180, 270}."""
    return int(rotation_degrees) % 360 // 90 * 90


def build_transform_record(
    page: int,
    dpi: float,
    rotation_degrees: int,
    page_width_pts: float,
    page_height_pts: float,
    provisional_points_per_metre: float | None = None,
) -> CoordinateTransformRecord:
    """
    Build the auditable transform record for one page.

    `provisional_points_per_metre` falls back to
    `settings.default_points_per_metre` — this is explicitly NOT an
    authoritative page->metric scale (spatial reasoning may find a real
    scale bar or cross-check against a known dimension and override it),
    it just ensures the metadata required by NormalizedGeometry is never
    silently fabricated later.
    """
    settings = get_settings()
    ppm = (
        provisional_points_per_metre
        if provisional_points_per_metre is not None
        else settings.default_points_per_metre
    )
    return CoordinateTransformRecord(
        page=page,
        raster_dpi=dpi,
        points_per_pixel=points_per_pixel(dpi),
        rotation_degrees=normalize_rotation(rotation_degrees),
        page_width_pts=page_width_pts,
        page_height_pts=page_height_pts,
        provisional_points_per_metre=ppm,
    )


__all__ = [
    "POINTS_PER_INCH",
    "points_per_pixel",
    "pixel_to_page_point",
    "pixel_bbox_to_page_bbox",
    "normalize_rotation",
    "build_transform_record",
]
