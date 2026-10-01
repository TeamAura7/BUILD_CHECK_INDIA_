"""
OpenCV raster/geometry evidence.

OpenCV is an EVIDENCE SOURCE, never the final semantic decision maker.
It supplements vector extraction (which can be sparse or entirely
absent on a scanned plan) with line/contour/connected-component
detection on the rasterized page, for wall evidence and geometry
verification.

Everything here returns raw pixel-space detections already converted to
PAGE_POINTS via `coordinates.py` — callers never see raw pixel
coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.cv_extraction.coordinates import pixel_bbox_to_page_bbox, pixel_to_page_point
from backend.cv_extraction.raw_types import RawLine, RawPolygon, RawRectangle, SourceKind
from backend.schemas.geometry import Line, Point, Polygon
from backend.tools.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)


def _import_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment issue, not logic
        raise RuntimeError(
            "opencv-python and numpy are required for OpenCV evidence extraction. "
            "Install with `pip install opencv-python-headless numpy`."
        ) from exc
    return cv2


def pil_to_gray_array(image) -> "np.ndarray":
    import numpy as np

    arr = np.array(image.convert("L"))
    return arr


def detect_lines(image, page_number: int, dpi: float, min_length_px: float = 35.0) -> list[RawLine]:
    """Detect long architectural line evidence.

    LSD is preferred because it is less sensitive to the threshold/angle tuning
    problems of HoughLinesP on dimension lines. Hough is retained as a fallback.
    """
    cv2 = _import_cv2()
    gray = pil_to_gray_array(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    results: list[RawLine] = []
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(gray)[0]
        segments = [] if detected is None else detected.reshape(-1, 4)
        for x1, y1, x2, y2 in segments:
            length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            if length < min_length_px:
                continue
            # Keep near-horizontal/vertical architectural evidence. Slightly
            # rotated plans are retained; arbitrary diagonal hatching is not.
            dx, dy = abs(x2-x1), abs(y2-y1)
            if min(dx, dy) > 0 and max(dx, dy) / min(dx, dy) < 3.0:
                continue
            results.append(RawLine(
                line=Line(start=pixel_to_page_point(float(x1), float(y1), dpi),
                           end=pixel_to_page_point(float(x2), float(y2), dpi)),
                page=page_number, source=SourceKind.OPENCV_RASTER,
            ))
        if results:
            return results
    except Exception:
        pass

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    detected = cv2.HoughLinesP(edges, 1, 3.141592653589793 / 180.0, threshold=80,
                               minLineLength=min_length_px, maxLineGap=12)
    if detected is None:
        return []
    for seg in detected[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in seg]
        results.append(RawLine(
            line=Line(start=pixel_to_page_point(x1, y1, dpi),
                       end=pixel_to_page_point(x2, y2, dpi)),
            page=page_number, source=SourceKind.OPENCV_RASTER,
        ))
    return results

def detect_contours(
    image,
    page_number: int,
    dpi: float,
    min_area_px: float = 400.0,
    approx_epsilon_ratio: float = 0.01,
) -> tuple[list[RawPolygon], list[RawRectangle]]:
    """
    Connected-component / contour detection for wall evidence and raster
    geometry verification. Near-rectangular contours are also reported
    as RawRectangle for convenience.
    """
    cv2 = _import_cv2()

    gray = pil_to_gray_array(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[RawPolygon] = []
    rectangles: list[RawRectangle] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon_ratio * perimeter, True)
        if len(approx) < 3:
            continue
        pts_px = [(float(p[0][0]), float(p[0][1])) for p in approx]
        points = [pixel_to_page_point(x, y, dpi) for x, y in pts_px]
        try:
            polygon = Polygon(points=points)
        except Exception:  # pragma: no cover - degenerate polygon
            continue
        polygons.append(
            RawPolygon(
                polygon=polygon,
                page=page_number,
                source=SourceKind.OPENCV_RASTER,
                is_closed=True,
                area_pts2=polygon.area,
            )
        )
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            rectangles.append(
                RawRectangle(
                    bounding_box=pixel_bbox_to_page_bbox(x, y, x + w, y + h, dpi),
                    page=page_number,
                    source=SourceKind.OPENCV_RASTER,
                )
            )

    return polygons, rectangles


def geometry_evidence_for_page(image, page_number: int, dpi: float) -> dict:
    """Convenience wrapper returning every OpenCV evidence type for a page."""
    lines = detect_lines(image, page_number, dpi)
    polygons, rectangles = detect_contours(image, page_number, dpi)
    return {"lines": lines, "polygons": polygons, "rectangles": rectangles}


__all__ = [
    "pil_to_gray_array",
    "detect_lines",
    "detect_contours",
    "geometry_evidence_for_page",
]
