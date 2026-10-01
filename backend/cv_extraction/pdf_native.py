"""
PDF-native extraction: vector geometry (lines/rects/polygons/paths) and
text with bounding boxes, via PyMuPDF.

This module never makes semantic decisions (no "this is a dimension",
no "this is the plot boundary") — it just faithfully lifts what's in the
PDF's content stream into `raw_types` primitives, in PAGE_POINTS space.

Never rely on this alone: a scanned/rasterized page will yield an empty
result here, which is exactly the signal `ocr_fallback.py` uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.cv_extraction.coordinates import normalize_rotation
from backend.cv_extraction.raw_types import (
    PageMetadata,
    RawLine,
    RawPolygon,
    RawRectangle,
    RawTextItem,
    SourceKind,
)
from backend.schemas.geometry import BoundingBox, Line, Point, Polygon
from backend.tools.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import fitz  # PyMuPDF

logger = get_logger(__name__)

# Below this many non-whitespace characters, treat a page as text-empty
# (i.e. a scanned page) rather than "has native text".
MIN_NATIVE_TEXT_CHARS = 3


def _import_fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment issue, not logic
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF-native extraction. Install with "
            "`pip install pymupdf`."
        ) from exc
    return fitz


def extract_page_metadata(page: "fitz.Page", page_number: int) -> PageMetadata:
    rect = page.rect
    rotation = normalize_rotation(getattr(page, "rotation", 0) or 0)
    return PageMetadata(
        page_number=page_number,
        width_pts=float(rect.width),
        height_pts=float(rect.height),
        rotation_degrees=rotation,
    )


def extract_text_items(page: "fitz.Page", page_number: int) -> list[RawTextItem]:
    """
    Extract every text span on a page with its bounding box, font size,
    and orientation. Never flattens to plain text.

    IMPORTANT (found via a real rotated architectural sheet, not caught
    by any synthetic fixture): `page.get_text("dict")` returns
    coordinates in the page's *unrotated* (raw mediabox) space, and so
    does `page.get_drawings()` (used by `extract_vector_geometry`
    below) -- an earlier version of this note claimed drawings already
    came back in display space, which is wrong and left rotated pages
    broken in the opposite direction. BOTH are transformed through
    `page.rotation_matrix` into the *rotated* display space that
    matches `page.rect` / `PageMetadata.width_pts/height_pts`. For an
    unrotated page (rotation == 0, by far the common case) these are
    identical and this made no visible difference — but for a rotated
    page (rotation in {90, 180, 270}, common for landscape architectural
    sheets stored as rotated-portrait PDFs) text and vector geometry
    silently ended up in two DIFFERENT coordinate frames. Every
    downstream proximity/distance computation between text and geometry
    (dimension-to-line association, plot/site label-proximity scoring,
    room-label rejection in building resolution) was comparing positions
    that had no real relationship to each other, while never raising a
    visible error — the numbers were just wrong.

    Every text bounding box (and its orientation) is transformed through
    `page.rotation_matrix` before being stored, matching the space
    `extract_vector_geometry` already uses. For rotation == 0 this
    matrix is the identity, so this is a strict correctness fix with no
    behavior change for unrotated pages/documents.
    """
    items: list[RawTextItem] = []
    rot_matrix = getattr(page, "rotation_matrix", None)
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block, 1 = image block
            continue
        for line in block.get("lines", []):
            wmode_dir = line.get("dir", (1.0, 0.0))
            dx, dy = wmode_dir[0], wmode_dir[1]
            if rot_matrix is not None:
                # Rotate the direction vector by the same (purely
                # rotational, no translation/scale) linear transform
                # applied to positions below.
                dx, dy = (
                    rot_matrix.a * wmode_dir[0] + rot_matrix.c * wmode_dir[1],
                    rot_matrix.b * wmode_dir[0] + rot_matrix.d * wmode_dir[1],
                )
            orientation = None
            try:
                import math

                orientation = math.degrees(math.atan2(-dy, dx)) % 360
            except Exception:  # pragma: no cover - defensive
                orientation = None
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                if rot_matrix is not None:
                    import fitz  # noqa: F401 (already required by this module; local import keeps this fn self-contained)

                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    transformed = [fitz.Point(px, py) * rot_matrix for px, py in corners]
                    xs = [p.x for p in transformed]
                    ys = [p.y for p in transformed]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                items.append(
                    RawTextItem(
                        text=text,
                        bounding_box=BoundingBox(min_x=x0, min_y=y0, max_x=x1, max_y=y1),
                        page=page_number,
                        source=SourceKind.PDF_TEXT,
                        font_size=span.get("size"),
                        orientation_degrees=orientation,
                    )
                )
    return items


def _rect_from_quad_points(points: list[tuple[float, float]]) -> BoundingBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BoundingBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _is_axis_aligned_rect(points: list[tuple[float, float]], tol: float = 1e-2) -> bool:
    if len(points) not in (4, 5):
        return False
    pts = points[:4]
    xs = sorted({round(p[0], 3) for p in pts})
    ys = sorted({round(p[1], 3) for p in pts})
    return len(xs) == 2 and len(ys) == 2


def extract_vector_geometry(
    page: "fitz.Page", page_number: int
) -> tuple[list[RawLine], list[RawRectangle], list[RawPolygon]]:
    """
    Extract lines, rectangles, and polygons from the page's vector
    drawing commands (strokes/fills of paths), via PyMuPDF's
    `get_drawings()`.
    """
    lines: list[RawLine] = []
    rectangles: list[RawRectangle] = []
    polygons: list[RawPolygon] = []

    # Put vector geometry in the SAME space as `extract_text_items` output.
    #
    # The note in that function says `get_drawings()` already returns
    # rotated/display-space coordinates while `get_text("dict")` does not, so
    # only text needed transforming. That is not what actually happens: on a
    # /Rotate 270 sheet (PLAN4) the text comes back spanning the landscape
    # page (x up to 1630 of 1684) while the drawings come back spanning the
    # portrait mediabox (y up to 1634 of 1684) -- i.e. the two are a quarter
    # turn apart, and the transform is needed on BOTH sides.
    #
    # The failure is silent and total: every text-to-geometry distance on
    # such a page compares positions from two different coordinate systems,
    # so dimension-to-line association, plot-label proximity and site-region
    # search all quietly match nothing. PLAN4 reconstructed zero rectangles
    # anywhere near its site plan for exactly this reason.
    #
    # For an unrotated page `rotation_matrix` is the identity, so this is a
    # no-op on every non-rotated sheet.
    rot_matrix = getattr(page, "rotation_matrix", None)

    def _pt(point) -> Point:
        if rot_matrix is None:
            return Point(x=point.x, y=point.y)
        transformed = point * rot_matrix
        return Point(x=transformed.x, y=transformed.y)

    fitz = _import_fitz()
    fitz_point = fitz.Point

    try:
        drawings = page.get_drawings()
    except Exception as exc:  # pragma: no cover - defensive, malformed content stream
        logger.warning("failed to read vector drawings", extra={"page": page_number, "error": str(exc)})
        return lines, rectangles, polygons

    for path in drawings:
        stroke_width = path.get("width")
        items = path.get("items", [])
        # Collect all points touched by this path's line/curve segments.
        poly_points: list[tuple[float, float]] = []
        for op in items:
            kind = op[0]
            if kind == "l":  # line: (p1, p2)
                start, end = _pt(op[1]), _pt(op[2])
                lines.append(
                    RawLine(
                        line=Line(start=start, end=end),
                        page=page_number,
                        source=SourceKind.VECTOR_PDF,
                        stroke_width=stroke_width,
                    )
                )
                poly_points.extend([(start.x, start.y), (end.x, end.y)])
            elif kind == "re":  # rectangle: (Rect, rotate)
                rect = op[1]
                # Transform all four corners: a 90/270 turn keeps the
                # rectangle axis-aligned but swaps its width and height, so
                # taking the bbox of the transformed corners is both correct
                # and rotation-agnostic.
                corners = [
                    _pt(fitz_point(rect.x0, rect.y0)), _pt(fitz_point(rect.x1, rect.y0)),
                    _pt(fitz_point(rect.x1, rect.y1)), _pt(fitz_point(rect.x0, rect.y1)),
                ]
                xs = [c.x for c in corners]
                ys = [c.y for c in corners]
                rectangles.append(
                    RawRectangle(
                        bounding_box=BoundingBox(
                            min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys)
                        ),
                        page=page_number,
                        source=SourceKind.VECTOR_PDF,
                    )
                )
                poly_points.extend([(c.x, c.y) for c in corners])
            elif kind in ("c", "qu"):  # curve/quad — approximate with endpoints
                pts = [_pt(p) for p in op[1:] if hasattr(p, "x")]
                poly_points.extend([(p.x, p.y) for p in pts])

        if len(poly_points) >= 3:
            deduped = list(dict.fromkeys(poly_points))
            if len(deduped) >= 3 and not _is_axis_aligned_rect(deduped):
                try:
                    polygons.append(
                        RawPolygon(
                            polygon=Polygon(points=[Point(x=x, y=y) for x, y in deduped]),
                            page=page_number,
                            source=SourceKind.VECTOR_PDF,
                            is_closed=path.get("closePath", False),
                        )
                    )
                except Exception:  # pragma: no cover - degenerate polygon
                    pass

    return lines, rectangles, polygons


def has_sufficient_native_text(text_items: list[RawTextItem]) -> bool:
    total_chars = sum(len(t.text.strip()) for t in text_items)
    return total_chars >= MIN_NATIVE_TEXT_CHARS


def open_document(document_path):
    fitz = _import_fitz()
    return fitz.open(str(document_path))


__all__ = [
    "extract_page_metadata",
    "extract_text_items",
    "extract_vector_geometry",
    "has_sufficient_native_text",
    "open_document",
    "MIN_NATIVE_TEXT_CHARS",
]
