"""
Visual debug output for the extraction pipeline.

Renders one PNG per page showing extracted text boxes, vector lines,
polygons, OCR boxes, and candidate dimensions, so a developer can
visually confirm what raw extraction actually found.
"""

from __future__ import annotations

from pathlib import Path

from backend.cv_extraction.raw_types import RawExtractionBundle, SourceKind

_COLORS = {
    "vector_line": (0, 102, 204),      # blue
    "vector_rect": (0, 153, 76),       # green
    "vector_poly": (0, 153, 76),       # green
    "opencv_line": (255, 140, 0),      # orange
    "opencv_poly": (204, 0, 204),      # magenta
    "pdf_text": (0, 0, 0),             # black
    "ocr_text": (204, 0, 0),           # red
    "dimension": (255, 204, 0),        # yellow highlight
}


def render_debug_overlays(
    document_path: Path,
    bundle: RawExtractionBundle,
    output_dir: Path,
    dpi: float = 150.0,
) -> list[Path]:
    """
    Render one annotated PNG per page. Returns the list of written paths.
    """
    from PIL import Image, ImageDraw

    from backend.cv_extraction import pdf_native
    from backend.cv_extraction.coordinates import points_per_pixel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pdf_native.open_document(document_path)
    written: list[Path] = []
    scale = 1.0 / points_per_pixel(dpi)  # page-points -> pixels at this render dpi

    try:
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            zoom = dpi / 72.0
            import fitz  # type: ignore

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
            draw = ImageDraw.Draw(image)

            for line in bundle.lines:
                if line.page != page_number:
                    continue
                color = _COLORS["opencv_line"] if line.source == SourceKind.OPENCV_RASTER else _COLORS["vector_line"]
                draw.line(
                    [
                        line.line.start.x * scale,
                        line.line.start.y * scale,
                        line.line.end.x * scale,
                        line.line.end.y * scale,
                    ],
                    fill=color,
                    width=2,
                )

            for rect in bundle.rectangles:
                if rect.page != page_number:
                    continue
                bbox = rect.bounding_box
                draw.rectangle(
                    [bbox.min_x * scale, bbox.min_y * scale, bbox.max_x * scale, bbox.max_y * scale],
                    outline=_COLORS["vector_rect"],
                    width=2,
                )

            for poly in bundle.polygons:
                if poly.page != page_number:
                    continue
                pts = [(p.x * scale, p.y * scale) for p in poly.polygon.points]
                color = _COLORS["opencv_poly"] if poly.source == SourceKind.OPENCV_RASTER else _COLORS["vector_poly"]
                if len(pts) >= 2:
                    draw.line(pts + [pts[0]], fill=color, width=2)

            for item in bundle.text_evidence:
                if item.page != page_number:
                    continue
                bbox = item.bounding_box
                color = _COLORS["ocr_text"] if item.source == SourceKind.OCR else _COLORS["pdf_text"]
                draw.rectangle(
                    [bbox.min_x * scale, bbox.min_y * scale, bbox.max_x * scale, bbox.max_y * scale],
                    outline=color,
                    width=1,
                )

            for cand in bundle.dimensions_candidates:
                if cand.page != page_number:
                    continue
                bbox = cand.bounding_box
                draw.rectangle(
                    [bbox.min_x * scale, bbox.min_y * scale, bbox.max_x * scale, bbox.max_y * scale],
                    outline=_COLORS["dimension"],
                    width=3,
                )

            out_path = output_dir / f"{bundle.document_id}_page{page_number}_debug.png"
            image.save(out_path)
            written.append(out_path)
    finally:
        doc.close()

    return written


__all__ = ["render_debug_overlays"]
