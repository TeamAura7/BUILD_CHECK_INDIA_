from __future__ import annotations

from pathlib import Path

import fitz


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: float = 200.0) -> list[dict]:
    """Render PDF pages to PNG files and return page metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[dict] = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            output_path = output_dir / f"page_{page_index + 1}.png"
            pix.save(str(output_path))
            pages.append({
                "page_number": page_index + 1,
                "image_path": str(output_path),
                "width_px": pix.width,
                "height_px": pix.height,
            })
    return pages


def render_pdf_region(
    pdf_path: Path,
    page_index: int,
    bbox_pts: tuple[float, float, float, float],
    output_path: Path,
    dpi: float = 400.0,
) -> dict:
    """Render ONE clipped region of one PDF page directly from the vector page, at `dpi`.

    This is a genuine higher-resolution re-render of just the region (via
    PyMuPDF's own `clip=` support), not an upscale-by-cropping of an
    already-rasterized full-page PNG. A focused Vision pass reading small
    printed text (e.g. setback decimals like ".46"/".47", or a coverage %
    tucked into a table cell) needs the extra real pixel detail this gives
    it -- cropping a 200 DPI full-page render to a small region just makes
    the same handful of blurry pixels bigger, it does not add information.

    `bbox_pts` is (x0, y0, x1, y1) in the page's own point space (same frame
    as `page.get_drawings()` / `CoordinateSpace.PAGE_POINTS` elsewhere in
    this project).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    x0, y0, x1, y1 = bbox_pts
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        pix.save(str(output_path))
        return {
            "image_path": str(output_path),
            "width_px": pix.width,
            "height_px": pix.height,
            "bbox_pts": [x0, y0, x1, y1],
        }
