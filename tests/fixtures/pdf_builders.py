"""
Synthetic PDF builders for Phase 2 extraction tests.

We don't have real architectural plan PDFs checked into the repo
(`data/test_plans/` is empty pending real sample uploads), so these
builders generate small, purpose-built PDFs that exercise each pipeline
path: native vector geometry + text, a fully rasterized/scanned page
(no selectable text), rotated pages, and non-default page sizes.

No fixture encodes a specific expected numeric answer that the pipeline
"already knows" — tests assert on structural/behavioral properties
(e.g. "some text was found", "OCR fired for the scanned page") rather
than hard-coding output values, per phase2.md's NO OVERFITTING rule.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def build_vector_plan_pdf(path: Path, width: float = 612, height: float = 792) -> Path:
    """
    A page with native text AND vector drawing (a plot outline rectangle,
    a smaller building outline rectangle inside it, and a dimension line
    with a text label) — simulates a simple architectural plan sheet.
    """
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)

    # Plot outline (large rectangle)
    plot_rect = fitz.Rect(60, 60, width - 60, height - 120)
    page.draw_rect(plot_rect, color=(0, 0, 0), width=1.5)

    # Building outline (smaller rectangle, inside the plot)
    building_rect = fitz.Rect(
        plot_rect.x0 + 40, plot_rect.y0 + 60, plot_rect.x1 - 60, plot_rect.y1 - 80
    )
    page.draw_rect(building_rect, color=(0, 0, 0), width=1.0)

    # A dimension line along the top of the plot, with a text label near it
    dim_y = plot_rect.y0 - 15
    page.draw_line(fitz.Point(plot_rect.x0, dim_y), fitz.Point(plot_rect.x1, dim_y))
    page.insert_text((plot_rect.x0 + 5, dim_y - 4), "12.50 m", fontsize=9)

    # A road label near a long thin strip below the plot (road candidate bait)
    road_rect = fitz.Rect(plot_rect.x0, plot_rect.y1 + 20, plot_rect.x1, plot_rect.y1 + 32)
    page.draw_rect(road_rect, color=(0, 0, 0), width=1.0)
    page.insert_text((plot_rect.x0, plot_rect.y1 + 45), "9.0 m WIDE ROAD", fontsize=9)

    # A feet-inches style label
    page.insert_text((plot_rect.x0 + 5, plot_rect.y0 + 20), "SETBACK 3'-6\"", fontsize=8)

    # Plan title text
    page.insert_text((60, 30), "SAMPLE ARCHITECTURAL PLAN - SITE LAYOUT", fontsize=11)

    doc.save(str(path))
    doc.close()
    return path


def build_text_only_pdf(path: Path) -> Path:
    """A page with native text but no vector drawing at all."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "PLOT AREA: 216.0 sq.m", fontsize=10)
    page.insert_text((72, 90), "BUILDING WIDTH: 9.0 m", fontsize=10)
    page.insert_text((72, 108), "FRONT SETBACK: 3.0 m", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def build_scanned_pdf(path: Path, width: int = 612, height: int = 792) -> Path:
    """
    A page with NO selectable text and NO vector drawing commands — only
    a raster image embedded on the page, simulating a scanned plan.
    Text is rendered as an image so no PDF text layer exists at all.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, width - 50, height - 100], outline="black", width=3)
    draw.text((60, 20), "SCANNED PLAN - PLOT 216 SQM", fill="black")
    draw.text((60, height - 90), "ROAD WIDTH 9M", fill="black")

    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    rect = fitz.Rect(0, 0, width, height)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


def build_scanned_pdf_with_sideways_text(path: Path, width: int = 612, height: int = 792) -> Path:
    """
    A scanned page (no PDF text layer, raster image only) whose text is
    baked into the image sideways (the whole page rotated 90 degrees before
    being scanned) -- the case `ocr_fallback.ocr_page_any_orientation`
    exists to recover and a plain upright `ocr_page` reads as garbage
    (audit finding: the main PDFHybridExtractor pass previously used the
    upright-only OCR path here; see BUILDCHECK_FORENSIC_AUDIT.md Sec
    4.1/5.4). Enough repeated words are rendered to clear
    `ocr_fallback._SIDEWAYS_MIN_ITEMS`/`_SIDEWAYS_TEXT_FRACTION`.
    """
    from PIL import Image, ImageDraw

    # Render upright first (a portrait page, narrow enough to rotate into
    # the target width x height once turned 90 degrees), then rotate the
    # whole raster -- this is exactly the "landscape sheet scanned/stored
    # as an unrotated portrait page" failure mode being reproduced, not a
    # PDF-level /Rotate flag (build_rotated_pdf covers that, separate case).
    upright = Image.new("RGB", (height, width), color="white")
    draw = ImageDraw.Draw(upright)
    lines = ["PLOT BOUNDARY LINE", "ROAD WIDTH NINE METRE", "SITE AREA TWO HUNDRED SQM"] * 4
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 24
    sideways = upright.rotate(90, expand=True)

    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    rect = fitz.Rect(0, 0, width, height)
    import io

    buf = io.BytesIO()
    sideways.save(buf, format="PNG")
    page.insert_image(rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
    return path


def build_rotated_pdf(path: Path, rotation: int = 90) -> Path:
    """A page with native text/vector content and a non-zero /Rotate entry."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(80, 80, 500, 700), color=(0, 0, 0), width=1.5)
    page.insert_text((100, 60), "ROTATED PLAN SHEET", fontsize=10)
    page.set_rotation(rotation)
    doc.save(str(path))
    doc.close()
    return path


def build_multi_page_size_pdf(path: Path) -> Path:
    """A multi-page document mixing A4-ish and Letter-ish page sizes."""
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)  # A4 in points
    p1.insert_text((50, 50), "PAGE 1 - A4", fontsize=10)
    p1.draw_rect(fitz.Rect(50, 70, 545, 780), color=(0, 0, 0), width=1)

    p2 = doc.new_page(width=612, height=792)  # US Letter in points
    p2.insert_text((50, 50), "PAGE 2 - LETTER", fontsize=10)
    p2.draw_rect(fitz.Rect(50, 70, 560, 730), color=(0, 0, 0), width=1)

    doc.save(str(path))
    doc.close()
    return path


def build_missing_text_pdf(path: Path) -> Path:
    """A page with vector drawing but genuinely zero text of any kind."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(60, 60, 500, 700), color=(0, 0, 0), width=1.5)
    page.draw_line(fitz.Point(60, 60), fitz.Point(500, 700))
    doc.save(str(path))
    doc.close()
    return path


__all__ = [
    "build_vector_plan_pdf",
    "build_text_only_pdf",
    "build_scanned_pdf",
    "build_scanned_pdf_with_sideways_text",
    "build_rotated_pdf",
    "build_multi_page_size_pdf",
    "build_missing_text_pdf",
]
