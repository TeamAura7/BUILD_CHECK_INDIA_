"""
OCR fallback extraction.

OCR is used ONLY when PDF-native text is missing/insufficient (scanned
page) or when raster interpretation is otherwise required. It is never
the first choice — see `pdf_native.has_sufficient_native_text`.

Backed by Tesseract via `pytesseract`. Swapping in PaddleOCR later only
requires a new function with the same return shape
(`list[RawTextItem]`), since nothing outside this module should know
which OCR engine ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.cv_extraction.coordinates import pixel_bbox_to_page_bbox
from backend.cv_extraction.raw_types import RawTextItem, SourceKind
from backend.tools.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

logger = get_logger(__name__)

DEFAULT_OCR_DPI = 300.0


def _import_ocr_stack():
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment issue, not logic
        raise RuntimeError(
            "pytesseract and Pillow are required for OCR fallback. Install with "
            "`pip install pytesseract pillow` and ensure the tesseract binary is on PATH."
        ) from exc

    # `pip install pytesseract` only installs the Python wrapper -- the actual
    # Tesseract binary is a separate OS-level install and is frequently not on
    # PATH (very common on Windows: pytesseract imports fine, then every call
    # fails with `TesseractNotFoundError`, which looks identical to "OCR is
    # missing" from the outside). Point pytesseract at an explicit path if the
    # operator configured one, instead of silently trusting PATH.
    try:
        from backend.config import get_settings

        cmd = get_settings().tesseract_cmd
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
    except Exception:  # pragma: no cover - config not available in some test contexts
        pass
    return pytesseract


def rasterize_page(page, dpi: float = DEFAULT_OCR_DPI) -> "Image.Image":
    """Rasterize a PyMuPDF page to a PIL Image at the given DPI."""
    from PIL import Image

    zoom = dpi / 72.0
    matrix = _matrix(zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img.convert("RGB")


def _matrix(zoom: float):
    import fitz  # type: ignore

    return fitz.Matrix(zoom, zoom)


def ocr_page(
    image: "Image.Image",
    page_number: int,
    dpi: float = DEFAULT_OCR_DPI,
    min_confidence: float = 0.0,
    rotation: int = 0,
    unrotated_size: tuple[float, float] | None = None,
) -> list[RawTextItem]:
    """
    Run OCR on a rasterized page image and return text items with bounding
    boxes converted back to PAGE_POINTS space.

    Returns TWO granularities, both in the same list:
      - one `RawTextItem` per individual word (Tesseract's native output
        granularity) -- precise bounding boxes, good for numeric-value
        candidates that need tight position information.
      - one additional `RawTextItem` per OCR *line* (`is_line_group=True`),
        built by grouping words sharing the same (block_num, par_num,
        line_num) and joining them in reading order. Multi-word phrase
        regexes ("SITE PLAN", "WIDE ROAD", area/height table labels) need
        this -- matching them against individual words can never succeed,
        since "SITE" and "PLAN" are two separate word-level detections.
    """
    pytesseract = _import_ocr_stack()
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        # Previously swallowed to `[]` here with only a logger.warning, which
        # meant a genuine, fixable problem (most commonly: pytesseract
        # imported fine but the `tesseract` binary itself isn't installed or
        # isn't on PATH -- pytesseract.TesseractNotFoundError, easy to
        # confuse with "OCR just found nothing") was invisible to callers.
        # Call sites already wrap this in their own try/except and turn it
        # into a proper warning message, so re-raising here surfaces the
        # real cause instead of hiding it.
        logger.warning("OCR failed for page", extra={"page": page_number, "error": str(exc)})
        raise

    items: list[RawTextItem] = []
    # (block_num, par_num, line_num) -> ordered list of (word_num, text, x, y, w, h, confidence)
    line_groups: dict[tuple[int, int, int], list[tuple[int, str, int, int, int, int, float]]] = {}

    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i]
        if not text or not text.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        confidence = max(0.0, min(1.0, conf / 100.0)) if conf >= 0 else 0.0
        if confidence < min_confidence:
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        px = _unrotate_pixel_bbox(x, y, x + w, y + h, rotation, *(unrotated_size or (0, 0)))
        bbox = pixel_bbox_to_page_bbox(*px, dpi)
        items.append(
            RawTextItem(
                text=text,
                bounding_box=bbox,
                page=page_number,
                source=SourceKind.OCR,
                ocr_confidence=confidence,
            )
        )
        key = (
            data.get("block_num", [0] * n)[i],
            data.get("par_num", [0] * n)[i],
            data.get("line_num", [0] * n)[i],
        )
        word_num = data.get("word_num", [0] * n)[i]
        line_groups.setdefault(key, []).append((word_num, text, x, y, w, h, confidence))

    for words in line_groups.values():
        if len(words) < 2:
            continue  # a single-word "line" is identical to its own word-level item already above
        words_sorted = sorted(words, key=lambda t: t[0])
        joined_text = " ".join(w[1] for w in words_sorted)
        min_x = min(w[2] for w in words_sorted)
        min_y = min(w[3] for w in words_sorted)
        max_x = max(w[2] + w[4] for w in words_sorted)
        max_y = max(w[3] + w[5] for w in words_sorted)
        avg_confidence = sum(w[6] for w in words_sorted) / len(words_sorted)
        px = _unrotate_pixel_bbox(
            min_x, min_y, max_x, max_y, rotation, *(unrotated_size or (0, 0))
        )
        bbox = pixel_bbox_to_page_bbox(*px, dpi)
        items.append(
            RawTextItem(
                text=joined_text,
                bounding_box=bbox,
                page=page_number,
                source=SourceKind.OCR,
                ocr_confidence=avg_confidence,
                is_line_group=True,
            )
        )
    return items


# --------------------------------------------------------------------------
# Rotated-sheet OCR
#
# A landscape architectural sheet is routinely stored as an unrotated
# portrait page with the whole drawing -- captions, tables and dimension
# labels included -- drawn rotated 90 degrees into it. The page reports
# /Rotate 0, so nothing in the file declares this; the only evidence is that
# the text renders sideways.
#
# Tesseract reads horizontal text. Given a sideways sheet it still returns
# output, which is what makes this failure quiet rather than loud: on PLAN5
# it recovered 522 items, enough to look like OCR had worked, but the
# area-statement table it needed was largely garbled and the site plan's
# own dimension labels ("17.59", "9.14", "13.09") were missed entirely.
# Building footprint and all four setbacks were unresolvable as a result.
#
# Re-running OCR on the rotated raster and mapping the boxes back costs one
# extra pass, and only on sheets that actually look sideways.
# --------------------------------------------------------------------------

# Fraction of word boxes that must be taller than they are wide before a page
# is treated as possibly sideways. Upright prose sits far below this; PLAN5
# measures 0.84.
_SIDEWAYS_TEXT_FRACTION = 0.30
_SIDEWAYS_MIN_ITEMS = 20


def _looks_sideways(items: list[RawTextItem]) -> bool:
    """Whether these OCR results suggest the page's text runs vertically."""
    words = [i for i in items if not i.is_line_group]
    if len(words) < _SIDEWAYS_MIN_ITEMS:
        return False
    tall = sum(1 for i in words if i.bounding_box.height > i.bounding_box.width)
    return (tall / len(words)) >= _SIDEWAYS_TEXT_FRACTION


def _unrotate_pixel_bbox(
    x0: float, y0: float, x1: float, y1: float,
    rotation: int, rotated_width: float, rotated_height: float,
) -> tuple[float, float, float, float]:
    """
    Map a pixel bbox measured in a rotated raster back to the original
    raster's pixel space.

    `rotation` is the counter-clockwise angle the image was turned by before
    OCR, and `rotated_width`/`rotated_height` are that turned image's own
    dimensions.
    """
    if rotation == 90:
        # CCW 90: original (x, y) -> rotated (y, W - x), where W is the
        # ORIGINAL width, which equals the rotated image's height.
        return (
            rotated_height - y1, x0,
            rotated_height - y0, x1,
        )
    if rotation == 270:
        # CCW 270 (i.e. clockwise 90): original (x, y) -> rotated (H - y, x),
        # with H the original height, equal to the rotated image's width.
        return (
            y0, rotated_width - x1,
            y1, rotated_width - x0,
        )
    if rotation == 180:
        return (
            rotated_width - x1, rotated_height - y1,
            rotated_width - x0, rotated_height - y0,
        )
    return x0, y0, x1, y1


def _ocr_confidence_mass(items: list[RawTextItem]) -> float:
    """
    How much readable text a pass recovered: characters weighted by OCR
    confidence. Used to choose between the two possible sideways rotations,
    since which one is correct cannot be known in advance -- the right one
    reads real words at high confidence, the wrong one reads mirrored
    nonsense at low confidence.
    """
    return sum(
        len(i.text.strip()) * (i.ocr_confidence or 0.0)
        for i in items
        if not i.is_line_group
    )


def _merge_ocr_items(
    primary: list[RawTextItem], extra: list[RawTextItem]
) -> list[RawTextItem]:
    """
    Combine two OCR passes, dropping items from `extra` that duplicate one
    already in `primary` (same text at substantially the same place).
    """
    def key(item: RawTextItem) -> tuple:
        b = item.bounding_box
        return (
            item.text.strip().lower(),
            item.is_line_group,
            round(b.center.x / 6.0),
            round(b.center.y / 6.0),
        )

    seen = {key(i) for i in primary}
    merged = list(primary)
    for item in extra:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        merged.append(item)
    return merged


def ocr_page_any_orientation(
    image: "Image.Image",
    page_number: int,
    dpi: float = DEFAULT_OCR_DPI,
    min_confidence: float = 0.0,
) -> list[RawTextItem]:
    """
    OCR a page, additionally reading it sideways when its text runs
    vertically.

    Always returns at least what the upright pass found, so this can replace
    a plain `ocr_page` call without losing anything. The rotated pass is only
    attempted when the upright result looks sideways, so upright sheets pay
    no extra cost.
    """
    upright = ocr_page(image, page_number, dpi=dpi, min_confidence=min_confidence)
    if not _looks_sideways(upright):
        return upright

    best: list[RawTextItem] = []
    best_mass = _ocr_confidence_mass(upright)
    for rotation in (90, 270):
        try:
            rotated = image.rotate(rotation, expand=True)
            candidate = ocr_page(
                rotated, page_number, dpi=dpi, min_confidence=min_confidence,
                rotation=rotation,
                unrotated_size=(rotated.width, rotated.height),
            )
        except Exception as exc:  # pragma: no cover - OCR/PIL environment issue
            logger.warning(
                "rotated OCR pass failed",
                extra={"page": page_number, "rotation": rotation, "error": str(exc)},
            )
            continue
        mass = _ocr_confidence_mass(candidate)
        if mass > best_mass:
            best, best_mass = candidate, mass

    if not best:
        return upright
    # Keep both passes: the rotated one reads the sideways body text, while
    # the upright one may still hold genuinely horizontal items (a stamp, a
    # north arrow label) that the rotated pass now sees sideways instead.
    return _merge_ocr_items(best, upright)


def needs_ocr(has_native_text: bool, native_text_char_count: int, force: bool = False) -> bool:
    """
    Decide whether a page needs the OCR fallback.

    Used when: PDF is scanned, vector text is missing, extraction is
    incomplete, or the caller explicitly requests raster interpretation.
    """
    if force:
        return True
    return (not has_native_text) or native_text_char_count < 3


__all__ = [
    "DEFAULT_OCR_DPI",
    "rasterize_page",
    "ocr_page",
    "ocr_page_any_orientation",
    "needs_ocr",
]
