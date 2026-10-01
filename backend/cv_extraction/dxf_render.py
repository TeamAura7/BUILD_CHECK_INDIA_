"""
Self-contained DXF -> raster image rendering for the Vision fallback path
(`dxf_extractor.py`'s `_vision_fallback_regions`).

Deliberately NOT using `ezdxf.addons.drawing`'s page-fit renderer: its
world<->pixel coordinate mapping (page auto-sizing, unit handling, how
`render_box` interacts with content-bbox auto-fit) proved difficult to
pin down and verify empirically during development -- an untrusted
transform here would turn every Vision-identified region bbox into a
confidently WRONG real-world measurement once mapped back to DXF
coordinates, exactly the failure mode this whole project exists to
prevent. Rasterizing the handful of primitive shapes actually needed here
(open/closed polyline chains) with a small, self-computed, directly-
verifiable affine transform is simpler to reason about and trust than a
third-party page-layout black box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class RasterTransform:
    """
    Affine world -> pixel mapping used for exactly one rendered image.

    pixel_x = (world_x - min_x) * scale
    pixel_y = height_px - (world_y - min_y) * scale   (Y flips: image
    pixel-y grows downward, DXF world-y grows upward)
    """

    min_x: float
    min_y: float
    scale: float  # pixels per world unit
    width_px: int
    height_px: int

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.min_x) * self.scale,
            self.height_px - (y - self.min_y) * self.scale,
        )

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        return (
            px / self.scale + self.min_x,
            (self.height_px - py) / self.scale + self.min_y,
        )

    def pixel_bbox_to_world_bbox(self, bbox_px: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """(px0,py0,px1,py1) -> (min_x,min_y,max_x,max_y) in world units."""
        x0, y0 = self.pixel_to_world(bbox_px[0], bbox_px[1])
        x1, y1 = self.pixel_to_world(bbox_px[2], bbox_px[3])
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def compute_transform(
    min_x: float, min_y: float, max_x: float, max_y: float,
    target_max_px: int = 1600, margin_fraction: float = 0.04,
) -> RasterTransform:
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    margin = margin_fraction * max(width, height)
    width += 2 * margin
    height += 2 * margin
    scale = target_max_px / max(width, height)
    width_px = max(1, round(width * scale))
    height_px = max(1, round(height * scale))
    return RasterTransform(
        min_x=min_x - margin, min_y=min_y - margin, scale=scale,
        width_px=width_px, height_px=height_px,
    )


def render_polylines_to_image(
    closed_polylines: Sequence[Sequence[tuple[float, float]]],
    open_polylines: Sequence[Sequence[tuple[float, float]]],
    output_path: Path,
    target_max_px: int = 1600,
) -> Optional[RasterTransform]:
    """
    Rasterize DXF line-work (closed polygon outlines + open polyline/line
    chains) to a PNG, white background / black strokes, using a self-
    computed affine transform. Returns the transform (needed to map any
    Vision-identified pixel bbox back to DXF world coordinates), or None
    if there was nothing to draw.
    """
    from PIL import Image, ImageDraw

    all_points = [pt for chain in closed_polylines for pt in chain] + [
        pt for chain in open_polylines for pt in chain
    ]
    if not all_points:
        return None
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    transform = compute_transform(min(xs), min(ys), max(xs), max(ys), target_max_px=target_max_px)

    img = Image.new("RGB", (transform.width_px, transform.height_px), "white")
    draw = ImageDraw.Draw(img)
    for chain in open_polylines:
        if len(chain) >= 2:
            draw.line([transform.world_to_pixel(x, y) for x, y in chain], fill="black", width=1)
    for chain in closed_polylines:
        if len(chain) >= 2:
            px_pts = [transform.world_to_pixel(x, y) for x, y in chain]
            draw.line(px_pts + [px_pts[0]], fill="black", width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return transform


__all__ = ["RasterTransform", "compute_transform", "render_polylines_to_image"]
