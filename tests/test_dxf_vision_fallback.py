"""
Tests for the DXF Vision-render fallback: when a DXF has no closed
building/road polygon the deterministic geometry search can identify (e.g.
a scanned sheet that got vectorized/traced, where walls were never
captured as one closed polyline), the extractor renders the DXF's own
line-work to an image and asks Vision to visually point at which region is
the building -- then computes the actual measurement from exact DXF world
coordinates within that region, never from anything Vision itself reports
as a number.

These tests never make a real API call: `get_vision_extractor` is
monkeypatched to a stub that returns a canned `VisionPageResult`, so the
coordinate-mapping and wiring logic is verified deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cv_extraction import dxf_render
from backend.cv_extraction.dxf_extractor import (
    DXFHybridExtractor,
    _normalize_vision_bbox,
    _vision_fallback_regions,
)
from backend.config import get_settings
from backend.schemas.geometry import BoundingBox
from backend.schemas.vision import VisionPageResult, VisionRegion


# --- dxf_render: transform correctness ---------------------------------------


def test_render_transform_round_trips_world_coordinates(tmp_path):
    closed = [[(0, 0), (20, 0), (20, 10), (0, 10)]]
    out = tmp_path / "render.png"
    transform = dxf_render.render_polylines_to_image(closed, [], out, target_max_px=800)
    assert transform is not None
    assert out.exists()
    px = transform.world_to_pixel(5.0, 5.0)
    world_back = transform.pixel_to_world(*px)
    assert world_back[0] == pytest.approx(5.0, abs=1e-6)
    assert world_back[1] == pytest.approx(5.0, abs=1e-6)


def test_render_transform_pixel_bbox_to_world_bbox_matches_input():
    closed = [[(0, 0), (20, 0), (20, 10), (0, 10)]]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        transform = dxf_render.render_polylines_to_image(closed, [], Path(d) / "r.png", target_max_px=800)
    px0 = transform.world_to_pixel(2.0, 2.0)
    px1 = transform.world_to_pixel(8.0, 6.0)
    world_bbox = transform.pixel_bbox_to_world_bbox((*px0, *px1))
    assert world_bbox[0] == pytest.approx(2.0, abs=1e-6)
    assert world_bbox[1] == pytest.approx(2.0, abs=1e-6)
    assert world_bbox[2] == pytest.approx(8.0, abs=1e-6)
    assert world_bbox[3] == pytest.approx(6.0, abs=1e-6)


def test_render_returns_none_with_no_geometry(tmp_path):
    assert dxf_render.render_polylines_to_image([], [], tmp_path / "empty.png") is None


# --- _normalize_vision_bbox: normalized-grid vs raw-pixel detection ----------


def test_normalize_vision_bbox_detects_normalized_grid():
    # Small values on a large render -> normalized 0-1000 grid.
    result = _normalize_vision_bbox([100, 200, 300, 400], width_px=1600, height_px=1200)
    assert result == pytest.approx((160.0, 240.0, 480.0, 480.0))


def test_normalize_vision_bbox_detects_raw_pixels_on_small_render():
    # Both the bbox and the render are small -- can't be normalized-grid.
    result = _normalize_vision_bbox([10, 20, 100, 200], width_px=300, height_px=250)
    assert result == pytest.approx((10.0, 20.0, 100.0, 200.0))


def test_normalize_vision_bbox_handles_missing_bbox():
    assert _normalize_vision_bbox(None, 800, 600) is None
    assert _normalize_vision_bbox([1, 2, 3], 800, 600) is None


# --- End-to-end wiring, with a stubbed Vision backend ------------------------


class _StubExtractor:
    def __init__(self, regions):
        self._regions = regions

    def analyze_image(self, image_path, page_number, native_spans=None, page_width_pts=0.0,
                       page_height_pts=0.0, *, ground_against_native_text=True,
                       prompt_override=None, max_new_tokens=None):
        return VisionPageResult(page_number=page_number, regions=self._regions)


def test_vision_fallback_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "vision_enabled", False)
    result = _vision_fallback_regions("doc", [], [[(0, 0), (10, 0)]], [])
    assert result == {}


def test_vision_fallback_maps_identified_bbox_to_world_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "vision_enabled", True)
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path)

    # A 20x10 plot with wall line-work (open chains, no closed building
    # polygon) roughly outlining a building from (5,3) to (15,7).
    open_chains = [
        [(5, 3), (15, 3)], [(15, 3), (15, 7)], [(15, 7), (5, 7)], [(5, 7), (5, 3)],
    ]
    closed = [[(0, 0), (20, 0), (20, 10), (0, 10)]]

    # Vision "sees" the building roughly where it actually is -- expressed
    # on the normalized 0-1000 grid the real backends use.
    stub_region = VisionRegion(id="r1", type="BUILDING", bbox=[240, 280, 760, 680], confidence=0.8)
    monkeypatch.setattr("backend.vision_extraction.get_vision_extractor", lambda: _StubExtractor([stub_region]))

    result = _vision_fallback_regions("doc", [], open_chains + [pts for pts in closed], [])
    # closed passed as open here is fine for rendering purposes (this test
    # only exercises rendering + the bbox mapping, not polygon closing).
    assert "BUILDING" in result
    bbox = result["BUILDING"]
    # The stub bbox covers ~24%-76% of the 0-1000 grid in both axes, over a
    # render whose content (0,0)-(20,10) plus margin -- expect it to land
    # roughly over the real (5,3)-(15,7) building area, not exactly (bbox
    # mapping through a padded render), but well within the plot.
    assert 0.0 <= bbox.min_x <= 10.0
    assert 0.0 <= bbox.min_y <= 10.0
    assert bbox.max_x > bbox.min_x
    assert bbox.max_y > bbox.min_y


def test_reconstruction_resolves_building_from_disconnected_wall_lines_without_vision(tmp_path, monkeypatch):
    """
    A DXF whose building is drawn as four disconnected LINE entities that
    DO form a closed loop once their endpoints are snapped together (never
    a single closed polyline, so the plain deterministic polygon search
    alone would leave building.* MISSING) is now resolved by
    `dxf_reconstruction.reconstruct_building_polygon` BEFORE Vision is ever
    consulted -- with Vision left disabled here, proving the path is
    genuinely model-independent.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 10), (0, 10)], close=True, dxfattribs={"layer": "0"})
    # Building walls as four disconnected LINE entities -- never a closed polygon.
    msp.add_line((5, 3), (15, 3), dxfattribs={"layer": "0"})
    msp.add_line((15, 3), (15, 7), dxfattribs={"layer": "0"})
    msp.add_line((15, 7), (5, 7), dxfattribs={"layer": "0"})
    msp.add_line((5, 7), (5, 3), dxfattribs={"layer": "0"})
    dxf_path = tmp_path / "open_walls.dxf"
    doc.saveas(str(dxf_path))

    monkeypatch.setattr(get_settings(), "vision_enabled", False)

    result = DXFHybridExtractor().extract(dxf_path, "open_walls")
    fields = {m.field: m for m in result.independent_cv.measurements}

    assert "building.footprint_area" in fields
    m = fields["building.footprint_area"]
    assert m.source == "VECTOR_GEOMETRY_RECONSTRUCTED"
    assert m.confidence <= 0.82
    assert m.value == pytest.approx(40.0, abs=0.5)  # exact 10 x 4 rectangle

    assert "building.width" in fields
    assert fields["building.width"].source == "VECTOR_GEOMETRY_RECONSTRUCTED"


def test_extract_uses_vision_fallback_when_reconstruction_also_finds_nothing(tmp_path, monkeypatch):
    """
    Full integration: a DXF whose building line-work has a real gap (one
    wall entirely missing, so no closed loop exists at any snap tolerance)
    cannot be resolved by reconstruction either -- it still resolves
    building.width/depth/footprint_area once Vision is enabled and
    identifies the region, tagged with the distinct VISION_DXF_RENDER
    source and a capped confidence, never silently indistinguishable from
    exact vector geometry.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (20, 0), (20, 10), (0, 10)], close=True, dxfattribs={"layer": "0"})
    # Building walls as THREE disconnected LINE entities -- one side (the
    # "top") is entirely missing, so no closed loop exists at any snap
    # tolerance and reconstruction must correctly return None.
    msp.add_line((5, 3), (15, 3), dxfattribs={"layer": "0"})
    msp.add_line((15, 3), (15, 7), dxfattribs={"layer": "0"})
    msp.add_line((5, 7), (5, 3), dxfattribs={"layer": "0"})
    dxf_path = tmp_path / "open_walls_gap.dxf"
    doc.saveas(str(dxf_path))

    monkeypatch.setattr(get_settings(), "vision_enabled", True)
    monkeypatch.setattr(get_settings(), "upload_dir", tmp_path / "uploads")
    stub_region = VisionRegion(id="r1", type="BUILDING", bbox=[240, 280, 760, 680], confidence=0.8)
    monkeypatch.setattr("backend.vision_extraction.get_vision_extractor", lambda: _StubExtractor([stub_region]))

    result = DXFHybridExtractor().extract(dxf_path, "open_walls_gap")
    fields = {m.field: m for m in result.independent_cv.measurements}

    assert "building.footprint_area" in fields
    m = fields["building.footprint_area"]
    assert m.source == "VISION_DXF_RENDER"
    assert m.confidence <= 0.65
    assert m.value is not None and m.value > 0
    # Vision's coarse stub bbox over the true (5,3)-(15,7) 10x4=40 m2
    # building should land in a plausible ballpark once mapped through the
    # padded render transform, not be wildly off.
    assert 20.0 <= m.value <= 80.0

    assert "building.width" in fields
    assert fields["building.width"].source == "VISION_DXF_RENDER"
