"""
Unit tests for `backend.cv_extraction.dxf_regions` -- generic, coordinate-
and filename-agnostic spatial clustering and sheet-border/frame detection.

These construct synthetic point clouds directly (no DXF I/O) so the
clustering/frame-detection algorithm itself is pinned independently of the
DXF entity-collection layer -- see `test_dxf_extractor.py` for end-to-end
DXF-level coverage of the same capabilities.
"""
from __future__ import annotations

from backend.cv_extraction.dxf_regions import (
    build_regions,
    cluster_entity_points,
    detect_frame_polygon_indices,
)
from backend.schemas.geometry import BoundingBox


def _grid_points(x0, y0, x1, y1, n=20):
    """A dense grid of points filling a rectangle -- stands in for one
    drawing's worth of geometry (many entities close together). Density
    matters here: the clustering algorithm sizes its grid cell relative to
    the OVERALL extent of everything being clustered (long_axis / 90), so
    a fixture needs `n` large enough that its own point spacing (extent /
    (n-1)) is smaller than that cell -- independent of the fixture's
    absolute physical size, this requires roughly n > 90 for a single,
    self-contained cluster (where the overall extent IS this cluster's own
    extent), or a smaller `n` when several such clusters are combined into
    one larger overall extent (see the two-region test below)."""
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append((x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * j / (n - 1)))
    return pts


def test_single_drawing_yields_one_region():
    """A DXF with only one drawing must cluster into exactly one region --
    this is the common case (this project's existing single-view synthetic
    fixtures) and must never be split spuriously."""
    entities = [[(x, y)] for x, y in _grid_points(0, 0, 10, 10, n=100)]
    clusters = cluster_entity_points(entities)
    assert len(clusters) == 1
    # Some sparse edge/corner points of the cluster's own periphery may be
    # eroded away as below-floor-density, same as a real drawing's own
    # outermost annotation reach -- the important property is that
    # everything stays in ONE region, not split into several.
    assert len(clusters[0]) >= len(entities) * 0.8


def test_two_widely_separated_drawings_yield_two_regions():
    """Two dense clusters of geometry, far apart with empty space between
    them, must be recognized as two distinct regions."""
    left = [[(x, y)] for x, y in _grid_points(0, 0, 5, 5)]
    right = [[(x, y)] for x, y in _grid_points(50, 50, 55, 55)]
    entities = left + right
    clusters = cluster_entity_points(entities)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    # Some sparse edge points of each cluster may be eroded away (see
    # above) -- the important property is that each region stays
    # overwhelmingly intact and separate, not merged with the other.
    assert sizes[0] >= 400 * 0.8
    assert sizes[1] >= 400 * 0.8


def test_thin_connecting_stroke_does_not_bridge_two_drawings():
    """A single long, thin entity connecting two otherwise-separate dense
    clusters (e.g. one long dimension/leader line, or a sheet border edge
    passing near both) must not merge them back into one region -- this
    pins the density-erosion step, not just plain flood-fill connectivity."""
    left = [[(x, y)] for x, y in _grid_points(0, 0, 5, 5)]
    right = [[(x, y)] for x, y in _grid_points(50, 0, 55, 5)]
    # One entity whose own point sequence walks all the way from inside
    # the left cluster to inside the right cluster -- a thin bridge.
    bridge = [[(float(x), 2.5) for x in range(0, 56)]]
    entities = left + right + bridge
    clusters = cluster_entity_points(entities)
    assert len(clusters) >= 2, "the two dense clusters must remain spatially distinct"


def test_frame_polygon_spanning_multiple_regions_is_flagged():
    """A polygon whose bbox covers most of the whole sheet AND materially
    overlaps more than one independently-clustered region is sheet-border/
    frame-like -- must be flagged regardless of its absolute coordinates."""
    region_a = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    region_b = BoundingBox(min_x=40, min_y=40, max_x=50, max_y=50)
    from backend.cv_extraction.dxf_regions import DxfRegion

    regions = [
        DxfRegion(id=0, bbox=region_a, entity_count=50),
        DxfRegion(id=1, bbox=region_b, entity_count=50),
    ]
    # Polygon 0: the frame -- spans both regions and most of the sheet.
    frame_bbox = BoundingBox(min_x=-1, min_y=-1, max_x=51, max_y=51)
    # Polygon 1: a legitimate small boundary entirely within region A only.
    local_bbox = BoundingBox(min_x=1, min_y=1, max_x=9, max_y=9)
    frame_idx = detect_frame_polygon_indices([frame_bbox, local_bbox], regions)
    assert frame_idx == {0}


def test_single_region_never_flags_a_frame():
    """With only one region in total, nothing can "span multiple regions"
    by definition -- a normal single-drawing DXF's own large plot boundary
    must never be flagged just for being big relative to the sheet."""
    from backend.cv_extraction.dxf_regions import DxfRegion

    region = BoundingBox(min_x=0, min_y=0, max_x=20, max_y=20)
    regions = [DxfRegion(id=0, bbox=region, entity_count=100)]
    plot_bbox = BoundingBox(min_x=0, min_y=0, max_x=20, max_y=20)
    frame_idx = detect_frame_polygon_indices([plot_bbox], regions)
    assert frame_idx == set()


def test_build_regions_partitions_polygons_segments_and_texts_together():
    """`build_regions` must cluster polygons, open-line segments, and text
    positions AS ONE combined point cloud, so a region's extent reflects
    all of its content, not just its closed shapes."""
    # A dense grid of tiny closed quads (as if many small fragments) for
    # each of two widely-separated drawings, so each drawing is dense
    # enough for the clustering algorithm's local-density erosion step to
    # recognize it as one solid blob rather than fragmenting further.
    def _quad_grid(cx, cy, n=15, spacing=0.3, size=0.05):
        quads = []
        for i in range(n):
            for j in range(n):
                x, y = cx + i * spacing, cy + j * spacing
                quads.append([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
        return quads

    polys_a = _quad_grid(0, 0)
    polys_b = _quad_grid(40, 40)
    seg_a = [(1.0, 1.0), (1.0, 3.0)]
    text_a = [(0.5, 0.5)]
    text_b = [(41.0, 41.0)]

    regions = build_regions(
        polygon_points=polys_a + polys_b,
        segment_points=[seg_a],
        text_points=[text_a, text_b],
    )
    assert len(regions) == 2
    # Both regions should have picked up their own nearby text.
    assert all(r.text_indices for r in regions)


def test_noise_cluster_is_dropped_relative_to_dominant_cluster():
    """A single stray entity far away from an otherwise dominant, dense
    cluster must not survive as its own spurious region."""
    dominant = [[(x, y)] for x, y in _grid_points(0, 0, 10, 10, n=10)]
    stray = [[(500.0, 500.0)]]
    clusters = cluster_entity_points(dominant + stray)
    total_clustered = sum(len(c) for c in clusters)
    assert total_clustered < len(dominant) + len(stray), (
        "the far-away stray entity should be dropped as noise, not kept as its own region"
    )
