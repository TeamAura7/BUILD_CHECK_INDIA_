"""
Synthetic DXF builders for the GNN entity-graph tests
(backend/gnn_extraction/graph_builder.py) -- analogous to `pdf_builders.py`
and `geometry_builders.py`, but writing real DXF files via `ezdxf` so the
graph builder is tested against its actual input format, not a mocked one.
"""
from __future__ import annotations

from pathlib import Path


def rectangular_site_plan_dxf(
    path: Path,
    plot_width: float = 12.0,
    plot_depth: float = 18.0,
    front_setback: float = 3.0,
    rear_setback: float = 2.0,
    left_setback: float = 1.5,
    right_setback: float = 1.0,
    road_width: float = 9.0,
    include_labels: bool = True,
) -> Path:
    """
    Write a rectangular plot (origin at 0,0) with a rectangular building
    footprint inset by the given setbacks, a road polygon along the plot's
    south (y=0) edge (making south the front), and a GATE block insert on
    the front edge -- so front-side identification is possible from
    geometry alone, with or without text labels.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6  # metres
    msp = doc.modelspace()
    for layer in ("PLOT", "BLDG", "ROAD", "NOTES"):
        doc.layers.add(layer)

    msp.add_lwpolyline(
        [(0, 0), (plot_width, 0), (plot_width, plot_depth), (0, plot_depth)],
        close=True,
        dxfattribs={"layer": "PLOT"},
    )
    msp.add_lwpolyline(
        [
            (left_setback, front_setback),
            (plot_width - right_setback, front_setback),
            (plot_width - right_setback, plot_depth - rear_setback),
            (left_setback, plot_depth - rear_setback),
        ],
        close=True,
        dxfattribs={"layer": "BLDG"},
    )
    msp.add_lwpolyline(
        [
            (-2, -road_width),
            (plot_width + 2, -road_width),
            (plot_width + 2, 0),
            (-2, 0),
        ],
        close=True,
        dxfattribs={"layer": "ROAD"},
    )

    if "GATE" not in doc.blocks:
        gate = doc.blocks.new(name="GATE")
        gate.add_circle((0, 0), radius=0.3)
    msp.add_blockref("GATE", (plot_width / 2, 0), dxfattribs={"layer": "NOTES"})

    if include_labels:
        # Placed close to the geometry they describe, as real annotations
        # usually are -- this is what makes a text_near graph edge form.
        area = plot_width * plot_depth
        msp.add_text(f"PLOT AREA = {area:.2f} SQM", dxfattribs={"layer": "NOTES"}).dxf.insert = (
            plot_width / 2 - 3,
            plot_depth + 0.5,
        )
        msp.add_text(f"{road_width:.1f}M WIDE ROAD", dxfattribs={"layer": "NOTES"}).dxf.insert = (
            plot_width / 2 - 1,
            -road_width / 2,
        )

    doc.saveas(str(path))
    return path


def rotated_rectangular_site_plan_dxf(
    path: Path,
    plot_width: float = 12.0,
    plot_depth: float = 18.0,
    front_setback: float = 3.0,
    rear_setback: float = 2.0,
    left_setback: float = 1.5,
    right_setback: float = 1.0,
    angle_degrees: float = 0.0,
) -> Path:
    """The same plot+building rectangles as `rectangular_site_plan_dxf`
    (road/labels omitted -- this fixture exists only to test that measured
    width/depth are independent of the sheet's rotation), but with every
    point rotated by `angle_degrees` about the origin before being written,
    so the polygons' own edges no longer align with the DXF's X/Y axes at
    all. An axis-aligned-bbox-based width/depth measurement would report a
    larger, wrong value at most of these angles; a correct oriented
    measurement reports the same {plot_width, plot_depth} /
    {building_width, building_depth} pair at every angle.
    """
    import math

    import ezdxf

    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rot(x: float, y: float) -> tuple[float, float]:
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6  # metres
    msp = doc.modelspace()
    for layer in ("PLOT", "BLDG"):
        doc.layers.add(layer)

    plot_pts = [(0, 0), (plot_width, 0), (plot_width, plot_depth), (0, plot_depth)]
    msp.add_lwpolyline(
        [rot(x, y) for x, y in plot_pts], close=True, dxfattribs={"layer": "PLOT"},
    )

    bldg_pts = [
        (left_setback, front_setback),
        (plot_width - right_setback, front_setback),
        (plot_width - right_setback, plot_depth - rear_setback),
        (left_setback, plot_depth - rear_setback),
    ]
    msp.add_lwpolyline(
        [rot(x, y) for x, y in bldg_pts], close=True, dxfattribs={"layer": "BLDG"},
    )

    doc.saveas(str(path))
    return path


def door_gap_building_site_plan_dxf(
    path: Path,
    plot_width: float = 20.0,
    plot_depth: float = 15.0,
    front_setback: float = 3.0,
    rear_setback: float = 2.0,
    left_setback: float = 1.5,
    right_setback: float = 1.0,
    door_width: float = 1.0,
) -> Path:
    """A plot boundary (one closed polygon, as usual) with a building
    drawn as individual open wall LINE segments -- never a closed
    polygon -- with a door-sized gap left in the middle of the front
    (south) wall, plus an interior partition wall creating a T-junction
    against the north wall. Neither `dxf_reconstruction.
    reconstruct_building_polygon` (its cycle detection cannot bridge the
    gap: no graph edge spans it) nor a single closed LWPOLYLINE exists
    for the building at all -- only `dxf_wall_union.
    reconstruct_building_via_wall_union`'s area-based approach can
    recover the correct footprint here.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for layer in ("PLOT", "WALL"):
        doc.layers.add(layer)

    msp.add_lwpolyline(
        [(0, 0), (plot_width, 0), (plot_width, plot_depth), (0, plot_depth)],
        close=True, dxfattribs={"layer": "PLOT"},
    )

    bx0, by0 = left_setback, front_setback
    bx1, by1 = plot_width - right_setback, plot_depth - rear_setback
    door_x0 = bx0 + (bx1 - bx0 - door_width) / 2.0
    door_x1 = door_x0 + door_width

    def wall(x1, y1, x2, y2):
        msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": "WALL"})

    wall(bx0, by0, door_x0, by0)      # front wall, west of the door
    wall(door_x1, by0, bx1, by0)      # front wall, east of the door
    wall(bx1, by0, bx1, by1)          # east wall
    wall(bx1, by1, bx0, by1)          # north wall
    wall(bx0, by1, bx0, by0)          # west wall
    # Interior partition wall: a T-junction against the north wall,
    # dangling in the room's interior at the other end (never touching
    # anything -- must not distort the OUTER footprint).
    mid_x = (bx0 + bx1) / 2.0
    wall(mid_x, by0 + (by1 - by0) * 0.3, mid_x, by1)

    doc.saveas(str(path))
    return path


def no_text_site_plan_dxf(path: Path, **kwargs) -> Path:
    """Same as `rectangular_site_plan_dxf` but with zero text labels --
    exercises the geometry-only reasoning path (no 'FRONT SETBACK' or
    'ROAD' strings anywhere in the file)."""
    kwargs["include_labels"] = False
    return rectangular_site_plan_dxf(path, **kwargs)


def _dash_fragment_rectangle(msp, x0, y0, x1, y1, layer, dash_spacing=0.15, dash_size=0.08):
    """Trace a rectangle's perimeter as many small closed quads (dashes),
    the same pattern a real vectorized/traced dash-dot boundary produces
    -- no single closed polygon covers the whole rectangle; only the
    scattered dash fragments do.

    `dash_spacing` is an ABSOLUTE distance (not a fraction of the side
    length), so the resulting dash density is independent of the
    rectangle's own size -- important because the region-clustering
    algorithm's grid cell size is relative to the WHOLE sheet's extent, so
    a fixed dash *count* per side would silently become "too sparse to
    cluster" on a rectangle placed on a much larger sheet.
    """
    def _dashes_along(p0, p1):
        x0_, y0_ = p0
        x1_, y1_ = p1
        length = ((x1_ - x0_) ** 2 + (y1_ - y0_) ** 2) ** 0.5
        n = max(2, int(length / dash_spacing))
        for k in range(n):
            t = k / max(n - 1, 1)
            cx = x0_ + (x1_ - x0_) * t
            cy = y0_ + (y1_ - y0_) * t
            msp.add_lwpolyline(
                [
                    (cx, cy), (cx + dash_size, cy),
                    (cx + dash_size, cy + dash_size), (cx, cy + dash_size),
                ],
                close=True, dxfattribs={"layer": layer},
            )

    _dashes_along((x0, y0), (x1, y0))
    _dashes_along((x1, y0), (x1, y1))
    _dashes_along((x1, y1), (x0, y1))
    _dashes_along((x0, y1), (x0, y0))


def multi_drawing_sheet_with_frame_dxf(
    path: Path,
    plot_width: float = 20.0,
    plot_depth: float = 15.0,
    building_width: float = 12.0,
    building_depth: float = 8.0,
) -> Path:
    """
    A single sheet containing THREE things, none of them layer-tagged
    (everything on layer '0', mirroring a real vectorized-trace DXF with no
    semantic layers at all):

      1. A large sheet-border/frame rectangle enclosing (almost) the whole
         page, drawn as one ordinary closed LWPOLYLINE -- structurally
         indistinguishable from a real plot boundary by shape alone.
      2. The REAL site plan: a plot boundary traced as many small dash
         fragments (`_dash_fragment_rectangle`, mirroring a real dash-dot
         property line) with a proper closed building rectangle nested
         inside it.
      3. A second, unrelated dense cluster of small closed fragments
         elsewhere on the sheet (a stand-in for another drawing -- a floor
         plan, an elevation, whatever) -- needed so the sheet contains
         more than one spatially distinct drawing at all, which is the
         precondition for the frame polygon to be recognized as spanning
         multiple regions rather than being the sole (and therefore
         legitimate) content.

    The correct extraction outcome is: the frame is rejected, the plot
    resolves (via envelope reconstruction) close to `plot_width` x
    `plot_depth`, and the building resolves close to `building_width` x
    `building_depth` -- NOT the frame's own (much larger) extent.
    """
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()

    sheet_w, sheet_h = 90.0, 70.0
    msp.add_lwpolyline(
        [(0, 0), (sheet_w, 0), (sheet_w, sheet_h), (0, sheet_h)],
        close=True, dxfattribs={"layer": "0"},
    )

    # Drawing 1: the real site plan, placed in the lower-left area.
    plot_x0, plot_y0 = 5.0, 5.0
    plot_x1, plot_y1 = plot_x0 + plot_width, plot_y0 + plot_depth
    _dash_fragment_rectangle(msp, plot_x0, plot_y0, plot_x1, plot_y1, layer="0")
    bldg_x0 = plot_x0 + (plot_width - building_width) / 2
    bldg_y0 = plot_y0 + (plot_depth - building_depth) / 2
    msp.add_lwpolyline(
        [
            (bldg_x0, bldg_y0), (bldg_x0 + building_width, bldg_y0),
            (bldg_x0 + building_width, bldg_y0 + building_depth), (bldg_x0, bldg_y0 + building_depth),
        ],
        close=True, dxfattribs={"layer": "0"},
    )
    # A ring of dash fragments a small, constant distance outside the
    # building's own outline (e.g. standing in for setback/dimension
    # marks reaching from the boundary toward the building) -- a real
    # site plan is never JUST a bare property line with nothing else
    # drawn in the gap between it and the building. This keeps the
    # region's interior non-empty enough for density-based clustering
    # (which only ever connects entities that are actually present, never
    # entities implied by their absence) to recognize the plot boundary
    # and the building as one connected drawing.
    gap = min(plot_width - building_width, plot_depth - building_depth) / 2.0
    ring_offset = max(0.3, gap * 0.4)
    _dash_fragment_rectangle(
        msp, bldg_x0 - ring_offset, bldg_y0 - ring_offset,
        bldg_x0 + building_width + ring_offset, bldg_y0 + building_depth + ring_offset,
        layer="0", dash_spacing=0.4,
    )

    # Drawing 2: an unrelated dense cluster elsewhere on the sheet (stands
    # in for a floor plan / elevation / any other drawing on the sheet).
    _dash_fragment_rectangle(msp, 55.0, 40.0, 80.0, 62.0, layer="0")

    doc.saveas(str(path))
    return path
