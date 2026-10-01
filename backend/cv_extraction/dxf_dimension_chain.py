"""
Native DXF DIMENSION-entity verification chain for Critical Requirement 5:
"dimension evidence must verify geometry through a real chain (text ->
dimension graphic -> measured span -> geometric edge -> semantic object ->
candidate region), NOT nearest-number matching or another OCR scoring
bonus."

This module is deliberately narrow and does NOT touch the existing global
whole-sheet evidence recovery architecture (`dxf_text_recovery.py`) or its
distance/orientation/length scoring (`dimension_candidates`) -- those exist
to recover a MEASURED VALUE from free-floating TEXT/vectorized digits that
carry no inherent geometric anchor of their own, where a proximity-based
association is the best available evidence. A native DXF LINEAR or ALIGNED
DIMENSION entity is a fundamentally different, stronger kind of evidence:
it is drawn with two explicit extension-line origin points (DXF group
codes 13/14 -- `dxf.defpoint2`/`dxf.defpoint3`) that mark EXACTLY which two
real-world points it measures between. When those two points coincide
(within a small snap tolerance -- the same order of magnitude real CAD
software itself snaps to) with the two endpoints of an actual polygon
edge, that is not a guess about which edge a number is probably labelling
-- it is the dimension's own definition data confirming, byte-for-byte,
which edge it measures. That is the full chain this module verifies:

    printed/rendered text (dim.raw_text)
        -> dimension graphic (the DIMENSION entity itself)
        -> measured span (dim.value, defined by defpoint2 <-> defpoint3)
        -> geometric edge (a specific edge of a specific candidate polygon,
           by exact endpoint coincidence, not proximity)
        -> semantic object (the caller attributes the edge to plot/
           building width or depth by comparing its length against the
           already-computed oriented measurement)
        -> candidate region (whichever polygon/region owns that edge --
           the caller's responsibility, this module only matches edges)

Radius/diameter/angular/ordinate dimensions don't have a "defpoint2 to
defpoint3 span" in this sense and are simply not matched (the caller is
expected to have left `defpoint2`/`defpoint3` as None for those, see
`dxf_extractor.py`'s DIMENSION-entity collection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from backend.schemas.geometry import Polygon
from backend.spatial_reasoning import geometry_utils as geo

# How close a dimension's two extension-line points must be to a polygon
# edge's two endpoints to count as "the same span" -- real CAD software
# snaps dimension origins to actual geometry, so this only needs to absorb
# floating-point/export noise, not searched proximity. Matches the same
# order of magnitude as `dxf_reconstruction.DEFAULT_SNAP_TOLERANCE`.
DEFAULT_DIMENSION_SNAP_TOLERANCE = 0.05


class DimensionLike(Protocol):
    value: float
    raw_text: str
    defpoint2: Optional[tuple[float, float]]
    defpoint3: Optional[tuple[float, float]]


@dataclass
class VerifiedDimensionMatch:
    value: float
    raw_text: str
    edge_length: float


def find_dimension_edge_matches(
    dims: Sequence[DimensionLike],
    polygon: Polygon,
    snap_tolerance: float = DEFAULT_DIMENSION_SNAP_TOLERANCE,
) -> list[VerifiedDimensionMatch]:
    """Find every dimension whose extension-line points EXACTLY coincide
    (within `snap_tolerance`) with the two endpoints of some edge of
    `polygon`, in either order. This is a coincidence check, not a
    nearest-neighbour search: a dimension whose points sit near an edge
    but don't match either endpoint closely is correctly NOT matched --
    it is evidence for some other span (a room dimension, a setback, an
    unrelated wall), not this edge.
    """
    if not dims:
        return []
    edges = geo.polygon_edges(polygon)
    if not edges:
        return []
    matches: list[VerifiedDimensionMatch] = []
    for d in dims:
        if d.defpoint2 is None or d.defpoint3 is None:
            continue
        p2, p3 = d.defpoint2, d.defpoint3
        for edge in edges:
            forward = (
                math.hypot(p2[0] - edge.start.x, p2[1] - edge.start.y)
                + math.hypot(p3[0] - edge.end.x, p3[1] - edge.end.y)
            )
            backward = (
                math.hypot(p2[0] - edge.end.x, p2[1] - edge.end.y)
                + math.hypot(p3[0] - edge.start.x, p3[1] - edge.start.y)
            )
            if min(forward, backward) <= snap_tolerance * 2.0:
                matches.append(VerifiedDimensionMatch(value=d.value, raw_text=d.raw_text, edge_length=edge.length))
                break
    return matches


__all__ = ["DEFAULT_DIMENSION_SNAP_TOLERANCE", "VerifiedDimensionMatch", "find_dimension_edge_matches"]
