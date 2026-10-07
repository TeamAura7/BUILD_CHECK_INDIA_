"""
SiteGraph: one shared spatial model of "what's on this sheet", built purely
from geometry (polygon shapes + their spatial relationships to each other),
with layer names and text used only as secondary, confirmatory evidence --
never as the primary signal.

Why this module exists
-----------------------
Today, "which polygon is the plot / the building / the road" is answered by
several DIFFERENT, hand-written heuristics that don't share a common
representation:

  - backend/cv_extraction/dxf_extractor.py: `_pick_plot_polygon`,
    `_pick_building_polygon`, `_pick_road_polygon` (DXF, production path)
  - backend/cv_extraction/site_plan.py: independent_cv's rectangle-search
    (PDF, production path)
  - backend/spatial_reasoning/building_resolution.py, plot_resolution.py,
    road_access.py: further re-scoring on TOP of the above, working from
    generic "candidate" lists rather than a shared graph
  - backend/gnn_extraction/graph_builder.py: `_pick_building_idx`,
    `_ROAD_LAYER_RE` (a THIRD, separately-maintained copy of the same
    idea, for GNN training)

Every one of them has independently reinvented some version of "is this
nested inside that", "is this the biggest thing", "is this adjacent to
that" -- and when one gets a fix (e.g. today's plot-containment
plausibility gate, or geometric road-adjacency), the other three don't
automatically benefit unless someone remembers to port it by hand. That's
exactly how the DXF pipeline and the GNN training pipeline ended up with
two different road-detection implementations that silently drifted apart.

SiteGraph is the fix for the architecture, not just the day's specific
bugs: ONE function builds a typed graph of every polygon on the sheet plus
its geometric relationships (containment, adjacency, distance) to every
other polygon. Role assignment (which node is "the plot", "the building",
"the road") becomes a query over that graph's structure, not a bespoke
scan through a flat polygon list. Both the deterministic extractor and the
GNN's training-graph builder are meant to consume the SAME SiteGraph going
forward, so a containment/adjacency fix made once (here) benefits both,
and the GNN's input features stop being a hand-copied approximation of
what the deterministic pipeline already knows.

This module intentionally has NO dependency on PDF or DXF specifics -- it
operates on plain `Polygon` + optional layer/text metadata, so the same
graph-construction code serves both formats. Format-specific extractors
are responsible only for turning their native entities into
`SiteGraphNode`s and handing them to `build_site_graph`.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.schemas.enums import EntityKind, RelationType
from backend.schemas.evidence_graph import DrawingNode, DrawingRelationship
from backend.schemas.geometry import BoundingBox, Line, Point, Polygon
from backend.spatial_reasoning import geometry_utils as geo


class NodeRole(str, Enum):
    """What a node's role query said it plausibly is. UNKNOWN is the
    honest default -- most polygons on a real sheet (furniture, hatching,
    dimension arrows, symbols) are none of these roles at all, and staying
    UNKNOWN for them is the correct, conservative answer, not a gap to
    fill in."""

    PLOT = "plot"
    BUILDING = "building"
    ROAD = "road"
    UNKNOWN = "unknown"


@dataclass
class SiteGraphNode:
    id: int
    polygon: Polygon
    layer: Optional[str] = None
    # Optional confirmatory text found on/near this polygon (e.g. "ROAD",
    # "PLOT NO. XX") -- used only as a tie-breaker/confidence signal in
    # role queries below, never as the primary basis for a role.
    nearby_text: list[str] = field(default_factory=list)
    role: NodeRole = NodeRole.UNKNOWN

    @property
    def bbox(self) -> BoundingBox:
        return self.polygon.bounding_box

    @property
    def area(self) -> float:
        return self.polygon.area


@dataclass
class SiteGraphEdge:
    """A directed geometric relationship: `source` relative to `target`."""

    source_id: int
    target_id: int
    contains: bool  # source's bbox nests target's bbox (within tolerance)
    gap: float  # min boundary-to-boundary distance (0.0 if touching/overlapping)


@dataclass
class SiteGraph:
    nodes: list[SiteGraphNode]
    edges: list[SiteGraphEdge]

    def node(self, node_id: int) -> SiteGraphNode:
        return self.nodes[node_id]

    def contained_by(self, node_id: int) -> list[int]:
        """IDs of nodes that geometrically contain `node_id`."""
        return [e.source_id for e in self.edges if e.target_id == node_id and e.contains]

    def contains(self, node_id: int) -> list[int]:
        """IDs of nodes that `node_id` geometrically contains."""
        return [e.target_id for e in self.edges if e.source_id == node_id and e.contains]

    def nearest(self, node_id: int, among: Optional[list[int]] = None) -> Optional[tuple[int, float]]:
        """(other_id, gap) of the closest other node, optionally restricted to `among`."""
        candidates = [
            (e.target_id, e.gap)
            for e in self.edges
            if e.source_id == node_id and (among is None or e.target_id in among)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t[1])

    def find_role(self, role: NodeRole) -> list[SiteGraphNode]:
        return [n for n in self.nodes if n.role is role]


def _bbox_nested(inner: BoundingBox, outer: BoundingBox, tol: float) -> bool:
    return (
        inner.min_x >= outer.min_x - tol
        and inner.max_x <= outer.max_x + tol
        and inner.min_y >= outer.min_y - tol
        and inner.max_y <= outer.max_y + tol
    )


def _bbox_gap(a: BoundingBox, b: BoundingBox) -> float:
    """Distance between two axis-aligned bboxes (0 if they overlap/touch).

    Always a valid LOWER BOUND on the true min distance between the two
    polygons those bboxes contain, since every polygon point lies within
    its own bbox -- used below to skip the expensive exact edge-to-edge
    computation for pairs that are unambiguously far apart.
    """
    dx = max(a.min_x - b.max_x, b.min_x - a.max_x, 0.0)
    dy = max(a.min_y - b.max_y, b.min_y - a.max_y, 0.0)
    return math.hypot(dx, dy)


# A pair's exact polygon-to-edges gap is only computed when the cheap bbox
# lower bound is within this many multiples of the larger shape's own
# scale. Beyond that, the pair is unambiguously "not nearby" for every
# downstream use of `gap` (nearest-neighbor search, adjacency-ratio
# thresholding against MAX_ROAD_ADJACENCY_GAP_RATIO below) -- the bbox
# lower bound is already conclusive, so substituting it for the exact
# value cannot flip a genuinely-far pair into looking "near", and it can
# never make a pair look closer than it truly is (a lower bound is never
# an overestimate). This is what makes a many-thousand-fragment DXF (each
# dash of a dash-dot boundary as its own tiny closed polygon) tractable:
# most such fragments sit nowhere near most other polygons on the sheet.
_EXACT_GAP_SCALE_MULTIPLE = 3.0


def build_site_graph(nodes: list[SiteGraphNode], containment_tol_ratio: float = 0.02) -> SiteGraph:
    """
    Compute every pairwise containment + adjacency-distance relationship
    once. O(n^2) in polygon count -- fine for a real sheet's few hundred
    closed polygons; callers with very large entity counts (thousands of
    furniture/hatch fragments) should pre-filter to closed, reasonably
    sized polygons before calling this (see role-query functions below for
    the filtering this module itself applies for role assignment).

    The expensive part of that O(n^2) is the exact polygon-to-edges
    distance (O(v1*v2) per pair) computed for every non-contained pair;
    `_bbox_gap`'s cheap lower bound short-circuits that for pairs that are
    obviously far apart (see `_EXACT_GAP_SCALE_MULTIPLE`).
    """
    edges: list[SiteGraphEdge] = []
    for a in nodes:
        a_bbox = a.bbox
        a_scale = max(a_bbox.width, a_bbox.height, 1e-9)
        a_edges = geo.polygon_edges(a.polygon)
        for b in nodes:
            if a.id == b.id:
                continue
            b_bbox = b.bbox
            tol = containment_tol_ratio * a_scale
            contains = _bbox_nested(b_bbox, a_bbox, tol)
            if contains:
                gap = 0.0
            else:
                bbox_gap = _bbox_gap(a_bbox, b_bbox)
                b_scale = max(b_bbox.width, b_bbox.height, 1e-9)
                if bbox_gap > max(a_scale, b_scale) * _EXACT_GAP_SCALE_MULTIPLE:
                    gap = bbox_gap
                else:
                    gap = geo.polygon_to_edges_distance(b.polygon, a_edges)
            edges.append(SiteGraphEdge(source_id=a.id, target_id=b.id, contains=contains, gap=gap))
    return SiteGraph(nodes=nodes, edges=edges)


# --------------------------------------------------------------------------
# Role queries. Each is a pure function of graph STRUCTURE (containment
# counts, adjacency, shape) -- layer-name/text hints are accepted as an
# optional look-aside for trusted labels, but every fallback path is
# structural, so the same logic works on a sheet with no useful layer
# names or text at all (the DXF case this whole file exists to improve on).
# --------------------------------------------------------------------------

MIN_PLOT_CONTAINMENT_FRACTION = 0.3
MAX_BUILDING_ASPECT_RATIO = 12.0
MAX_ROAD_ADJACENCY_GAP_RATIO = 0.12


def infer_plot_node(
    graph: SiteGraph, layer_hint: Optional[list[SiteGraphNode]] = None
) -> Optional[SiteGraphNode]:
    """
    The plot is structurally the site's outer boundary: whichever node
    contains the largest FRACTION of everything else, provided that
    fraction clears a real bar (see module docstring's E-B-ELEV symbol
    example -- a random large shape containing nothing is not a plot no
    matter how big its raw area is).
    """
    if layer_hint:
        return max(layer_hint, key=lambda n: n.area)
    if not graph.nodes:
        return None
    best: Optional[SiteGraphNode] = None
    best_fraction = 0.0
    for n in sorted(graph.nodes, key=lambda n: n.area, reverse=True):
        others = [m for m in graph.nodes if m.id != n.id]
        if not others:
            continue
        contained = len(graph.contains(n.id))
        fraction = contained / len(others)
        if fraction >= MIN_PLOT_CONTAINMENT_FRACTION and fraction > best_fraction:
            best, best_fraction = n, fraction
    return best


def infer_building_node(
    graph: SiteGraph, plot_node: SiteGraphNode, layer_hint: Optional[list[SiteGraphNode]] = None
) -> Optional[SiteGraphNode]:
    """
    The building is the largest node NESTED INSIDE the plot whose shape is
    plausible (not a 60:1 sliver -- see graph_builder._pick_building_idx's
    verified real-corpus failure mode this mirrors). Refuses (returns
    None) rather than guessing when nothing nested clears the shape bar,
    same philosophy as that function.
    """
    nested_ids = graph.contains(plot_node.id)
    candidates = [graph.node(i) for i in nested_ids]
    if layer_hint:
        tagged = [n for n in candidates if n in layer_hint]
        if tagged:
            candidates = tagged
    plausible = [n for n in candidates if geo.aspect_ratio(n.bbox) <= MAX_BUILDING_ASPECT_RATIO]
    if not plausible:
        return None
    return max(plausible, key=lambda n: n.area)


def infer_road_node(
    graph: SiteGraph, plot_node: SiteGraphNode, layer_hint: Optional[list[SiteGraphNode]] = None
) -> Optional[SiteGraphNode]:
    """
    The road is a node NOT nested inside the plot, adjacent to its
    boundary, and shaped like a strip -- see
    road_access.infer_road_polygon_by_adjacency, which this delegates to
    for the actual scoring (kept as one implementation rather than
    duplicating it here).
    """
    if layer_hint:
        return max(layer_hint, key=lambda n: n.area)
    from backend.spatial_reasoning.road_access import infer_road_polygon_by_adjacency

    others = [n.polygon for n in graph.nodes if n.id != plot_node.id]
    winner_polygon = infer_road_polygon_by_adjacency(plot_node.polygon, others)
    if winner_polygon is None:
        return None
    return next((n for n in graph.nodes if n.polygon is winner_polygon), None)


_NODE_ROLE_TO_ENTITY_KIND = {
    NodeRole.PLOT: EntityKind.PLOT,
    NodeRole.BUILDING: EntityKind.BUILDING,
    NodeRole.ROAD: EntityKind.ROAD,
    NodeRole.UNKNOWN: None,
}

# Below this absolute perpendicular-offset-to-scale ratio, two near-parallel
# characteristic edges are considered to lie on the SAME infinite line
# (COLLINEAR) rather than merely PARALLEL.
_COLLINEAR_OFFSET_RATIO = 0.02

# Below this endpoint-gap-to-scale ratio, two collinear characteristic edges
# are considered to CONTINUE one another (e.g. two dashes of the same
# dash-dot boundary) rather than merely being COLLINEAR but unrelated (two
# parallel walls on opposite sides of a corridor, say).
_CONTINUES_GAP_RATIO = 0.05

# alignment (1.0 = parallel, 0.0 = perpendicular, from geo.orientation_alignment)
# thresholds for classifying two characteristic edges' relative orientation.
_PARALLEL_ALIGNMENT_THRESHOLD = 0.95
_PERPENDICULAR_ALIGNMENT_THRESHOLD = 0.05


def _point_to_infinite_line_distance(point: Point, edge: Line) -> float:
    """Perpendicular distance from `point` to the INFINITE line through `edge`
    (unlike `geo.point_segment_distance`, which clamps to the finite segment)."""
    dx, dy = edge.end.x - edge.start.x, edge.end.y - edge.start.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return math.hypot(point.x - edge.start.x, point.y - edge.start.y)
    px, py = point.x - edge.start.x, point.y - edge.start.y
    cross = dx * py - dy * px
    return abs(cross) / length


def _characteristic_edge(polygon: Polygon) -> Line:
    """The polygon's own longest ring edge -- a cheap, deterministic proxy
    for "this shape's dominant orientation", used only to relate two nodes
    that are already known to be NEAR/ADJACENT/ENCLOSING (see
    `to_drawing_graph`), not as a full pairwise all-edges comparison. A
    genuinely finer-grained (all-edges) version of PARALLEL/COLLINEAR/
    CONTINUES is future work, not attempted here -- see module docstring."""
    return max(geo.polygon_edges(polygon), key=lambda e: e.length)


def _orientation_relation(
    edge_a: Line, edge_b: Line,
) -> Optional[tuple[RelationType, float, str]]:
    """Deterministic geometric relation between two characteristic edges, or
    None if neither near-parallel nor near-perpendicular (an oblique pair is
    not asserted as any of these relations rather than forcing the nearest
    label). Returns (relation, confidence, note)."""
    scale = max(edge_a.length, edge_b.length, 1e-9)
    if geo.segment_segment_distance(edge_a, edge_b) <= 1e-6 * scale:
        return (RelationType.INTERSECTS, 1.0, "characteristic edges intersect")

    orient_a = geo.line_orientation_degrees(edge_a)
    orient_b = geo.line_orientation_degrees(edge_b)
    alignment = geo.orientation_alignment(orient_a, orient_b)

    if alignment >= _PARALLEL_ALIGNMENT_THRESHOLD:
        offset = _point_to_infinite_line_distance(geo.edge_midpoint(edge_b), edge_a)
        if offset <= _COLLINEAR_OFFSET_RATIO * scale:
            endpoint_gap = min(
                geo.point_segment_distance(edge_a.start, edge_b),
                geo.point_segment_distance(edge_a.end, edge_b),
                geo.point_segment_distance(edge_b.start, edge_a),
                geo.point_segment_distance(edge_b.end, edge_a),
            )
            if endpoint_gap <= _CONTINUES_GAP_RATIO * scale:
                return (RelationType.CONTINUES, alignment, f"collinear with nearby endpoints (gap={endpoint_gap:.4f})")
            return (RelationType.COLLINEAR, alignment, f"collinear, offset={offset:.4f}")
        return (RelationType.PARALLEL, alignment, f"parallel, perpendicular offset={offset:.4f}")

    if alignment <= _PERPENDICULAR_ALIGNMENT_THRESHOLD:
        return (RelationType.PERPENDICULAR, 1.0 - alignment, "near-perpendicular characteristic edges")

    return None


def to_drawing_graph(
    graph: SiteGraph, *, near_gap_ratio: float = 0.15, adjacent_gap_ratio: float = 0.01,
) -> tuple[list[DrawingNode], list[DrawingRelationship]]:
    """
    Project a `SiteGraph` into the generalized Drawing Evidence Graph shape
    (`backend.schemas.evidence_graph.DrawingNode`/`DrawingRelationship`) --
    see ARCHITECTURE_V2.md, Deliverable C.2 item 2.

    This is purely additive: `SiteGraph`/`build_site_graph`/`infer_*_node`
    are unchanged, and no existing caller of them needs to change. Only
    DETERMINISTIC geometric relations are computed here (no scoring, no
    learned component) -- MEASURES/BELONGS_TO/CONFLICTS_WITH (association
    relations produced by scoring logic, or a future GNN relationship-
    inference pass) are out of scope for this function; it only ever emits
    `derivation="deterministic_geometry"` rows.

    ENCLOSES/INSIDE generalize the existing `SiteGraphEdge.contains` in both
    directions. ADJACENT/NEAR generalize `SiteGraphEdge.gap`, bounded to
    pairs within `near_gap_ratio` of the larger node's own scale -- this
    keeps relationship count proportional to what `build_site_graph` already
    computes (a many-thousand-fragment DXF sheet's fragments are only near a
    handful of neighbours each, not every other fragment on the sheet), not
    a new O(n^2) blowup. PARALLEL/PERPENDICULAR/COLLINEAR/CONTINUES/
    INTERSECTS are computed only between the two nodes' own longest ("
    characteristic") edges, and only for pairs already found to be
    ENCLOSES/ADJACENT/NEAR -- a deliberate, documented scoping choice (see
    `_characteristic_edge`'s docstring), not a full pairwise all-edges
    comparison. ALIGNED_WITH is defined in `RelationType` but not computed
    by this function -- it is reserved for a future association-scoring or
    GNN-inferred pass, since "two entities share an axis" (as opposed to
    "two edges are parallel", already covered by PARALLEL) has no
    established deterministic definition here yet.
    """
    nodes = [
        DrawingNode(
            id=str(n.id), node_kind="geometry", geometry_ref=n.polygon,
            role_hint=_NODE_ROLE_TO_ENTITY_KIND.get(n.role),
        )
        for n in graph.nodes
    ]

    edge_lookup = {(e.source_id, e.target_id): e for e in graph.edges}
    relationships: list[DrawingRelationship] = []
    node_by_id = {n.id: n for n in graph.nodes}
    rel_counter = 0

    def _next_rel_id() -> str:
        nonlocal rel_counter
        rel_counter += 1
        return f"rel-{rel_counter}"

    for i, j in itertools.combinations((n.id for n in graph.nodes), 2):
        e_ij, e_ji = edge_lookup.get((i, j)), edge_lookup.get((j, i))
        if e_ij is None or e_ji is None:
            continue

        if e_ij.contains or e_ji.contains:
            outer, inner = (i, j) if e_ij.contains else (j, i)
            relationships.append(
                DrawingRelationship(
                    id=_next_rel_id(), relation=RelationType.ENCLOSES,
                    source_node_id=str(outer), target_node_id=str(inner),
                    confidence=1.0, derivation="deterministic_geometry",
                    note="bbox containment (within tolerance)",
                )
            )
            relationships.append(
                DrawingRelationship(
                    id=_next_rel_id(), relation=RelationType.INSIDE,
                    source_node_id=str(inner), target_node_id=str(outer),
                    confidence=1.0, derivation="deterministic_geometry",
                    note="bbox containment (within tolerance)",
                )
            )
            continue

        node_i, node_j = node_by_id[i], node_by_id[j]
        scale = max(node_i.bbox.width, node_i.bbox.height, node_j.bbox.width, node_j.bbox.height, 1e-9)
        gap = e_ij.gap
        if gap <= adjacent_gap_ratio * scale:
            relation = RelationType.ADJACENT
        elif gap <= near_gap_ratio * scale:
            relation = RelationType.NEAR
        else:
            continue

        relationships.append(
            DrawingRelationship(
                id=_next_rel_id(), relation=relation, source_node_id=str(i), target_node_id=str(j),
                confidence=1.0, derivation="deterministic_geometry", note=f"boundary-to-boundary gap={gap:.4f}",
            )
        )

        try:
            edge_a, edge_b = _characteristic_edge(node_i.polygon), _characteristic_edge(node_j.polygon)
            orientation = _orientation_relation(edge_a, edge_b)
        except Exception:
            orientation = None
        if orientation is not None:
            rel_type, confidence, note = orientation
            relationships.append(
                DrawingRelationship(
                    id=_next_rel_id(), relation=rel_type, source_node_id=str(i), target_node_id=str(j),
                    confidence=confidence, derivation="deterministic_geometry", note=note,
                )
            )

    return nodes, relationships


def assign_roles(
    graph: SiteGraph,
    plot_layer_hint: Optional[list[SiteGraphNode]] = None,
    building_layer_hint: Optional[list[SiteGraphNode]] = None,
    road_layer_hint: Optional[list[SiteGraphNode]] = None,
) -> SiteGraph:
    """Convenience: run all three role queries and set `.role` on the winning nodes in place."""
    plot_node = infer_plot_node(graph, plot_layer_hint)
    if plot_node is None:
        return graph
    plot_node.role = NodeRole.PLOT

    building_node = infer_building_node(graph, plot_node, building_layer_hint)
    if building_node is not None:
        building_node.role = NodeRole.BUILDING

    road_node = infer_road_node(graph, plot_node, road_layer_hint)
    if road_node is not None:
        road_node.role = NodeRole.ROAD

    return graph
