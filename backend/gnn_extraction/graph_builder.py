"""
Entity-graph construction for DWG/DXF site plans (see
docs/dwg_gnn_compliance_plan.md for the full pipeline this feeds into).

Every DXF entity that matters for site-plan reasoning becomes a graph node:
whole polygons (plot/building/other), the four sides of the plot's bounding
box as their own nodes (since setback reasoning needs per-side classification,
not just per-polygon), text labels, and block inserts (doors/gates/entrances).
Edges connect nodes that are spatially or logically related -- "this text sits
next to this edge", "this edge's nearest gap is to this building polygon",
"this block sits on this edge" -- so a graph neural network trained on this
representation can reason about the whole site layout at once (message
passing), not classify one entity in isolation.

This module reuses backend.cv_extraction.dxf_extractor's DXF-walking,
unit-resolution, and polygon-building helpers, so the graph a future GNN sees
and the geometry the deterministic extractor sees are always built from the
same primitives -- there is exactly one way this codebase reads a DXF file.

No dependency on torch/PyTorch Geometric at import time: `build_entity_graph`
returns a plain, framework-agnostic `EntityGraph`. Call `.to_pyg_data()` only
when you actually need a training/inference tensor -- it imports torch/PyG
lazily, so this module (and any labeling/inspection tooling built on it) works
without a training stack installed.
"""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.cv_extraction.dxf_extractor import (
    _BUILDING_LAYER_RE,
    _ROAD_LAYER_RE,
    _RawPoly,
    _RawText,
    _iter_entities_flat,
    _make_polygon,
    _resolve_unit_factor,
)
from backend.schemas.enums import SourceType
from backend.schemas.evidence import TextEvidence
from backend.schemas.geometry import BoundingBox, Point as GeoPoint
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.front_side import resolve_front_side
from backend.spatial_reasoning.road_access import collect_access_evidence, infer_road_polygon_by_adjacency

# ---------------------------------------------------------------------------
# Node / edge / graph containers
# ---------------------------------------------------------------------------

# `node_type` is the fine-grained INTERNAL classification used to build the
# graph's structure (which polygon plays which role when computing gaps,
# etc). It is deliberately NOT what the model is trained to predict, and
# NOT what goes into the input feature vector for polygon-kind nodes --
# doing so would let the model (and the training metric) just copy back the
# same layer-name regex heuristic it's meant to learn to improve on.
NODE_TYPES = ["plot_polygon", "building_candidate", "road_polygon", "other_polygon", "plot_edge", "text", "block"]
EDGE_TYPES = ["belongs_to", "nearest_gap", "text_near", "block_near", "contains", "knn"]

# A real building footprint, even an elongated or L-shaped one, doesn't
# look like a 60:1 sliver -- that shape is a wall, fence, or boundary line
# drawn with a nominal thickness (a common CAD convention), which can still
# have a larger raw AREA than the real building. Verified against a real
# corpus file where a 1.08m x 66.00m sliver got picked as "the building"
# purely on area, ~300 units from the actual plot.
_MAX_BUILDING_ASPECT_RATIO = 8.0

# `entity_kind` is the STRUCTURAL type -- what kind of DXF primitive this
# is (a closed polygon vs. a plot side vs. a text run vs. a block insert).
# This is legitimate input: it's parsed directly from the file, not guessed.
ENTITY_KINDS = ["polygon", "plot_edge", "text", "block"]

_NODE_TYPE_TO_ENTITY_KIND = {
    "plot_polygon": "polygon",
    "building_candidate": "polygon",
    "road_polygon": "polygon",
    "other_polygon": "polygon",
    "plot_edge": "plot_edge",
    "text": "text",
    "block": "block",
}

# The training target vocabulary: polygon roles and plot-edge sides share one
# classification head, since a node is only ever one structural kind, so the
# classes never actually compete with each other for a given node.
# "unknown"/None nodes (text, block, or an edge the heuristic couldn't
# resolve) are excluded from the training loss via a mask, not given a class.
LABEL_CLASSES = ["plot", "building", "road", "other", "front", "rear", "left", "right"]

# Fixed, ordered feature schema -- every node gets a vector of this length,
# with 0.0 for features that don't apply to its node_type. Append new
# features to the end in future versions; never insert or reorder, or a
# previously-trained model's weights will silently misalign with the columns.
FEATURE_NAMES = [
    "area_ratio_to_plot",
    "aspect_ratio",
    "vertex_count",
    "centroid_offset_norm",
    "edge_length_ratio",
    "edge_gap_norm",
    "edge_gap_rank",
    "dist_to_road_norm",
    "dist_to_block_norm",
    "text_length_norm",
    "mentions_road",
    "mentions_setback",
    "mentions_area",
    "has_number",
]

_ENTRANCE_BLOCK_RE = re.compile(r"GATE|ENTR|DOOR|DR[-_]?\d", re.I)
_ROAD_WORD_RE = re.compile(r"\bROAD\b", re.I)
_SETBACK_WORD_RE = re.compile(r"SETBACK", re.I)
_AREA_WORD_RE = re.compile(r"AREA", re.I)
_NUMBER_RE = re.compile(r"\d")


def _point_bbox(pos: tuple[float, float], eps: float = 0.01) -> BoundingBox:
    x, y = pos
    return BoundingBox(min_x=x - eps, min_y=y - eps, max_x=x + eps, max_y=y + eps)


@dataclass
class GraphNode:
    id: int
    node_type: str
    layer: str
    position: tuple[float, float]
    features: dict[str, float] = field(default_factory=dict)
    raw_text: Optional[str] = None
    # Training target, filled in by silver-labeling (from a heuristic --
    # layer-name regex for polygons, resolve_front_side for plot edges) and
    # correctable by a human reviewer. Left as None where the heuristic
    # itself has no answer (e.g. no building found, or front-side
    # unresolvable) -- those nodes are excluded from the training loss, not
    # guessed. Always None for text/block nodes in this version.
    label: Optional[str] = None
    # Real-world values (m, m2) for whichever node this is -- NEVER fed to
    # the model (normalized `features` above is what the model sees). This
    # exists purely so a prediction can be traced back to an actual number:
    # "node 7 is predicted 'plot'" is only useful once you can also ask
    # "and what's node 7's actual area/width/depth". Populated for polygon
    # and plot_edge nodes; empty for text/block.
    raw_measurements: dict[str, float] = field(default_factory=dict)

    @property
    def entity_kind(self) -> str:
        return _NODE_TYPE_TO_ENTITY_KIND[self.node_type]

    @property
    def label_id(self) -> int:
        """-1 (ignore-index convention) if unlabeled."""
        if self.label is None:
            return -1
        return LABEL_CLASSES.index(self.label)

    def feature_vector(self) -> list[float]:
        kind_onehot = [1.0 if self.entity_kind == k else 0.0 for k in ENTITY_KINDS]
        named = [float(self.features.get(name, 0.0)) for name in FEATURE_NAMES]
        return kind_onehot + named


@dataclass
class GraphEdge:
    source: int
    target: int
    edge_type: str


@dataclass
class EntityGraph:
    document_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    unit_reason: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.node_type] = counts.get(n.node_type, 0) + 1
        return counts

    def to_pyg_data(self):
        """Convert to a torch_geometric.data.Data object. Imports torch/PyG
        lazily -- only needed once you actually train or run inference."""
        import torch
        from torch_geometric.data import Data

        x = torch.tensor([n.feature_vector() for n in self.nodes], dtype=torch.float)
        if self.edges:
            edge_index = torch.tensor(
                [[e.source for e in self.edges], [e.target for e in self.edges]], dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        y = torch.tensor([n.label_id for n in self.nodes], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = y >= 0  # nodes with no silver/human label are excluded from loss, not guessed
        data.document_id = self.document_id
        data.node_types = [n.node_type for n in self.nodes]
        return data


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_dist(pt: tuple[float, float], points: list[tuple[float, float]]) -> Optional[float]:
    if not points:
        return None
    return min(_dist(pt, p) for p in points)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _bbox_nested(inner, outer, tol: float = 0.5) -> bool:
    return (
        inner.min_x >= outer.min_x - tol
        and inner.max_x <= outer.max_x + tol
        and inner.min_y >= outer.min_y - tol
        and inner.max_y <= outer.max_y + tol
    )


def _pick_building_idx(polygons: list[_RawPoly], plot_idx: int) -> Optional[int]:
    """
    Pick the SINGLE best building-footprint candidate, mirroring
    dxf_extractor.py's `_pick_building_polygon` -- NOT "every polygon
    smaller than the plot", which is true of almost any decorative or
    detail polygon on a real site plan (north arrow, hatch fragment,
    parking bay marking, compound-wall corner detail) and was silently
    labeling dozens of unrelated polygons "building" per file on real
    corpora (verified: ~60 per file on a real 1,022-file training run,
    vs. the ~1 per file a real site plan actually has).

    Two further filters, both verified against a real corpus file where
    the previous version picked a 1.08m x 66.00m sliver (61:1 aspect
    ratio) sitting ~300 units from the plot as "the building", which then
    corrupted every setback distance on that file too (gaps are measured
    to whatever this function returns -- see build_entity_graph below):

    1. Aspect ratio: a real building footprint isn't a 60:1 sliver -- real
       CAD files commonly draw a wall, fence, or boundary strip as a thin
       rectangle for line weight, and those can have a larger raw AREA
       than the actual building despite being nothing like one shape-wise.
       "Largest remaining area" alone doesn't filter these out.
    2. Fallback discipline: if nothing is nested inside the plot bbox AND
       nothing matches a building-ish layer name, that's a real "no
       confident candidate" situation, not a reason to fall back to
       "biggest of whatever's left regardless of where it is". Returning
       None here (which correctly propagates to no building_candidate
       node, and therefore no fabricated setback gaps either) is the
       consistent behavior with how every other MISSING field in this
       codebase is handled -- prefer no answer over a confidently wrong
       one.
    """
    plot_poly = polygons[plot_idx].polygon
    plot_bbox = plot_poly.bounding_box
    plot_area = plot_poly.area
    candidates = [i for i in range(len(polygons)) if i != plot_idx and polygons[i].polygon.area < plot_area * 0.98]
    if not candidates:
        return None

    def _aspect_ok(i: int) -> bool:
        bbox = polygons[i].polygon.bounding_box
        w, h = bbox.width, bbox.height
        if w <= 0 or h <= 0:
            return False
        return max(w, h) / min(w, h) <= _MAX_BUILDING_ASPECT_RATIO

    candidates = [i for i in candidates if _aspect_ok(i)]
    if not candidates:
        return None

    layer_matches = [i for i in candidates if _BUILDING_LAYER_RE.search(polygons[i].layer or "")]
    pool = layer_matches if layer_matches else candidates
    nested = [i for i in pool if _bbox_nested(polygons[i].polygon.bounding_box, plot_bbox)]

    if nested:
        pool2 = nested
    elif layer_matches:
        # Not nested, but explicitly layer-tagged as a building -- trust
        # the tag over position (a real building footprint can legitimately
        # poke past a loosely-drawn or partial plot boundary).
        pool2 = layer_matches
    else:
        # Neither nested inside the plot NOR layer-tagged as a building --
        # too weak a signal to confidently call anything "the building".
        return None

    return max(pool2, key=lambda i: polygons[i].polygon.area)


def build_entity_graph(document_path: Path, document_id: str, k_nearest: int = 4) -> EntityGraph:
    """
    Parse a DXF file (a DWG converted to DXF via ODA File Converter works
    identically -- both open through the same `ezdxf.readfile` call) and
    build its entity graph.
    """
    import ezdxf
    from ezdxf import recover as ezdxf_recover

    warnings: list[str] = []
    try:
        doc = ezdxf.readfile(str(document_path))
    except ezdxf.DXFStructureError:
        doc, auditor = ezdxf_recover.readfile(str(document_path))
        warnings.append("Recovered a structurally-damaged DXF; some entities may be missing.")
        if auditor.has_errors:
            warnings.append(f"{len(auditor.errors)} unresolved issue(s) remained after recovery.")

    msp = doc.modelspace()

    raw_polygons: list[_RawPoly] = []
    raw_texts: list[_RawText] = []
    raw_blocks: list[tuple[str, tuple[float, float]]] = []

    for e in _iter_entities_flat(msp, warnings):
        try:
            dxftype = e.dxftype()
        except Exception:
            continue
        try:
            layer = e.dxf.layer
        except Exception:
            layer = "0"
        try:
            if dxftype == "LWPOLYLINE" and e.closed:
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
                poly = _make_polygon(pts)
                if poly is not None:
                    raw_polygons.append(_RawPoly(layer=layer, polygon=poly))
            elif dxftype == "POLYLINE" and e.is_2d_polyline and e.is_closed:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                poly = _make_polygon(pts)
                if poly is not None:
                    raw_polygons.append(_RawPoly(layer=layer, polygon=poly))
            elif dxftype in ("TEXT", "ATTRIB", "ATTDEF"):
                txt = (e.dxf.text or "").strip()
                if txt:
                    pos = e.dxf.insert
                    raw_texts.append(_RawText(text=txt, position=(pos.x, pos.y), layer=layer))
            elif dxftype == "MTEXT":
                try:
                    txt = e.plain_text().strip()
                except Exception:
                    txt = (getattr(e, "text", "") or "").strip()
                if txt:
                    pos = e.dxf.insert
                    raw_texts.append(_RawText(text=txt, position=(pos.x, pos.y), layer=layer))
            elif dxftype == "INSERT":
                name = getattr(e.dxf, "name", "") or ""
                pos = e.dxf.insert
                raw_blocks.append((name, (pos.x, pos.y)))
        except Exception as exc:  # pragma: no cover - defensive against malformed entities
            warnings.append(f"Skipped an unparsable {dxftype} entity: {exc}")

    largest_raw_area = max((p.polygon.area for p in raw_polygons), default=None)
    insunits = None
    try:
        insunits = doc.header.get("$INSUNITS")
    except Exception:
        pass
    factor, unit_reason, _unit_confidence = _resolve_unit_factor(insunits, largest_raw_area)

    polygons = [p.scaled(factor) for p in raw_polygons]
    texts = [t.scaled(factor) for t in raw_texts]
    blocks = [(name, (x * factor, y * factor)) for name, (x, y) in raw_blocks]

    if not polygons:
        warnings.append("No closed polygons found in this file -- graph contains text/block nodes only.")
        return EntityGraph(document_id=document_id, nodes=[], edges=[], unit_reason=unit_reason, warnings=warnings)

    plot_idx = max(range(len(polygons)), key=lambda i: polygons[i].polygon.area)
    plot_bbox = polygons[plot_idx].polygon.bounding_box
    plot_area = polygons[plot_idx].polygon.area
    plot_diag = math.hypot(plot_bbox.width, plot_bbox.height) or 1.0
    plot_quarter_perimeter = (plot_bbox.width + plot_bbox.height) / 2.0 or 1.0
    plot_centroid = ((plot_bbox.min_x + plot_bbox.max_x) / 2.0, (plot_bbox.min_y + plot_bbox.max_y) / 2.0)

    entrance_points = [pos for name, pos in blocks if _ENTRANCE_BLOCK_RE.search(name)]
    road_segments = [seg for p in polygons if _ROAD_LAYER_RE.search(p.layer or "") for seg in geo.polygon_edges(p.polygon)]

    def _nearest_seg_dist(pt: tuple[float, float], segments: list) -> Optional[float]:
        # Perpendicular distance to the nearest polygon EDGE, not nearest
        # vertex -- for axis-aligned rectangles a vertex-only distance
        # overstates the gap on the side that's actually closest (it can
        # only "see" corners, not the straight edge between them).
        if not segments:
            return None
        p = GeoPoint(x=pt[0], y=pt[1])
        return min(geo.point_segment_distance(p, seg) for seg in segments)

    nodes: list[GraphNode] = []
    node_positions: list[tuple[float, float]] = []
    poly_node_id: dict[int, int] = {}
    building_idx = _pick_building_idx(polygons, plot_idx)

    # Geometry-first road detection (see infer_road_polygon_by_adjacency):
    # which polygon actually sits adjacent to the plot boundary and is
    # shaped like a road strip, rather than only trusting whichever layer
    # name a particular drafter happened to use. Checked BEFORE the
    # per-polygon loop below so the loop can prefer this over the
    # layer-name regex, not just fall back to it.
    plot_polygon_obj = polygons[plot_idx].polygon
    geom_road_polygon = infer_road_polygon_by_adjacency(
        plot_polygon_obj, [p.polygon for i, p in enumerate(polygons) if i != plot_idx]
    )
    road_idx_by_geometry = next(
        (i for i, p in enumerate(polygons) if p.polygon is geom_road_polygon), None
    ) if geom_road_polygon is not None else None

    # --- polygon nodes ---
    for i, rp in enumerate(polygons):
        poly = rp.polygon
        bbox = poly.bounding_box
        centroid = ((bbox.min_x + bbox.max_x) / 2.0, (bbox.min_y + bbox.max_y) / 2.0)
        centroid_offset = _dist(centroid, plot_centroid) / plot_diag

        if i == plot_idx:
            node_type = "plot_polygon"
        elif i == road_idx_by_geometry or _ROAD_LAYER_RE.search(rp.layer or ""):
            node_type = "road_polygon"
        elif i == building_idx:
            node_type = "building_candidate"
        else:
            node_type = "other_polygon"
        # Silver label = training target. Deliberately the SAME layer-name
        # heuristic used to build the graph's internal structure above --
        # this is the free bootstrap label described in the build plan, not
        # a separate ground truth. It gets corrected by human review before
        # being trusted, and the model is expected to eventually disagree
        # with (and improve on) this heuristic on files where it's wrong.
        role_label = {"plot_polygon": "plot", "road_polygon": "road", "building_candidate": "building"}.get(node_type, "other")

        aspect = (bbox.width / bbox.height) if bbox.height else 0.0
        node = GraphNode(
            id=len(nodes),
            node_type=node_type,
            layer=rp.layer,
            position=centroid,
            label=role_label,
            features={
                "area_ratio_to_plot": (poly.area / plot_area) if plot_area else 0.0,
                "aspect_ratio": aspect,
                "vertex_count": float(len(poly.points)),
                "centroid_offset_norm": centroid_offset,
            },
            raw_measurements={"area_m2": poly.area, "width_m": bbox.width, "depth_m": bbox.height},
        )
        poly_node_id[i] = node.id
        nodes.append(node)
        node_positions.append(centroid)

    building_polys = [polygons[i] for i in range(len(polygons)) if nodes[poly_node_id[i]].node_type == "building_candidate"]
    building_segments = [seg for bp in building_polys for seg in geo.polygon_edges(bp.polygon)]
    road_polys_for_labeling = [polygons[i] for i in range(len(polygons)) if nodes[poly_node_id[i]].node_type == "road_polygon"]

    # The four sides of the plot's bounding box -- a rectangular-plot
    # simplification. The natural next step once this is validated
    # end-to-end is one node per actual polygon edge, for non-rectangular
    # plots.
    b = plot_bbox
    sides = {
        "N": ((b.min_x, b.max_y), (b.max_x, b.max_y)),
        "S": ((b.min_x, b.min_y), (b.max_x, b.min_y)),
        "E": ((b.max_x, b.min_y), (b.max_x, b.max_y)),
        "W": ((b.min_x, b.min_y), (b.min_x, b.max_y)),
    }

    # --- silver labels for plot-edge front/rear/left/right, via the SAME
    # resolve_front_side heuristic the deterministic extractor uses (road
    # proximity, then access/entrance text, then a low-confidence geometric
    # fallback). Matched back to our four bbox sides by nearest midpoint --
    # exact for the common rectangular-plot case; a non-rectangular plot may
    # leave some sides unmatched (label stays None), which is the correct,
    # conservative outcome rather than a guess.
    plot_poly_obj = polygons[plot_idx].polygon
    road_bbox_for_labeling = max((p.polygon.bounding_box for p in road_polys_for_labeling), key=lambda bb: bb.width * bb.height, default=None)
    access_evidence = collect_access_evidence(
        [TextEvidence(source_type=SourceType.TEXT, raw_text=t.text, page=0, bounding_box=_point_bbox(t.position)) for t in texts],
        page=0,
    )
    fsr = resolve_front_side(plot_poly_obj, road_bbox_for_labeling, access_evidence)
    side_label: dict[str, str] = {}

    def _side_key_for_point(pt: tuple[float, float]) -> Optional[str]:
        best_side, best_d = None, None
        for s, (sp1, sp2) in sides.items():
            smid = ((sp1[0] + sp2[0]) / 2.0, (sp1[1] + sp2[1]) / 2.0)
            d = _dist(pt, smid)
            if best_d is None or d < best_d:
                best_side, best_d = s, d
        return best_side if best_d is not None and best_d < 0.5 else None

    for role, role_edges in (("front", fsr.front_edges), ("rear", fsr.rear_edges), ("left", fsr.left_edges), ("right", fsr.right_edges)):
        # Only ever turn resolve_front_side's output into a silver label when
        # it came from real evidence (a FRONT label, ROAD candidate, or
        # STREET/ACCESS/MAIN-ENTRY/GATE text -- see _PRIORITY_TO_CONFIDENCE
        # in front_side.py). Its LOW-confidence fallback (no evidence found
        # at all -> arbitrary first-edge-in-ring-order) is a real, useful
        # value for the deterministic extractor's UI, where it's clearly
        # flagged LOW to a human -- but it is NOT a label: an arbitrary edge
        # tied to vertex-storage order carries no real front/rear/left/right
        # signal, and training on it teaches the model a false-but-consistent
        # pattern instead of nothing. Skipping it here means those plot_edge
        # nodes keep label=None and are excluded from the training loss, per
        # the docstring on GraphNode.label above.
        if fsr.evidence_level is None:
            continue
        for edge in role_edges:
            emid = ((edge.start.x + edge.end.x) / 2.0, (edge.start.y + edge.end.y) / 2.0)
            side_key = _side_key_for_point(emid)
            if side_key is not None:
                side_label[side_key] = role

    gaps: dict[str, Optional[float]] = {}
    for side, (p1, p2) in sides.items():
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        gaps[side] = _nearest_seg_dist(mid, building_segments)
    ranked_sides = sorted((s for s in gaps if gaps[s] is not None), key=lambda s: gaps[s])
    rank_of = {s: r for r, s in enumerate(ranked_sides)}

    plot_node_id = poly_node_id[plot_idx]
    edge_node_ids: list[int] = []
    for side, (p1, p2) in sides.items():
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        length = _dist(p1, p2)
        gap = gaps[side]
        # NOTE: use `is not None`, never `x or default` -- a real distance of
        # exactly 0.0 (an edge touching the road) is falsy in Python and
        # would otherwise be silently replaced by the "not found" fallback,
        # which is precisely the case that most clearly marks the front edge.
        d_road = _nearest_seg_dist(mid, road_segments)
        d_block = _nearest_dist(mid, entrance_points)
        node = GraphNode(
            id=len(nodes),
            node_type="plot_edge",
            layer=polygons[plot_idx].layer,
            position=mid,
            label=side_label.get(side),
            features={
                "edge_length_ratio": length / plot_quarter_perimeter if plot_quarter_perimeter else 0.0,
                "edge_gap_norm": (gap / plot_diag) if gap is not None else 0.0,
                "edge_gap_rank": float(rank_of.get(side, -1)),
                "dist_to_road_norm": (d_road if d_road is not None else plot_diag) / plot_diag,
                "dist_to_block_norm": (d_block if d_block is not None else plot_diag) / plot_diag,
            },
            raw_measurements=(
                {"gap_m": gap, "evidence_level": fsr.evidence_level}
                if gap is not None
                else {"evidence_level": fsr.evidence_level}
            ),
        )
        edge_node_ids.append(node.id)
        nodes.append(node)
        node_positions.append(mid)

    # --- text nodes ---
    text_node_ids: list[int] = []
    for t in texts:
        node = GraphNode(
            id=len(nodes),
            node_type="text",
            layer=t.layer,
            position=t.position,
            raw_text=t.text,
            features={
                "text_length_norm": min(len(t.text) / 40.0, 1.0),
                "mentions_road": 1.0 if _ROAD_WORD_RE.search(t.text) else 0.0,
                "mentions_setback": 1.0 if _SETBACK_WORD_RE.search(t.text) else 0.0,
                "mentions_area": 1.0 if _AREA_WORD_RE.search(t.text) else 0.0,
                "has_number": 1.0 if _NUMBER_RE.search(t.text) else 0.0,
            },
        )
        text_node_ids.append(node.id)
        nodes.append(node)
        node_positions.append(t.position)

    # --- block nodes ---
    block_node_ids: list[int] = []
    for name, pos in blocks:
        node = GraphNode(id=len(nodes), node_type="block", layer="", position=pos, raw_text=name, features={})
        block_node_ids.append(node.id)
        nodes.append(node)
        node_positions.append(pos)

    # --- edges ---
    edges: list[GraphEdge] = []

    for eid in edge_node_ids:
        edges.append(GraphEdge(source=eid, target=plot_node_id, edge_type="belongs_to"))

    for i in range(len(polygons)):
        if i != plot_idx:
            edges.append(GraphEdge(source=poly_node_id[i], target=plot_node_id, edge_type="contains"))

    geometry_node_ids = [n.id for n in nodes if n.node_type in ("plot_polygon", "building_candidate", "road_polygon", "other_polygon", "plot_edge")]
    text_cutoff = plot_diag * 0.15
    for tid in text_node_ids:
        tpos = nodes[tid].position
        best_id, best_d = None, None
        for gid in geometry_node_ids:
            d = _dist(tpos, nodes[gid].position)
            if best_d is None or d < best_d:
                best_id, best_d = gid, d
        if best_id is not None and best_d <= text_cutoff:
            edges.append(GraphEdge(source=tid, target=best_id, edge_type="text_near"))

    for bid in block_node_ids:
        bpos = nodes[bid].position
        best_id, best_d = None, None
        for eid in edge_node_ids:
            d = _dist(bpos, nodes[eid].position)
            if best_d is None or d < best_d:
                best_id, best_d = eid, d
        if best_id is not None:
            edges.append(GraphEdge(source=bid, target=best_id, edge_type="block_near"))

    building_node_ids = [poly_node_id[i] for i in range(len(polygons)) if nodes[poly_node_id[i]].node_type == "building_candidate"]
    for eid in edge_node_ids:
        epos = nodes[eid].position
        best_id, best_d = None, None
        for bnid in building_node_ids:
            d = _dist(epos, nodes[bnid].position)
            if best_d is None or d < best_d:
                best_id, best_d = bnid, d
        if best_id is not None:
            edges.append(GraphEdge(source=eid, target=best_id, edge_type="nearest_gap"))

    # generic k-nearest-neighbor edges among ALL nodes, for general message
    # passing beyond the hand-designed relations above. O(n^2) -- fine for a
    # few hundred entities per plan; swap for a KD-tree if a file has
    # thousands of entities.
    n = len(nodes)
    for i in range(n):
        dists = [(_dist(node_positions[i], node_positions[j]), j) for j in range(n) if j != i]
        for _, j in heapq.nsmallest(k_nearest, dists):
            edges.append(GraphEdge(source=i, target=j, edge_type="knn"))

    return EntityGraph(document_id=document_id, nodes=nodes, edges=edges, unit_reason=unit_reason, warnings=warnings)


__all__ = [
    "GraphNode", "GraphEdge", "EntityGraph", "build_entity_graph",
    "NODE_TYPES", "ENTITY_KINDS", "EDGE_TYPES", "FEATURE_NAMES", "LABEL_CLASSES",
]