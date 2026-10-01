"""
Architecture V2, Phase 3 -- tests for `site_graph.to_drawing_graph`, the
additive projection of `SiteGraph` into the generalized Drawing Evidence
Graph shape (`backend.schemas.evidence_graph.DrawingNode`/
`DrawingRelationship`). See ARCHITECTURE_V2.md, Deliverable C.2 item 2.

These are new tests for a new, purely additive function -- they do not
exercise `SiteGraph`/`build_site_graph`/`infer_*_node` behavior, which is
already covered indirectly through `test_dxf_extractor.py`'s real-fixture
tests and is unchanged by this phase.
"""

from __future__ import annotations

from backend.schemas.enums import EntityKind, RelationType
from backend.schemas.geometry import Point, Polygon
from backend.spatial_reasoning.site_graph import (
    NodeRole,
    SiteGraphNode,
    build_site_graph,
    to_drawing_graph,
)


def _rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon(points=[Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1)])


def test_every_site_graph_node_becomes_a_drawing_node():
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0, 0, 100, 100), role=NodeRole.PLOT),
        SiteGraphNode(id=1, polygon=_rect(10, 10, 30, 30), role=NodeRole.BUILDING),
    ]
    graph = build_site_graph(nodes)
    drawing_nodes, _relationships = to_drawing_graph(graph)

    assert {n.id for n in drawing_nodes} == {"0", "1"}
    by_id = {n.id: n for n in drawing_nodes}
    assert by_id["0"].node_kind == "geometry"
    assert by_id["0"].role_hint == EntityKind.PLOT
    assert by_id["1"].role_hint == EntityKind.BUILDING


def test_unknown_role_maps_to_no_role_hint():
    nodes = [SiteGraphNode(id=0, polygon=_rect(0, 0, 10, 10))]
    graph = build_site_graph(nodes)
    drawing_nodes, _ = to_drawing_graph(graph)
    assert drawing_nodes[0].role_hint is None


def test_containment_becomes_encloses_and_inside():
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0, 0, 100, 100)),   # plot
        SiteGraphNode(id=1, polygon=_rect(10, 10, 30, 30)),   # building, nested
    ]
    graph = build_site_graph(nodes)
    _nodes, rels = to_drawing_graph(graph)

    encloses = [r for r in rels if r.relation == RelationType.ENCLOSES]
    inside = [r for r in rels if r.relation == RelationType.INSIDE]
    assert len(encloses) == 1 and encloses[0].source_node_id == "0" and encloses[0].target_node_id == "1"
    assert len(inside) == 1 and inside[0].source_node_id == "1" and inside[0].target_node_id == "0"
    assert encloses[0].confidence == 1.0
    assert encloses[0].derivation == "deterministic_geometry"
    # A contained pair is never also reported as NEAR/ADJACENT -- containment subsumes it.
    assert not any(r.relation in (RelationType.NEAR, RelationType.ADJACENT) for r in rels)


def test_far_apart_nodes_get_no_relationship():
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0, 0, 1, 1)),
        SiteGraphNode(id=1, polygon=_rect(100, 100, 101, 101)),
    ]
    graph = build_site_graph(nodes)
    _nodes, rels = to_drawing_graph(graph)
    assert rels == []


def test_two_collinear_near_fragments_continue_each_other():
    """Two dashes of the same dash-dot boundary: near, collinear, close endpoints."""
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0.0, 0.0, 2.0, 0.5)),
        SiteGraphNode(id=1, polygon=_rect(2.05, 0.0, 4.05, 0.5)),
    ]
    graph = build_site_graph(nodes)
    _nodes, rels = to_drawing_graph(graph)

    relation_types = {r.relation for r in rels}
    assert RelationType.CONTINUES in relation_types
    assert RelationType.NEAR in relation_types or RelationType.ADJACENT in relation_types
    assert RelationType.INTERSECTS not in relation_types


def test_two_perpendicular_near_fragments_are_flagged_perpendicular():
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0.0, 0.0, 2.0, 0.5)),
        SiteGraphNode(id=1, polygon=_rect(2.05, 0.0, 2.55, 2.0)),
    ]
    graph = build_site_graph(nodes)
    _nodes, rels = to_drawing_graph(graph)

    perpendicular = [r for r in rels if r.relation == RelationType.PERPENDICULAR]
    assert len(perpendicular) == 1
    assert perpendicular[0].confidence > 0.9


def test_relationship_ids_are_unique():
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0, 0, 100, 100)),
        SiteGraphNode(id=1, polygon=_rect(10, 10, 30, 30)),
        SiteGraphNode(id=2, polygon=_rect(40, 40, 60, 60)),
    ]
    graph = build_site_graph(nodes)
    _nodes, rels = to_drawing_graph(graph)
    ids = [r.id for r in rels]
    assert len(ids) == len(set(ids))


def test_deterministic_relations_never_carry_association_or_gnn_derivation():
    # This function only ever computes deterministic geometric relations --
    # MEASURES/BELONGS_TO/CONFLICTS_WITH and gnn_inference rows are added by
    # later phases (association scoring, GNN relationship inference), never here.
    nodes = [
        SiteGraphNode(id=0, polygon=_rect(0, 0, 100, 100)),
        SiteGraphNode(id=1, polygon=_rect(10, 10, 30, 30)),
    ]
    graph = build_site_graph(nodes)
    _nodes, rels = to_drawing_graph(graph)
    assert all(r.derivation == "deterministic_geometry" for r in rels)
    assert all(r.relation not in (RelationType.MEASURES, RelationType.BELONGS_TO, RelationType.CONFLICTS_WITH) for r in rels)
