"""
Tests for backend/gnn_extraction/graph_builder.py.

These validate the graph-construction logic (node/edge/feature correctness)
against synthetic DXF fixtures with known ground truth. They do NOT test a
trained model -- there isn't one yet; this is the data layer the model will
eventually be trained on. See docs/dwg_gnn_compliance_plan.md.
"""
from __future__ import annotations

import math

import pytest

from backend.gnn_extraction.graph_builder import ENTITY_KINDS, LABEL_CLASSES, NODE_TYPES, build_entity_graph
from tests.fixtures.dxf_builders import no_text_site_plan_dxf, rectangular_site_plan_dxf

PLOT_W, PLOT_D = 12.0, 18.0
FRONT, REAR, LEFT, RIGHT = 3.0, 2.0, 1.5, 1.0
ROAD_W = 9.0
PLOT_DIAG = math.hypot(PLOT_W, PLOT_D)


def _node_counts(graph):
    return graph.summary()


def test_node_counts_match_fixture(tmp_path):
    dxf_path = rectangular_site_plan_dxf(tmp_path / "plan.dxf")
    graph = build_entity_graph(dxf_path, "plan")

    assert graph.warnings == []
    counts = _node_counts(graph)
    assert counts["plot_polygon"] == 1
    assert counts["building_candidate"] == 1
    assert counts["road_polygon"] == 1  # the road polygon now gets its own bucket, not lumped into "other"
    assert counts["plot_edge"] == 4
    assert counts["text"] == 2
    assert counts["block"] == 1
    assert all(nt in NODE_TYPES for nt in counts)


def test_setback_gaps_are_geometrically_exact(tmp_path):
    """The core correctness claim of this module: edge_gap_norm * plot
    diagonal must equal the true perpendicular gap to the building, not an
    approximation -- this is the whole reason to use DXF vector geometry
    instead of a CV/VLM estimate."""
    dxf_path = rectangular_site_plan_dxf(
        tmp_path / "plan.dxf",
        plot_width=PLOT_W, plot_depth=PLOT_D,
        front_setback=FRONT, rear_setback=REAR, left_setback=LEFT, right_setback=RIGHT,
        road_width=ROAD_W,
    )
    graph = build_entity_graph(dxf_path, "plan")

    expected_by_position = {
        (PLOT_W / 2, PLOT_D): REAR,   # north edge -> rear setback
        (PLOT_W / 2, 0.0): FRONT,     # south edge -> front setback (faces the road)
        (PLOT_W, PLOT_D / 2): RIGHT,  # east edge
        (0.0, PLOT_D / 2): LEFT,      # west edge
    }

    edge_nodes = [n for n in graph.nodes if n.node_type == "plot_edge"]
    assert len(edge_nodes) == 4
    for node in edge_nodes:
        expected_gap = expected_by_position[node.position]
        actual_gap = node.features["edge_gap_norm"] * PLOT_DIAG
        assert actual_gap == pytest.approx(expected_gap, abs=0.05)


def test_front_edge_identifiable_from_geometry_alone(tmp_path):
    """No text anywhere in the file -- front-side identification must still
    work from road proximity + gate-block proximity alone. Rear setback is
    deliberately made the LARGEST gap here, so the test actually exercises
    disambiguation (road/gate proximity) rather than happening to agree with
    'largest gap = front' by coincidence."""
    front, rear, left, right = 2.0, 4.5, 1.5, 1.0  # rear > front on purpose
    dxf_path = no_text_site_plan_dxf(
        tmp_path / "plan_no_text.dxf",
        plot_width=PLOT_W, plot_depth=PLOT_D,
        front_setback=front, rear_setback=rear, left_setback=left, right_setback=right,
        road_width=ROAD_W,
    )
    graph = build_entity_graph(dxf_path, "plan_no_text")

    assert _node_counts(graph).get("text", 0) == 0

    edge_nodes = [n for n in graph.nodes if n.node_type == "plot_edge"]
    front_edge = min(edge_nodes, key=lambda n: n.features["dist_to_road_norm"])
    assert front_edge.position == (PLOT_W / 2, 0.0)
    assert front_edge.features["dist_to_road_norm"] == pytest.approx(0.0, abs=1e-6)
    assert front_edge.features["dist_to_block_norm"] == pytest.approx(0.0, abs=1e-6)

    # The largest gap is the REAR edge in this fixture (rear=4.5 > front=2.0)
    # -- confirm it's correctly NOT the one closest to the road, i.e. gap
    # size alone would mislabel the front here, but road/gate distance won't.
    largest_gap_edge = max(edge_nodes, key=lambda n: n.features["edge_gap_norm"])
    assert largest_gap_edge.position != front_edge.position
    assert largest_gap_edge.position == (PLOT_W / 2, PLOT_D)  # north edge = rear


def test_text_nodes_carry_geometric_flags_without_hardcoded_keywords(tmp_path):
    dxf_path = rectangular_site_plan_dxf(tmp_path / "plan.dxf", road_width=ROAD_W)
    graph = build_entity_graph(dxf_path, "plan")

    text_nodes = [n for n in graph.nodes if n.node_type == "text"]
    assert len(text_nodes) == 2
    road_text = next(n for n in text_nodes if "ROAD" in (n.raw_text or ""))
    area_text = next(n for n in text_nodes if "AREA" in (n.raw_text or ""))
    assert road_text.features["mentions_road"] == 1.0
    assert area_text.features["mentions_area"] == 1.0
    assert area_text.features["mentions_road"] == 0.0


def test_edges_connect_expected_node_types(tmp_path):
    dxf_path = rectangular_site_plan_dxf(tmp_path / "plan.dxf")
    graph = build_entity_graph(dxf_path, "plan")

    edge_types_present = {e.edge_type for e in graph.edges}
    assert {"belongs_to", "contains", "text_near", "block_near", "nearest_gap", "knn"} <= edge_types_present

    id_to_node = {n.id: n for n in graph.nodes}
    for e in graph.edges:
        if e.edge_type == "belongs_to":
            assert id_to_node[e.source].node_type == "plot_edge"
            assert id_to_node[e.target].node_type == "plot_polygon"
        if e.edge_type == "nearest_gap":
            assert id_to_node[e.source].node_type == "plot_edge"
            assert id_to_node[e.target].node_type == "building_candidate"


def test_feature_vector_has_fixed_length(tmp_path):
    dxf_path = rectangular_site_plan_dxf(tmp_path / "plan.dxf")
    graph = build_entity_graph(dxf_path, "plan")
    lengths = {len(n.feature_vector()) for n in graph.nodes}
    assert len(lengths) == 1, "every node must produce the same-length feature vector regardless of node_type"


def test_semantic_role_is_a_label_not_a_feature(tmp_path):
    """The whole point of separating entity_kind (input) from label (target):
    the feature vector must not let the model just read off the answer it's
    being trained to predict. A polygon and a plot_edge node with identical
    entity_kind must be distinguishable ONLY by their geometric features
    (area ratio, gap, etc), never by a hidden 'this is secretly the building'
    signal baked into the vector."""
    dxf_path = rectangular_site_plan_dxf(tmp_path / "plan.dxf")
    graph = build_entity_graph(dxf_path, "plan")

    polygon_nodes = [n for n in graph.nodes if n.entity_kind == "polygon"]
    assert {n.node_type for n in polygon_nodes} == {"plot_polygon", "building_candidate", "road_polygon"}
    # all polygon nodes must share the same 4-length one-hot prefix (entity_kind),
    # i.e. nothing in the feature vector reveals which specific polygon role
    # a node has -- that information lives only in `.label`.
    kind_prefixes = {tuple(n.feature_vector()[: len(ENTITY_KINDS)]) for n in polygon_nodes}
    assert kind_prefixes == {tuple(1.0 if k == "polygon" else 0.0 for k in ENTITY_KINDS)}

    labels = {n.node_type: n.label for n in polygon_nodes}
    assert labels["plot_polygon"] == "plot"
    assert labels["building_candidate"] == "building"
    assert labels["road_polygon"] == "road"
    for n in polygon_nodes:
        assert n.label in LABEL_CLASSES
        assert n.label_id == LABEL_CLASSES.index(n.label)


def test_plot_edge_labels_match_known_setback_layout(tmp_path):
    """Front/rear/left/right labels come from resolve_front_side (road +
    access-evidence based), not from any information baked into the plot_edge
    feature vector -- confirms the silver-labeling loop actually resolves a
    side for every edge in the common case (road present, rectangular plot)."""
    dxf_path = rectangular_site_plan_dxf(tmp_path / "plan.dxf")
    graph = build_entity_graph(dxf_path, "plan")

    edge_nodes = [n for n in graph.nodes if n.node_type == "plot_edge"]
    labeled = {n.label for n in edge_nodes if n.label is not None}
    assert labeled == {"front", "rear", "left", "right"}, "expected all four sides to resolve a label for this fixture"

    front_node = next(n for n in edge_nodes if n.label == "front")
    assert front_node.position == (PLOT_W / 2, 0.0)  # the edge that touches the road


def test_no_closed_polygons_returns_empty_graph_not_crash(tmp_path):
    import ezdxf

    doc = ezdxf.new("R2018")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_text("JUST SOME TEXT, NO GEOMETRY").dxf.insert = (0, 0)
    path = tmp_path / "empty.dxf"
    doc.saveas(str(path))

    graph = build_entity_graph(path, "empty")
    assert graph.nodes == []
    assert graph.edges == []
    assert any("No closed polygons" in w for w in graph.warnings)
