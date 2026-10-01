"""
Run a trained checkpoint (from train.py) on a new DXF's entity graph.

    from backend.gnn_extraction.predict import predict_roles
    predictions = predict_roles(Path("new_plan.dxf"), Path("gnn_checkpoint.pt"))

Returns per-node predicted role + confidence for every polygon/plot_edge
node. This is the ONLY place the model's output should be consumed --
remember the value itself (width, area, gap distance) still comes from the
node's real geometry, never from the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from backend.gnn_extraction.graph_builder import GraphNode, build_entity_graph
from backend.gnn_extraction.model import SitePlanGAT


@dataclass
class RolePrediction:
    node: GraphNode
    predicted_label: str
    confidence: float
    heuristic_label: Optional[str]  # what the silver-labeling heuristic said, for comparison
    agrees_with_heuristic: bool


def load_model(checkpoint_path: Path) -> tuple[SitePlanGAT, list[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SitePlanGAT(
        in_channels=ckpt["in_channels"],
        num_classes=len(ckpt["label_classes"]),
        hidden_channels=ckpt["hidden_channels"],
        num_layers=ckpt["num_layers"],
        heads=ckpt.get("heads", 4),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["label_classes"]


@torch.no_grad()
def predict_roles(dxf_path: Path, checkpoint_path: Path, k_nearest: int = 4) -> list[RolePrediction]:
    model, label_classes = load_model(checkpoint_path)
    graph = build_entity_graph(dxf_path, dxf_path.stem, k_nearest=k_nearest)
    if not graph.nodes:
        return []

    data = graph.to_pyg_data()
    logits = model(data.x, data.edge_index)
    probs = F.softmax(logits, dim=1)

    results = []
    for i, node in enumerate(graph.nodes):
        if node.entity_kind not in ("polygon", "plot_edge"):
            continue  # text/block nodes are never classified
        pred_idx = int(probs[i].argmax())
        results.append(
            RolePrediction(
                node=node,
                predicted_label=label_classes[pred_idx],
                confidence=float(probs[i, pred_idx]),
                heuristic_label=node.label,
                agrees_with_heuristic=(node.label == label_classes[pred_idx]),
            )
        )
    return results


__all__ = ["RolePrediction", "load_model", "predict_roles"]
