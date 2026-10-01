"""
GAT node classifier for site-plan entity graphs (backend.gnn_extraction.graph_builder).

Predicts, for every polygon and plot_edge node, one of LABEL_CLASSES
(plot/building/road/other for polygons; front/rear/left/right for edges).
Text and block nodes are never classified (they're excluded from the
training loss via `train_mask`) -- they exist in the graph purely to give
polygon/edge nodes something to reason about (a nearby "ROAD" text, a nearby
gate block), via message passing.

Deliberately small: this dataset size (hundreds to low thousands of site
plans, not millions) will overfit a large model. Empirically (see
train.py), 2 GATConv layers outperform 3 on these graphs -- with only
~10-20 nodes per plan, a third round of message passing measurably
over-smooths a node's own distinguishing feature into its neighbors'
average (e.g. the plot node's area_ratio_to_plot=1.0 signal gets diluted).
Resist the urge to go deeper or wider before you have evidence of
underfitting, not overfitting.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SitePlanGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 (at least one hidden layer + one output layer)")

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        # Final layer: concat=False averages the heads down to num_classes
        # directly, rather than needing a separate linear head.
        self.convs.append(GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


__all__ = ["SitePlanGAT"]
