"""
Train the site-plan GNN node classifier on a directory of DXF files.

    python -m backend.gnn_extraction.train /path/to/dxf_dir --epochs 100

Splits BY FILE (never by node) into train/validation, so no graph's nodes
leak between the two sets. Reports per-class accuracy, not just overall
accuracy -- with only 8 classes and a real class imbalance (every plan has
exactly one "plot" node but usually several "other" nodes), overall accuracy
alone can look good while the model has simply learned to always guess the
majority class.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from backend.gnn_extraction.dataset import build_dataset
from backend.gnn_extraction.graph_builder import ENTITY_KINDS, FEATURE_NAMES, LABEL_CLASSES
from backend.gnn_extraction.model import SitePlanGAT

IN_CHANNELS = len(ENTITY_KINDS) + len(FEATURE_NAMES)


def split_by_file(dataset: list, val_fraction: float, seed: int) -> tuple[list, list]:
    items = dataset[:]
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * val_fraction)) if len(items) > 1 else 0
    return items[n_val:], items[:n_val]


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, dict[str, float | None], dict[str, int]]:
    model.eval()
    correct = total = 0
    per_class_correct = {c: 0 for c in LABEL_CLASSES}
    per_class_total = {c: 0 for c in LABEL_CLASSES}
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        mask = batch.train_mask
        if mask.sum() == 0:
            continue
        pred = out[mask].argmax(dim=1)
        true = batch.y[mask]
        correct += (pred == true).sum().item()
        total += mask.sum().item()
        for p, t in zip(pred.tolist(), true.tolist()):
            cls = LABEL_CLASSES[t]
            per_class_total[cls] += 1
            if p == t:
                per_class_correct[cls] += 1
    acc = correct / total if total else 0.0
    per_class_acc = {
        c: (per_class_correct[c] / per_class_total[c] if per_class_total[c] else None) for c in LABEL_CLASSES
    }
    return acc, per_class_acc, per_class_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxf_dir", type=Path, help="Directory of DXF files (convert DWG->DXF first)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2, help="GATConv layers. Keep this small (2 is the default) -- these graphs have only ~10-20 nodes per plan, and 3+ layers measurably over-smooth a node's own distinguishing feature (e.g. the plot node's area_ratio_to_plot=1.0) into its neighbors' average. Verified empirically: 3 layers gave 18.8%% accuracy on the 'plot' class, 2 layers gave 75%% on the same run.")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("gnn_checkpoint.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Building graphs from {args.dxf_dir} ...")
    report = build_dataset(args.dxf_dir)
    print(f"  {len(report.data)} usable graph(s), {len(report.empty)} with no closed polygons, {len(report.skipped)} failed to parse.")
    for name, reason in report.skipped[:10]:
        print(f"    SKIPPED {name}: {reason}")
    if len(report.skipped) > 10:
        print(f"    ... and {len(report.skipped) - 10} more (see full list above/in logs)")

    if not report.data:
        print(
            "\nERROR: 0 usable graphs were built -- nothing to train on. "
            "Common causes: a missing dependency (check the SKIPPED reasons above -- "
            "e.g. 'No module named ezdxf' means `pip install ezdxf`), the folder path "
            "is wrong, or the files aren't actually DXF (still raw .dwg, or renamed)."
        )
        raise SystemExit(1)

    if len(report.data) < 5:
        print(
            "\nWARNING: fewer than 5 usable graphs. This can still run end-to-end for a "
            "code smoke test, but treat any resulting checkpoint as illustrative only -- "
            "it is nowhere near enough data to trust for real extraction. Point this at "
            "your actual DWG-derived DXF batch once you have more files converted.\n"
        )

    train_set, val_set = split_by_file(report.data, args.val_fraction, args.seed)
    print(f"Split: {len(train_set)} train file(s), {len(val_set)} validation file(s).")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size) if val_set else DataLoader(train_set, batch_size=args.batch_size)

    model = SitePlanGAT(IN_CHANNELS, num_classes=len(LABEL_CLASSES), hidden_channels=args.hidden, num_layers=args.layers, heads=args.heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            mask = batch.train_mask
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[mask], batch.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            val_acc, _, _ = evaluate(model, val_loader)
            avg_loss = total_loss / n_batches if n_batches else float("nan")
            print(f"epoch {epoch:4d}  train_loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    print("\n=== Final validation ===")
    val_acc, per_class_acc, per_class_n = evaluate(model, val_loader)
    print(f"overall accuracy: {val_acc:.3f}")
    for cls in LABEL_CLASSES:
        acc = per_class_acc[cls]
        n = per_class_n[cls]
        print(f"  {cls:10s}  n={n:4d}  acc={'n/a' if acc is None else f'{acc:.3f}'}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": IN_CHANNELS,
            "hidden_channels": args.hidden,
            "num_layers": args.layers,
            "heads": args.heads,
            "label_classes": LABEL_CLASSES,
            "entity_kinds": ENTITY_KINDS,
            "feature_names": FEATURE_NAMES,
        },
        args.out,
    )
    print(f"\nSaved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
