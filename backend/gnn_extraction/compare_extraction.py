"""
Run the trained GNN on a real DXF and compare its predicted dimensions
against the deterministic heuristic (DXFHybridExtractor) side by side.

THIS IS A DIAGNOSTIC TOOL, NOT A TRUSTED EXTRACTOR. As of the last
recorded training run, per-class validation accuracy was:

    plot: 64%  building: 71%  front: 40%  rear: 13%  left: 13%  right: 42%
    road: no examples at all (not yet trained)

front/rear/left are at or below the 25% random-guess baseline for a
4-class problem. Do not wire this into anything that produces a compliance
verdict. What it IS useful for: seeing where the model agrees or disagrees
with the heuristic on real files, which is exactly the active-learning
review queue from the build plan -- disagreements are what should get
human-reviewed next, not silently trusted either way.

Usage:
    python -m backend.gnn_extraction.compare_extraction plan.dxf gnn_checkpoint.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
from backend.gnn_extraction.predict import RolePrediction, predict_roles

# Which raw_measurements key to read for each predicted role, and the
# heuristic's equivalent IndependentMeasurement field name(s) to compare
# against -- polygon roles report width/depth/area, plot_edge roles report
# a single gap distance.
_ROLE_TO_HEURISTIC_FIELDS = {
    "plot": [("width_m", "plot.width"), ("depth_m", "plot.depth"), ("area_m2", "plot.area")],
    "building": [("width_m", "building.width"), ("depth_m", "building.depth"), ("area_m2", "building.footprint_area")],
    "road": [("width_m", "road.width")],
    "front": [("gap_m", "setbacks.front")],
    "rear": [("gap_m", "setbacks.rear")],
    "left": [("gap_m", "setbacks.left")],
    "right": [("gap_m", "setbacks.right")],
}


def _best_prediction_per_role(predictions: list[RolePrediction]) -> dict[str, RolePrediction]:
    best: dict[str, RolePrediction] = {}
    for p in predictions:
        if p.predicted_label not in best or p.confidence > best[p.predicted_label].confidence:
            best[p.predicted_label] = p
    return best


def _heuristic_measurements(dxf_path: Path) -> dict[str, tuple[Optional[float], float]]:
    """field -> (value, confidence) from the deterministic extractor, for
    every field it managed to resolve on this file."""
    extraction = DXFHybridExtractor().extract(dxf_path, dxf_path.stem)
    out: dict[str, tuple[Optional[float], float]] = {}
    if extraction.independent_cv:
        for m in extraction.independent_cv.measurements:
            value = m.value_m if m.value_m is not None else m.value
            out[m.field] = (value, m.confidence)
    return out


def compare(dxf_path: Path, checkpoint_path: Path) -> list[dict]:
    predictions = predict_roles(dxf_path, checkpoint_path)
    best_by_role = _best_prediction_per_role(predictions)
    heuristic = _heuristic_measurements(dxf_path)

    rows = []
    for role, field_pairs in _ROLE_TO_HEURISTIC_FIELDS.items():
        pred = best_by_role.get(role)
        for raw_key, heuristic_field in field_pairs:
            model_value = pred.node.raw_measurements.get(raw_key) if pred else None
            model_conf = pred.confidence if pred else None
            heur_value, heur_conf = heuristic.get(heuristic_field, (None, None))
            agree = None
            if model_value is not None and heur_value is not None:
                tol = max(0.15, 0.05 * max(abs(model_value), abs(heur_value)))
                agree = abs(model_value - heur_value) <= tol
            rows.append({
                "field": heuristic_field,
                "model_value": model_value,
                "model_confidence": model_conf,
                "heuristic_value": heur_value,
                "heuristic_confidence": heur_conf,
                "agree": agree,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dxf_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    args = parser.parse_args()

    rows = compare(args.dxf_path, args.checkpoint_path)

    print(f"{'field':<24} {'model':>10} {'conf':>6}   {'heuristic':>10} {'conf':>6}   agree?")
    print("-" * 78)
    n_agree = n_compared = n_model_only = n_heuristic_only = n_neither = 0
    for r in rows:
        mv = f"{r['model_value']:.2f}" if r["model_value"] is not None else "--"
        mc = f"{r['model_confidence']:.2f}" if r["model_confidence"] is not None else "--"
        hv = f"{r['heuristic_value']:.2f}" if r["heuristic_value"] is not None else "--"
        hc = f"{r['heuristic_confidence']:.2f}" if r["heuristic_confidence"] is not None else "--"
        agree_str = "n/a" if r["agree"] is None else ("YES" if r["agree"] else "NO <<<")
        print(f"{r['field']:<24} {mv:>10} {mc:>6}   {hv:>10} {hc:>6}   {agree_str}")
        if r["agree"] is True:
            n_agree += 1
        if r["agree"] is not None:
            n_compared += 1
        elif r["model_value"] is not None and r["heuristic_value"] is None:
            n_model_only += 1
        elif r["model_value"] is None and r["heuristic_value"] is not None:
            n_heuristic_only += 1
        else:
            n_neither += 1

    print(f"\n{n_agree}/{n_compared} comparable field(s) agree within tolerance.")
    if n_model_only:
        print(f"{n_model_only} field(s) only the model produced a value for.")
    if n_heuristic_only:
        print(f"{n_heuristic_only} field(s) only the heuristic produced a value for.")
    print(
        "\nDisagreements ('NO <<<') and model-only fields are exactly what the "
        "active-learning review queue from the build plan is for -- worth a human "
        "look before trusting either side, not an automatic tie-break."
    )


if __name__ == "__main__":
    main()
