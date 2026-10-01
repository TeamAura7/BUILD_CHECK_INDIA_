"""
Builds a training dataset from a directory of DXF files (a DWG batch that's
already been converted via ODA File Converter, or native DXFs) by running
`build_entity_graph` on each and converting to `torch_geometric.data.Data`.

One file that fails to parse should never abort a 1,500-file batch job --
failures are logged and skipped, and reported in the returned `skipped` list
so you can see which files need attention without losing the rest of the run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.gnn_extraction.graph_builder import build_entity_graph


@dataclass
class DatasetBuildReport:
    data: list  # list[torch_geometric.data.Data]
    skipped: list[tuple[str, str]]  # (filename, reason)
    empty: list[str]  # files that parsed but had no closed polygons


def build_dataset(dxf_dir: Path, k_nearest: int = 4, pattern: str = "*.dxf") -> DatasetBuildReport:
    data = []
    skipped: list[tuple[str, str]] = []
    empty: list[str] = []

    for path in sorted(Path(dxf_dir).glob(pattern)):
        try:
            graph = build_entity_graph(path, path.stem, k_nearest=k_nearest)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill a 1500-file batch
            skipped.append((path.name, str(exc)))
            continue
        if not graph.nodes:
            empty.append(path.name)
            continue
        data.append(graph.to_pyg_data())

    return DatasetBuildReport(data=data, skipped=skipped, empty=empty)


__all__ = ["DatasetBuildReport", "build_dataset"]
