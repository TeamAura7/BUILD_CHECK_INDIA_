"""Stress / abstention run of the DXF pipeline over a folder of DXFs.

Needs no ground truth. For every file it records: crash / timeout, wall time,
and every scored field's value and confidence level. Each file is judged by
`scripts.classify_site_vs_floor_plans.score_entities` (a transparent heuristic,
not a trained model) so results can be split into "looks like a site plan" and
"does not". The headline number is how often the pipeline states a
HIGH/MEDIUM-confidence value on a region that is NOT a site plan -- there such
a value cannot be right for the reason the compliance check needs it.

    python -m backend.tools.stress_dxf D:/site_plan_regions --out data/stress/run.jsonl -j 6
    python -m backend.tools.stress_dxf --summarize data/stress/run.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TIMEOUT_S = 180.0
REPO = Path(__file__).resolve().parents[2]


def _worker(path: str) -> None:
    """Runs in a child process: print one JSON line."""
    from backend.corpus.runner import plan_to_predictions
    from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
    from backend.spatial_reasoning.pipeline import build_normalized_plan
    from scripts.classify_site_vs_floor_plans import _classify_one

    verdict = _classify_one(Path(path))
    started = time.time()
    out = {"file": Path(path).name, "site_label": verdict.label, "site_score": verdict.score}
    try:
        extraction = DXFHybridExtractor().extract(Path(path), Path(path).stem)
        plan = build_normalized_plan(extraction, plan_id=Path(path).stem)
        out["fields"] = plan_to_predictions(plan)
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a crash is a result
        out["status"] = f"crash:{type(exc).__name__}"
        out["error"] = str(exc)[:300]
    out["seconds"] = round(time.time() - started, 1)
    print("@@RESULT@@" + json.dumps(out, default=str))


def _run_one(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "backend.tools.stress_dxf", "--worker", str(path)],
            capture_output=True, text=True, timeout=TIMEOUT_S, cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        return {"file": path.name, "status": "timeout", "seconds": TIMEOUT_S}
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"file": path.name, "status": "hard_crash", "error": (proc.stderr or "")[-300:]}


def summarize(rows: list[dict]) -> str:
    lines = [f"files: {len(rows)}", "status: " + str(dict(Counter(r['status'] for r in rows)))]
    ok = [r for r in rows if r["status"] == "ok"]
    times = sorted(r["seconds"] for r in rows)
    if times:
        lines.append(f"seconds median {times[len(times)//2]:.1f}  p95 {times[int(len(times)*0.95)]:.1f}  max {times[-1]:.1f}")
    for label in ("site_plan", "uncertain", "floor_plan"):
        group = [r for r in ok if r["site_label"] == label]
        if not group:
            continue
        n = len(group)
        def answered(r, levels):
            return any(f["value"] is not None and f["confidence"] in levels
                       for name, f in r["fields"].items() if name != "building_use")
        anyv = sum(answered(r, {"HIGH", "MEDIUM", "LOW"}) for r in group)
        conf = sum(answered(r, {"HIGH", "MEDIUM"}) for r in group)
        high = sum(answered(r, {"HIGH"}) for r in group)
        flagged = sum(any(f["flagged"] for f in r["fields"].values()) for r in group)
        lines.append(f"{label:10s} n={n:3d}  any value {anyv:3d} ({100*anyv/n:.0f}%)  "
                     f"HIGH/MED value {conf:3d} ({100*conf/n:.0f}%)  HIGH value {high:3d} ({100*high/n:.0f}%)  flagged {flagged}")
    per_field = Counter()
    for r in ok:
        if r["site_label"] == "floor_plan":
            for name, f in r["fields"].items():
                if name != "building_use" and f["value"] is not None and f["confidence"] in {"HIGH", "MEDIUM"}:
                    per_field[name] += 1
    if per_field:
        lines.append("HIGH/MED values asserted on floor-plan-labelled files, by field: " + str(dict(per_field.most_common())))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--worker")
    ap.add_argument("--summarize")
    a = ap.parse_args()
    if a.worker:
        return _worker(a.worker)
    if a.summarize:
        print(summarize([json.loads(l) for l in Path(a.summarize).read_text(encoding="utf-8").splitlines() if l.strip()]))
        return
    files = sorted(Path(a.folder).glob("*.dxf"))[: a.limit]
    done = set()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # resumable
        done = {json.loads(l)["file"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [f for f in files if f.name not in done]
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(a.j) as pool:
        for i, row in enumerate(pool.map(_run_one, todo), 1):
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print(summarize([json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]))


if __name__ == "__main__":
    main()
