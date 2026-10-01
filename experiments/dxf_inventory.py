"""Phase 5: raw DXF content inventory (what evidence each DXF actually carries). Read-only, ezdxf only."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import ezdxf

REPO = Path(__file__).resolve().parents[1]
rows = []
for p in sorted((REPO / "data/test_plans").glob("PLAN*.dxf")):
    try:
        doc = ezdxf.readfile(p)
    except Exception as exc:
        rows.append({"plan": p.stem, "error": str(exc)[:200]})
        continue
    msp = doc.modelspace()
    types = Counter(e.dxftype() for e in msp)
    texts = [e.dxf.text for e in msp if e.dxftype() == "TEXT"] + [e.text for e in msp if e.dxftype() == "MTEXT"]
    numeric_like = [t for t in texts if any(ch.isdigit() for ch in t)]
    layers = Counter(e.dxf.layer for e in msp)
    blocks_with_text = 0
    for ins in msp.query("INSERT"):
        blk = doc.blocks.get(ins.dxf.name)
        if blk is not None and any(e.dxftype() in ("TEXT", "MTEXT") for e in blk):
            blocks_with_text += 1
    ext = (doc.header.get("$EXTMIN"), doc.header.get("$EXTMAX"))
    rows.append({
        "plan": p.stem, "size_mb": round(p.stat().st_size / 2**20, 1), "acadver": doc.dxfversion,
        "insunits": doc.header.get("$INSUNITS"), "measurement": doc.header.get("$MEASUREMENT"),
        "n_entities": sum(types.values()), "n_layers_used": len(layers),
        "LINE": types.get("LINE", 0), "LWPOLYLINE": types.get("LWPOLYLINE", 0), "POLYLINE": types.get("POLYLINE", 0),
        "ARC": types.get("ARC", 0), "CIRCLE": types.get("CIRCLE", 0), "HATCH": types.get("HATCH", 0),
        "SPLINE": types.get("SPLINE", 0), "SOLID": types.get("SOLID", 0),
        "TEXT": types.get("TEXT", 0), "MTEXT": types.get("MTEXT", 0), "DIMENSION": types.get("DIMENSION", 0),
        "INSERT": types.get("INSERT", 0), "inserts_whose_block_has_text": blocks_with_text,
        "text_with_digits": len(numeric_like), "text_samples": ("[redacted: may contain applicant/property identifiers]" if any("Khata" in t or "PID" in t for t in texts) else json.dumps(numeric_like[:12])),
        "top_layers": json.dumps(layers.most_common(8)),
        "extents": json.dumps([list(map(lambda v: round(v, 2), e)) if e is not None else None for e in ext]),
    })
    print(rows[-1]["plan"], {k: rows[-1][k] for k in ("insunits", "n_entities", "TEXT", "MTEXT", "DIMENSION", "LINE", "LWPOLYLINE")}, flush=True)
out = REPO / "experiments/results/fresh/dxf_inventory.csv"
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
