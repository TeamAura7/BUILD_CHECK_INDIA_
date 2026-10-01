"""Paired significance tests on the development corpus (no new data).
- Exact McNemar (two-sided binomial on discordant pairs) for field correctness (84 values)
  and for rule-level correctness (32 decidable checks), each configuration vs the default.
- Wilson 95% intervals for confident-wrong rates.
- Plan-level sign-flip randomisation test (8 plans => 2^8 relabelings) on per-plan
  difference in correct counts, as a check that accounts for within-plan correlation."""
import csv, itertools, json, math, sys
from collections import defaultdict
from pathlib import Path
F = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/fresh")

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))

def signflip(diffs):
    obs = abs(sum(diffs)); cnt = 0; tot = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        tot += 1; cnt += abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-9
    return cnt / tot

field = defaultdict(dict)
for r in csv.DictReader(open(F / "field_level.csv")):
    if r["status"] in ("CORRECT", "WRONG", "MISSING", "ABSTAINED_CONFLICT"):
        field[r["variant"]][(r["plan"], r["field"])] = r["status"] == "CORRECT"
rule = defaultdict(dict)
for r in csv.DictReader(open(F / "compliance_rule_level.csv")):
    if r["development_area"] == "A" and r["gt_status"] in ("PASS", "FAIL"):
        rule[r["variant"]][(r["plan"], r["rule_id"])] = r["outcome"] == "CORRECT_DECISION"
base = "pdf_hybrid"; out = []
for v in sorted(field):
    if v == base: continue
    keys = field[base].keys() & field[v].keys()
    b = sum(field[base][k] and not field[v][k] for k in keys); c = sum(field[v][k] and not field[base][k] for k in keys)
    plans = sorted({k[0] for k in keys})
    diffs = [sum(field[base][k] for k in keys if k[0] == p) - sum(field[v][k] for k in keys if k[0] == p) for p in plans]
    row = {"comparison": f"{base} vs {v}", "n_fields": len(keys), "base_only": b, "other_only": c,
           "mcnemar_p_fields": round(mcnemar_exact(b, c), 5), "signflip_p_plans": round(signflip(diffs), 4)}
    if v in rule:
        rk = rule[base].keys() & rule[v].keys()
        rb = sum(rule[base][k] and not rule[v][k] for k in rk); rc = sum(rule[v][k] and not rule[base][k] for k in rk)
        row.update({"n_checks": len(rk), "rule_base_only": rb, "rule_other_only": rc, "mcnemar_p_rules": round(mcnemar_exact(rb, rc), 5)})
    out.append(row)
cw = {}
for r in csv.DictReader(open(F / "pipeline_comparison.csv")):
    k, n = int(r["confident_wrong"]), int(r["confident_answers"])
    lo, hi = wilson(k, n); cw[r["variant"]] = {"cw": f"{k}/{n}", "wilson95": [round(lo, 3), round(hi, 3)]}
json.dump({"paired_tests": out, "confident_wrong_wilson": cw}, open(F / "stats_tests.json", "w"), indent=1)
for r in out: print(r)
for k, v in cw.items(): print(k, v)

# Gate ablation: false verdicts (FALSE_PASS/FALSE_FAIL) with vs without the LOW->review gate.
fv = defaultdict(dict)
for r in csv.DictReader(open(F / "compliance_rule_level.csv")):
    if r["development_area"] == "A" and r["gt_status"] in ("PASS", "FAIL"):
        fv[r["variant"]][(r["plan"], r["rule_id"])] = r["outcome"] in ("FALSE_PASS", "FALSE_FAIL")
gate = []
for v in ("dxf_hybrid", "dxf_evidence_decision", "pdf_hybrid", "pdf_evidence_decision"):
    u = v + "__ungated"
    if v in fv and u in fv:
        ks = fv[v].keys() & fv[u].keys()
        b = sum(fv[u][k] and not fv[v][k] for k in ks); c = sum(fv[v][k] and not fv[u][k] for k in ks)
        gate.append({"config": v, "false_verdicts_ungated_only": b, "false_verdicts_gated_only": c,
                     "mcnemar_p": round(mcnemar_exact(b, c), 5)})
d = json.load(open(F / "stats_tests.json")); d["gate_ablation"] = gate
json.dump(d, open(F / "stats_tests.json", "w"), indent=1)
for g in gate: print(g)
