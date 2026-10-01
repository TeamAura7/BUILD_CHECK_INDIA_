"""
Cross-platform orchestrator (Windows/macOS/Linux). One fresh process per plan.

  python experiments/run_vlm_eval.py                     # all 8 plans, backend from .env
  python experiments/run_vlm_eval.py --backend api --model <model-id>
  python experiments/run_vlm_eval.py --plans PLAN5 PLAN7

Writes experiments/results/fresh/variants/pdf_hybrid_vlm_<backend>/ and then
runs the unchanged analyze_extraction.py and compliance_eval.py, which pick up
any extra variant directory automatically. Finally prints vlm_summary.json.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLANS = ["PLAN1", "PLAN2", "PLAN4", "PLAN5", "PLAN6", "PLAN7", "PLAN8", "PLAN9"]

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default=None, help="smolvlm | qwen | api (default: settings)")
ap.add_argument("--model", default=None, help="model id (api) or HF id/local path (smolvlm/qwen)")
ap.add_argument("--plans", nargs="*", default=PLANS)
ap.add_argument("--timeout", type=int, default=3600, help="seconds per plan")
ap.add_argument("--skip-analysis", action="store_true")
a = ap.parse_args()

env = dict(os.environ, VISION_ENABLED="true")
if a.backend:
    env["VISION_BACKEND"] = a.backend
if a.model:
    env["VISION_API_MODEL" if (a.backend or env.get("VISION_BACKEND")) == "api" else "VISION_MODEL_NAME"] = a.model
backend = env.get("VISION_BACKEND", "smolvlm")
out = REPO / "experiments/results/fresh/variants" / f"pdf_hybrid_vlm_{backend}"
for pid in a.plans:
    try:
        subprocess.run([sys.executable, str(REPO / "experiments/run_vlm_variant.py"), pid, str(out)],
                       cwd=REPO, env=env, timeout=a.timeout, check=False)
    except subprocess.TimeoutExpired:
        (out / f"{pid}.meta.json").write_text(json.dumps({"plan": pid, "status": "timeout", "vlm_valid": False}))
        print(pid, "timeout")

metas = [json.loads(p.read_text()) for p in sorted(out.glob("*.meta.json"))]
summary = {
    "variant": out.name, "backend": backend,
    "runs": len(metas), "vlm_valid_runs": sum(bool(m.get("vlm_valid")) for m in metas),
    "invalid": {m["plan"]: m.get("status") if m.get("status") != "ok" else "vision failed (see warnings)"
                for m in metas if not m.get("vlm_valid")},
    "dropped_ungrounded_total": sum(m.get("dropped_ungrounded", 0) for m in metas),
    "grounding": {m["plan"]: m.get("grounding") for m in metas},
    "seconds": {m["plan"]: m.get("total_seconds") for m in metas},
}
(out / "vlm_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
# Invalid runs are CV-only fallbacks: hide their plan.json so the analyzer scores them as not run.
for m in metas:
    if not m.get("vlm_valid"):
        pj = out / f"{m['plan']}.plan.json"
        if pj.exists():
            pj.replace(out / f"{m['plan']}.plan.json.invalid")
if summary["vlm_valid_runs"] < len(metas):
    print("\nWARNING: some runs are NOT valid VLM runs (the VLM failed and the pipeline fell back to CV). "
          "Do not report them as VLM results.")
if not a.skip_analysis:
    for script in ("analyze_extraction.py", "compliance_eval.py"):
        p = REPO / "experiments" / script
        if p.exists():
            subprocess.run([sys.executable, str(p)], cwd=REPO, check=False)
print(f"\nDone. Send back the folder: {out}  plus experiments/results/fresh/*.csv")
