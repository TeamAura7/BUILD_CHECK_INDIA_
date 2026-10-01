#!/usr/bin/env bash
# Runs every extraction variant over every corpus plan, sequentially (one
# process at a time, so wall-clock timings and the extractors' internal 180 s
# budget are not distorted by CPU contention). Each run is a fresh Python
# process with an outer 600 s timeout (the same as backend/corpus/runner.py's
# EXTRACTION_TIMEOUT_S); an outer timeout is RECORDED, never skipped.
set -u
cd "$(dirname "$0")/.."
OUT=experiments/results/fresh/variants
PLANS="PLAN1 PLAN2 PLAN4 PLAN5 PLAN6 PLAN7 PLAN8 PLAN9"
VARIANTS="${VARIANTS:-pdf_hybrid pdf_native_only pdf_native_ocr pdf_legacy_resolver pdf_evidence_decision dxf_hybrid dxf_evidence_decision}"
for v in $VARIANTS; do
  mkdir -p "$OUT/$v"
  for p in $PLANS; do
    if [ -f "$OUT/$v/$p.meta.json" ]; then echo "skip $v $p"; continue; fi
    envs=""
    [ "$v" = "pdf_evidence_decision" ] && envs="USE_EVIDENCE_DECISION_ENGINE=true"
    [ "$v" = "dxf_evidence_decision" ] && envs="USE_DXF_EVIDENCE_DECISION_ENGINE=true"
    start=$(date +%s)
    env $envs timeout 600 python3 experiments/run_variant.py "$v" "$p" "$OUT/$v" > "$OUT/$v/$p.stdout.log" 2>&1
    rc=$?
    wall=$(( $(date +%s) - start ))
    if [ $rc -eq 124 ]; then
      printf '{"variant":"%s","plan":"%s","status":"outer_timeout_600s","total_seconds":%s}\n' "$v" "$p" "$wall" > "$OUT/$v/$p.meta.json"
    elif [ ! -f "$OUT/$v/$p.meta.json" ]; then
      printf '{"variant":"%s","plan":"%s","status":"process_failed_rc_%s","total_seconds":%s}\n' "$v" "$p" "$rc" "$wall" > "$OUT/$v/$p.meta.json"
    fi
    echo "$(date +%T) $v $p rc=$rc ${wall}s"
  done
done
echo ALL_DONE
