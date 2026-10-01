#!/usr/bin/env bash
# Full reproduction of every fresh result in experiments/results/fresh/ (about 45-60 min on 2 CPUs).
# Run from the repository root:   bash experiments/run_everything.sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p experiments/results/fresh experiments/logs experiments/figures
git rev-parse HEAD > experiments/results/fresh/commit.txt

# 1. Canonical harness (repository's own evaluation, unchanged): pdf / dxf / reconciled   (~19 min)
python3 -m backend.tools.run_corpus eval --split dev --no-cache \
    --json experiments/results/fresh/corpus_eval_dev.json | tee experiments/logs/corpus_eval_dev.log

# 2. Dataset + DXF content inventory (no extraction)
python3 experiments/dataset_table.py
python3 experiments/dxf_inventory.py

# 3. All pipeline variants, one fresh process per (variant, plan), sequential    (~30 min)
rm -rf experiments/results/fresh/variants
bash experiments/run_all_variants.sh | tee experiments/results/fresh/variants_run_log.txt

# 4. Scoring, reconciliation, compliance propagation, margins, DXF scale
python3 experiments/analyze_extraction.py
python3 experiments/probe_reconciliation.py
python3 experiments/reconciliation_stats.py
python3 experiments/compliance_eval.py
python3 experiments/margin_analysis.py
python3 experiments/dxf_scale_analysis.py

echo "done"
