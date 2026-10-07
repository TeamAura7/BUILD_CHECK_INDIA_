# Reproducing the IJCM manuscript results

This repository is the evaluated code snapshot (internal commit `699fb74`), published with the permit drawings removed. Three things have been added to it:

- `experiments/`: the scripts and results behind every number in the manuscript.
- Re-verified truth files in `data/corpus/truth/`.
- This guide.

No program logic in `backend/` was changed. For publication, a real property ID, a permit number and a planning-district name copied from the drawings were replaced with made-up values in three backend comments and two test files.

## Sheet names in the paper

The manuscript anonymises the plans as sheets S1–S8. Result files and CSV `plan` columns use the corpus IDs (there is no PLAN3):

| Paper | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|---|---|---|---|---|---|---|---|
| Files | PLAN1 | PLAN2 | PLAN4 | PLAN5 | PLAN6 | PLAN7 | PLAN8 | PLAN9 |

Site and plot numbers copied from the drawings into free-text explanations in the saved outputs have been replaced with `XX`; they were never used to identify results.

## What is where

| Manuscript item | Evidence |
|---|---|
| Table 2 (corpus, truth tiers) | `data/corpus/truth/*.json`, `experiments/results/fresh/dataset.csv`, `experiments/results/fresh/dxf_inventory.csv` |
| Truth re-verification (58 transferred values) | `experiments/results/legacy_verification.csv`, `experiments/results/LEGACY_TRUTH_VERIFICATION.md` |
| Table 1 (reconciliation probe) | `experiments/results/fresh/reconciliation_probes.csv` |
| Table 3 (extraction) | `experiments/results/fresh/pipeline_comparison.csv`, `experiments/results/fresh/field_level.csv` |
| Table 5 (rule-level outcomes) | `experiments/results/fresh/compliance_summary.csv`, `experiments/results/fresh/compliance_rule_level.csv` |
| Table 6 (failure modes) | `experiments/results/fresh/variants/*/*.meta.json`, `compliance_rule_level.csv` |
| Threshold margins | `experiments/results/fresh/threshold_margin.csv` |
| Significance tests | `experiments/results/fresh/stats_tests.json` (script: `experiments/stats_tests.py`) |
| VLM run (28 Sep 2026) | `experiments/results/fresh/variants/pdf_hybrid_vlm_api/`, which holds `*.vision.json` (raw VLM output with grounding flags), `*.plan.json`, `*.meta.json` and `vlm_summary.json` |
| VLM re-run (30 Sep 2026), including its replay | `experiments/results/vlm_rerun_2026-09-30/` |
| Partial repeat (28 Sep 2026, stopped at quota) | `experiments/results/vlm_repeat_partial_2026-09-28/` |
| Table 4 (offline replay) | `experiments/results/replay/<variant>/`, `experiments/results/replay/compliance_summary_with_replay.csv` |

## Commands

Set up the environment:

```bash
pip install -r requirements.txt
```

Run the eight VLM-off configurations. This regenerates the extraction and compliance tables:

```bash
bash experiments/run_all_variants.sh
python experiments/analyze_extraction.py
python experiments/compliance_eval.py
python experiments/stats_tests.py
```

Run the VLM. This needs a Groq key and uses about 170k tokens for all 8 plans:

```bash
export VISION_API_KEY=<key>
export VISION_API_BASE_URL=https://api.groq.com/openai/v1
python experiments/run_vlm_eval.py --backend api --model qwen/qwen3.8-27b
```

If the rasterised sheet PLAN5 hits the default 180 s extraction timeout, re-run just that plan with a longer timeout:

```bash
EXTRACTION_TIMEOUT_SECONDS=420 VISION_ENABLED=true VISION_BACKEND=api \
  VISION_API_MODEL=qwen/qwen3.8-27b \
  python experiments/run_vlm_variant.py PLAN5 experiments/results/fresh/variants/pdf_hybrid_vlm_api
```

Run the offline replay (Table 4). It makes no API calls and uses the saved VLM outputs:

```bash
for v in replay tol5 grounded_only cv_priority vlm_only_low no_region_floor; do
  for p in PLAN1 PLAN2 PLAN4 PLAN5 PLAN6 PLAN7 PLAN8 PLAN9; do
    python experiments/replay_vlm.py $v $p experiments/results/replay/$v
  done
done
python experiments/score_dirs.py experiments/results/replay/*
```

## VLM safeguards built into the scripts

- **Prompt decontamination.** The shipped prompts contain example values (9.14, 8.22, 222.83) that equal development truth. Before any model call, these are replaced in memory with 5.27, 4.87 and 317.46; `run_vlm_variant.py` asserts the swap succeeded and records it in every `meta.json`.
- **Validity check.** The pipeline silently falls back to CV when the VLM fails. A run counts only if the VLM answered on every page (`vlm_valid=true`); invalid runs are excluded from scoring.

## Re-run result (30 Sep 2026)

A full re-run with a new API key reproduced the 28 Sep run exactly. All 84 scored fields got the same status, confidence and value, and all rule-level outcomes matched: 17 correct, 14 insufficient data, 1 conflict, and no false verdicts.

The raw VLM outputs did differ slightly between the two runs:

| | 28 Sep run | 30 Sep re-run |
|---|---|---|
| Items returned | 122 | 125 |
| Grounded items | 67 | 70 |

PLAN5 needed a 420 s extraction timeout in the re-run. Replaying the re-run's outputs with the same five fusion variants reproduced every row of Table 4 exactly (`experiments/results/vlm_rerun_2026-09-30/replay/`).

The earlier partial repeat on 28 Sep matched 63 of 75 outcomes on six plans. So the VLM path is not fully deterministic even at temperature 0.

## Privacy

The original permit drawings (`data/test_plans/`, and the drawing copies under `data/corpus/corrections/`) include applicant and engineer names and are not distributed. They are available from the authors on request, subject to the owners' permission. Without them, the extraction runs above cannot be repeated, but every saved output in `experiments/results/` can be re-scored and re-analysed. The legacy truth spreadsheet (`GROUND_TRUTH_2.xlsx`, named as a source in the truth files) is withheld for the same reason: it lists the plots' addresses.
