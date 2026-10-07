# Verification of the 58 "legacy" truth values

## What "legacy" meant

In `data/corpus/truth/*.json`, 58 of the 84 scored truth values had `"verification": "legacy"`. They were copied from an older answer file, either `data/test_plans/PLANx.expected.json` or `GROUND_TRUTH_2.xlsx`, and were never checked again against the drawings. The breakdown by plan was:

| Plan | Legacy values |
|---|---|
| PLAN2 | 14 |
| PLAN5 | 14 |
| PLAN4 | 13 |
| PLAN6 | 8 |
| PLAN7 | 3 |
| PLAN8 | 3 |
| PLAN9 | 3 |

## Was it bad for the system?

No. It was bad for the **evaluation**, not for the system. The label says nothing about how the pipeline behaves. It only says that the answer key had unknown provenance. A reviewer could argue that an answer key assembled during development might partly mirror tool output, and at 58 of 84 values (69 %) that argument would have undermined every accuracy number in the paper.

## What was done

Every value was traced to a location on the sheet:

- **Vector PDFs:** the native text layer was searched, and crops were checked visually.
- **PLAN5 (raster):** each value was read visually from high-resolution crops.

The results are in `legacy_verification.csv`, which has one row per value with the sheet location. `proposed_truth/` holds updated truth files with new tiers. The old value is kept as `legacy_value`, and the evidence text is updated.

| New tier | Count | Meaning |
|---|---|---|
| printed | 50 | The number or text is printed on the sheet. PLAN4's 9 values are feet-inch dimensions converted to metres. |
| derived | 1 | PLAN4 coverage, 185.62 / 303.79 = 61.10 %. It is not printed. |
| inspected | 7 | Floor counts read from floor-plan titles, PLAN7 building use inferred from room names, and PLAN6 plot width (10.00 is printed only inside a road-widening note, but 10.00 × 13.10 = 131.00 matches the printed area). |

Two values were wrong and have been corrected, both on PLAN5:

- **Footprint:** 93.43 → **93.46 m²**. The area statement prints "GROUND COVERAGE AREA 93.46", and 13.09 × 7.14 = 93.46.
- **Coverage:** 58.11 → **58.13 %**. The statement prints "Coverage 58.13 %".

**Effect on results: none.** Both differences are far inside the scoring tolerance. The changed rows were re-scored with the repository rule |ŷ − y| ≤ max(0.15, 0.05|y|), and 0 of 10 field statuses changed. MAE (length fields only) and MdAPE are unchanged, and so is the one compliance check that uses these values (coverage ≤ 65 %). A full re-run of `analyze_extraction.py` and `compliance_eval.py` with the proposed truth files confirmed this. `compliance_summary.csv` is byte-identical. In `pipeline_comparison.csv`, the only change is the mean APE, which is not reported in the paper: it moves by less than 0.002 percentage points.

After the fix, the scored truth tiers are 71 printed, 5 derived, 8 inspected and 0 legacy.

## Other finding: ambiguous truth on PLAN9 (reported in the paper)

The pipeline's only compliance error is on PLAN9, rear setback ≥ 1.5 m. The truth file itself says: "1.50 M at the wider end of the top gap; the top edge leans 0.7 deg so the true minimum is about 1.34 M".

- If a setback means the **minimum** distance, the true value is below 1.5 m, and the system's FAIL is arguably correct.
- The paper keeps the printed value and still counts the case as an error, but it now states the ambiguity.

## Not changed (out of scope, for a human annotator)

On PLAN6, the site plan shows 0.90 m front, 0.70 m rear and 0.70 m on the right. The truth file holds these as *unverified* values and puts the side setback on the left. Unverified values are not scored, so they were left untouched.

## Still open

This was an AI-assisted trace by one annotator. A second human check of `legacy_verification.csv` is recommended before camera-ready. To adopt the files:

1. Copy `proposed_truth/*.json` into `data/corpus/truth/`.
2. Re-run `experiments/analyze_extraction.py` and `compliance_eval.py`. The numbers should come out identical.
