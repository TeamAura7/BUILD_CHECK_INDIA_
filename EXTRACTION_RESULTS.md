# BUILDCheck India — extraction results by evidence source

All 8 corpus plans (the corpus has no PLAN3), PDF and DXF, code snapshot 699fb74.

| Column | What it is | Run |
|---|---|---|
| **CV** | Independent CV layer of the PDF path before fusion: native text, vector geometry, OCR and OpenCV raster evidence through the site-plan resolver | fresh re-run |
| **PDF** | Final PDF pipeline output, vision off (CV + pipeline reasoning such as floor count, building use, derived coverage/FAR) | fresh re-run |
| **VLM** | Vision only: saved Qwen3.8-27B output passed through `final_fusion` with no CV, i.e. what vision alone would ship (grounding caps applied) | 30 Sep 2026 VLM run |
| **FUSION** | PDF pipeline with vision on: CV and VLM reconciled by `final_fusion` | 30 Sep 2026 VLM run |
| **DXF** | DXF pipeline alone | fresh re-run |
| **PDF+DXF** | `pdf_dxf_reconciliation` of the PDF and DXF plans (PDF value ships; DXF disagreement only adds a flag, shown as ⚑) | fresh re-run |

Cell format: `value (level) mark`. Levels: H high, M medium, L low; `withheld (C)` = conflicting evidence, no value shipped; — = no value.
Marks against the re-verified truth, tolerance max(0.15, 5 %): ✓ correct, ✗ wrong, miss = no answer, ⊘ answered a must-abstain field, ✓abst = correctly abstained, ↔ correct as an unordered width/depth pair (axes swapped), · truth unverified/absent (not scored).
Units: lengths m, areas m², coverage %.

## Summary over all plans (84 scored truth values, 10 must-abstain fields)

| Source | Correct | Wrong | Missing | Withheld (conflict) | Answered must-abstain | Correctly abstained |
|---|---|---|---|---|---|---|
| CV | 61 | 2 | 21 | 0 | 0 | 10 |
| PDF | 78 | 2 | 4 | 0 | 4 | 6 |
| VLM | 19 | 10 | 55 | 0 | 2 | 8 |
| FUSION | 56 | 4 | 11 | 13 | 5 | 5 |
| DXF | 6 | 46 | 32 | 0 | 0 | 10 |
| PDF+DXF | 78 | 2 | 4 | 0 | 4 | 6 |

Notes:
- CV and VLM are intermediate layers; they are not what the system ships. PDF, FUSION, DXF and PDF+DXF are shipped outputs.
- VLM covers only the fields `final_fusion` takes from vision (lengths, areas, coverage, FAR); floor count and building use come from the pipeline, so they are blank in that column.
- CV does not produce floor count, building use or derived coverage/FAR on its own; the PDF pipeline adds these.
- Level shown for PDF+DXF is the PDF value's level after reconciliation (agreement can promote it to H; disagreement leaves it unchanged and adds ⚑).

Run notes (30 Sep 2026):
- The PDF and DXF pipelines were re-run today for all 8 plans (VLM off, default settings, fresh processes).
- **PDF:** identical to the paper's run on all 84 values (status, confidence and value).
- **DXF:** 6 correct / 46 wrong today versus 5 / 47 in the paper. On PLAN6 (S5) the DXF plot outline changed between runs (plot width 16.96 m → 12.78 m, area 78.05 → 58.80 m²). The width now scores correct only as an unordered width/depth pair (12.78 m is within tolerance of the 13.10 m depth). The DXF reconstruction is therefore not fully deterministic. All DXF values, including the new one, remain LOW, so rule-level containment is unchanged.
- **VLM and FUSION:** taken from the 30 Sep VLM re-run, which reproduced the paper's VLM run on all 84 values. No new API calls were made for this report.

## PLAN1 (paper sheet S1)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 12.19 (printed) | 12.19 (H) ✓ | 12.19 (H) ✓ | 12.19 (H) ✓ | — miss | 12.74 (L) ✓ | 12.19 (H) ✓ |
| `plot.depth` | 9.14 (printed) | 9.14 (H) ✓ | 9.14 (H) ✓ | 11.72 (L) ✗ | withheld (C) | 7.73 (L) ✗ | 9.14 (H) ⚑ ✓ |
| `plot.area` | 111.42 (derived) | — miss | 111.42 (H) ✓ | — miss | — miss | 98.54 (L) ✗ | 111.42 (H) ⚑ ✓ |
| `building.width` | 11.72 (printed) | 11.70 (H) ✓ | 11.70 (H) ✓ | 11.72 (L) ✓ | — miss | 10.17 (L) ✗ | 11.70 (H) ⚑ ✓ |
| `building.depth` | 8.22 (printed) | 8.20 (H) ✓ | 8.20 (H) ✓ | 8.22 (H) ✓ | — miss | 6.39 (L) ✗ | 8.20 (H) ⚑ ✓ |
| `building.footprint_area` | 96.34 (printed) | 96.34 (H) ✓ | 96.34 (H) ✓ | 96.34 (H) ✓ | withheld (C) | 65.03 (L) ✗ | 96.34 (H) ⚑ ✓ |
| `building.floor_count` | 4 (printed) | — miss | 4 (M) ✓ | — miss | 4 (M) ✓ | — miss | 4 (M) ✓ |
| `road.width` | must abstain | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst |
| `setbacks.front` | 0.47 (printed) | 0.47 (H) ✓ | 0.47 (H) ✓ | — miss | — miss | 0.00 (L) ✗ | 0.47 (H) ⚑ ✓ |
| `setbacks.rear` | 0.00 (inspected) | 0.00 (H) ✓ | 0.00 (H) ✓ | — miss | — miss | 1.34 (L) ✗ | 0.00 (H) ⚑ ✓ |
| `setbacks.left` | 0.46 (printed) | 0.46 (H) ✓ | 0.46 (H) ✓ | — miss | — miss | 2.57 (L) ✗ | 0.46 (H) ⚑ ✓ |
| `setbacks.right` | 0.46 (printed) | 0.46 (H) ✓ | 0.46 (H) ✓ | — miss | — miss | 0.00 (L) ✗ | 0.46 (H) ⚑ ✓ |
| `coverage` | 86.47 (derived) | — miss | 86.47 (H) ✓ | — miss | — miss | 65.99 (L) ✗ | 86.47 (H) ⚑ ✓ |
| `far` | 3.459 (derived) | — miss | 3.459 (M) ✓ | — miss | — miss | 0.660 (L) ✗ | 3.459 (M) ⚑ ✓ |
| `building_use` | residential (printed) | — miss | residential (H) ✓ | — miss | residential (H) ✓ | — miss | residential (H) ✓ |

## PLAN2 (paper sheet S2)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 12.19 (printed) | 12.19 (H) ✓ | 12.19 (H) ✓ | 12.19 (L) ✓ | 12.19 (H) ✓ | 11.04 (L) ✗ | 12.19 (H) ⚑ ✓ |
| `plot.depth` | 18.28 (printed) | 18.29 (H) ✓ | 18.29 (H) ✓ | — miss | 18.29 (H) ✓ | 15.60 (L) ✗ | 18.29 (H) ⚑ ✓ |
| `plot.area` | 222.83 (printed) | 222.83 (H) ✓ | 222.83 (H) ✓ | — miss | 222.83 (H) ✓ | 172.25 (L) ✗ | 222.83 (H) ⚑ ✓ |
| `building.width` | 10.59 (printed) | 10.59 (H) ✓ | 10.59 (H) ✓ | 12.19 (L) ✗ | withheld (C) | 10.17 (L) ✓ | 10.59 (H) ✓ |
| `building.depth` | 16.48 (printed) | 16.47 (H) ✓ | 16.47 (H) ✓ | 12.19 (L) ✗ | withheld (C) | 14.49 (L) ✗ | 16.47 (H) ⚑ ✓ |
| `building.footprint_area` | 174.52 (printed) | 174.52 (H) ✓ | 174.52 (H) ✓ | 77.62 (H) ✗ | withheld (C) | 122.41 (L) ✗ | 174.52 (H) ⚑ ✓ |
| `building.floor_count` | 4 (inspected) | — miss | 4 (M) ✓ | — miss | 4 (M) ✓ | — miss | 4 (M) ✓ |
| `road.width` | 9.20 (printed) | 9.20 (H) ✓ | 9.20 (H) ✓ | 9.20 (H) ✓ | 9.20 (H) ✓ | — miss | 9.20 (H) ✓ |
| `setbacks.front` | 1.00 (printed) | 1.00 (H) ✓ | 1.00 (H) ✓ | — miss | 1.00 (H) ✓ | 0.21 (L) ✗ | 1.00 (H) ⚑ ✓ |
| `setbacks.rear` | 0.80 (printed) | 0.80 (H) ✓ | 0.80 (H) ✓ | — miss | 0.80 (H) ✓ | 0.41 (L) ✗ | 0.80 (H) ⚑ ✓ |
| `setbacks.left` | 0.80 (printed) | 0.80 (H) ✓ | 0.80 (H) ✓ | — miss | 0.80 (H) ✓ | 0.44 (L) ✗ | 0.80 (H) ⚑ ✓ |
| `setbacks.right` | 0.80 (printed) | 0.80 (H) ✓ | 0.80 (H) ✓ | — miss | 0.80 (H) ✓ | 0.60 (L) ✗ | 0.80 (H) ⚑ ✓ |
| `coverage` | 78.32 (printed) | 78.32 (H) ✓ | 78.32 (H) ✓ | — miss | 78.32 (H) ✓ | 71.06 (L) ✗ | 78.32 (H) ⚑ ✓ |
| `far` | 1.730 (printed) | 1.730 (H) ✓ | 1.730 (H) ✓ | — miss | 1.730 (H) ✓ | 0.711 (L) ✗ | 1.730 (H) ⚑ ✓ |
| `building_use` | — | — | residential (H) | — | residential (H) | — | residential (H) |

## PLAN4 (paper sheet S3)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 18.29 (printed) | 18.29 (H) ✓ | 18.29 (H) ✓ | — miss | 18.29 (H) ✓ | 4.11 (L) ✗ | 18.29 (H) ⚑ ✓ |
| `plot.depth` | 16.61 (printed) | 16.61 (H) ✓ | 16.61 (H) ✓ | 20.00 (H) ✗ | withheld (C) | 7.54 (L) ✗ | 16.61 (H) ⚑ ✓ |
| `plot.area` | 303.79 (printed) | 303.79 (H) ✓ | 303.79 (H) ✓ | 303.79 (H) ✓ | 303.79 (H) ✓ | 303.79 (H) ✓ | 303.79 (H) ✓ |
| `building.width` | 16.46 (printed) | 16.46 (H) ✓ | 16.46 (H) ✓ | — miss | 16.46 (H) ✓ | — miss | 16.46 (H) ✓ |
| `building.depth` | 11.28 (printed) | 11.28 (H) ✓ | 11.28 (H) ✓ | — miss | 11.28 (H) ✓ | — miss | 11.28 (H) ✓ |
| `building.footprint_area` | 185.62 (printed) | 185.62 (H) ✓ | 185.62 (H) ✓ | 185.62 (H) ✓ | 185.62 (H) ✓ | — miss | 185.62 (H) ✓ |
| `building.floor_count` | 5 (inspected) | — miss | 5 (L) ✓ | — miss | 5 (L) ✓ | — miss | 5 (L) ✓ |
| `road.width` | 7.62 (printed) | 7.62 (H) ✓ | 7.62 (H) ✓ | — miss | 7.62 (H) ✓ | 7.62 (H) ✓ | 7.62 (H) ✓ |
| `setbacks.front` | 0.91 (printed) | 0.91 (H) ✓ | 0.91 (H) ✓ | — miss | 0.91 (H) ✓ | — miss | 0.91 (H) ✓ |
| `setbacks.rear` | 0.91 (printed) | 0.91 (H) ✓ | 0.91 (H) ✓ | — miss | 0.91 (H) ✓ | — miss | 0.91 (H) ✓ |
| `setbacks.left` | 2.67 (printed) | 2.66 (H) ✓ | 2.66 (H) ✓ | — miss | 2.66 (H) ✓ | — miss | 2.66 (H) ✓ |
| `setbacks.right` | 2.67 (printed) | 2.67 (H) ✓ | 2.67 (H) ✓ | — miss | 2.67 (H) ✓ | — miss | 2.67 (H) ✓ |
| `coverage` | 61.10 (derived) | — miss | 61.10 (H) ✓ | — miss | 61.10 (H) ✓ | — miss | 61.10 (H) ✓ |
| `far` | 0.611 (unverified) | — | 3.055 (L) | — | 3.055 (L) | — | 3.055 (L) |
| `building_use` | — | — | — | — | — | — | — |

## PLAN5 (paper sheet S4)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 17.59 (printed) | 17.59 (H) ✓ | 17.59 (H) ✓ | 17.09 (L) ✓ | withheld (C) | 4.74 (L) ✗ | 17.59 (H) ⚑ ✓ |
| `plot.depth` | 9.14 (printed) | 9.21 (H) ✓ | 9.21 (H) ✓ | 11.14 (L) ✗ | withheld (C) | 3.50 (L) ✗ | 9.21 (H) ⚑ ✓ |
| `plot.area` | 160.77 (printed) | 160.77 (H) ✓ | 160.77 (H) ✓ | 189.00 (L) ✗ | 160.77 (M) ✓ | 16.59 (L) ✗ | 160.77 (H) ⚑ ✓ |
| `building.width` | 13.09 (printed) | 13.09 (H) ✓ | 13.09 (H) ✓ | 13.00 (L) ✓ | 13.04 (H) ✓ | — miss | 13.09 (H) ✓ |
| `building.depth` | 7.14 (printed) | 7.14 (H) ✓ | 7.14 (H) ✓ | 7.14 (L) ✓ | 7.14 (H) ✓ | — miss | 7.14 (H) ✓ |
| `building.footprint_area` | 93.46 (printed) | 93.46 (H) ✓ | 93.46 (H) ✓ | — miss | 93.46 (H) ✓ | — miss | 93.46 (H) ✓ |
| `building.floor_count` | 3 (printed) | — miss | 3 (M) ✓ | — miss | 3 (M) ✓ | — miss | 3 (M) ✓ |
| `road.width` | 10.00 (printed) | 10.00 (H) ✓ | 10.00 (H) ✓ | 3.00 (L) ✗ | withheld (C) | — miss | 10.00 (H) ✓ |
| `setbacks.front` | 3.00 (printed) | 3.00 (H) ✓ | 3.00 (H) ✓ | 1.50 (L) ✗ | withheld (C) | — miss | 3.00 (H) ✓ |
| `setbacks.rear` | 1.50 (printed) | 1.50 (H) ✓ | 1.50 (H) ✓ | 4.00 (L) ✗ | withheld (C) | — miss | 1.50 (H) ✓ |
| `setbacks.left` | 1.00 (printed) | 1.07 (H) ✓ | 1.07 (H) ✓ | 1.00 (L) ✓ | withheld (C) | — miss | 1.07 (H) ✓ |
| `setbacks.right` | 1.00 (printed) | 1.00 (H) ✓ | 1.00 (H) ✓ | 1.00 (L) ✓ | 1.00 (H) ✓ | — miss | 1.00 (H) ✓ |
| `coverage` | 58.13 (printed) | — miss | 58.13 (H) ✓ | — miss | 58.13 (M) ✓ | — miss | 58.13 (H) ✓ |
| `far` | — | — | 1.744 (M) | 1.000 (L) | 1.000 (L) | — | 1.744 (M) |
| `building_use` | residential (printed) | — miss | residential (H) ✓ | — miss | residential (H) ✓ | — miss | residential (H) ✓ |

## PLAN6 (paper sheet S5)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 10.00 (inspected) | 10.00 (H) ✓ | 10.00 (H) ✓ | 10.00 (L) ✓ | 10.00 (H) ✓ | 12.78 (L) ✓↔ | 10.00 (H) ⚑ ✓ |
| `plot.depth` | 13.10 (printed) | 13.10 (H) ✓ | 13.10 (H) ✓ | — miss | 13.10 (H) ✓ | 4.60 (L) ✗↔ | 13.10 (H) ⚑ ✓ |
| `plot.area` | 131.00 (printed) | 131.00 (H) ✓ | 131.00 (H) ✓ | — miss | 131.00 (H) ✓ | 58.80 (L) ✗ | 131.00 (H) ⚑ ✓ |
| `building.width` | 9.30 (unverified) | 9.31 (H) | 9.31 (H) | 10.00 (L) | withheld (C) · | 3.66 (L) | 9.31 (H) ⚑ |
| `building.depth` | 9.15 (unverified) | 9.16 (H) | 9.16 (H) | — | 9.16 (H) | 3.60 (L) | 9.16 (H) ⚑ |
| `building.footprint_area` | 85.09 (printed) | 85.09 (H) ✓ | 85.09 (H) ✓ | — miss | 85.09 (H) ✓ | 13.18 (L) ✗ | 85.09 (H) ⚑ ✓ |
| `building.floor_count` | 3 (inspected) | — miss | 3 (M) ✓ | — miss | 3 (M) ✓ | — miss | 3 (M) ✓ |
| `road.width` | 7.30 (printed) | 7.30 (H) ✓ | 7.30 (H) ✓ | 7.30 (H) ✓ | 7.30 (H) ✓ | — miss | 7.30 (H) ✓ |
| `setbacks.front` | 0.90 (unverified) | 3.25 (H) | 3.25 (H) | — | 3.25 (H) | 0.61 (L) | 3.25 (H) ⚑ |
| `setbacks.rear` | 0.70 (unverified) | 0.70 (H) | 0.70 (H) | — | 0.70 (H) | 4.28 (L) | 0.70 (H) ⚑ |
| `setbacks.left` | 0.70 (unverified) | 0.00 (H) | 0.00 (H) | — | 0.00 (H) | 0.37 (L) | 0.00 (H) ⚑ |
| `setbacks.right` | — | 0.69 (H) | 0.69 (H) | — | 0.69 (H) | — | 0.69 (H) |
| `coverage` | 79.15 (printed) | 79.15 (H) ✓ | 79.15 (H) ✓ | — miss | 79.15 (H) ✓ | 22.42 (L) ✗ | 79.15 (H) ⚑ ✓ |
| `far` | 1.740 (printed) | 1.740 (H) ✓ | 1.740 (H) ✓ | — miss | 1.740 (H) ✓ | 0.224 (L) ✗ | 1.740 (H) ⚑ ✓ |
| `building_use` | — | — | residential (H) | — | residential (H) | — | residential (H) |

## PLAN7 (paper sheet S6)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | must abstain | — ✓abst | 19.37 (L) ⊘ | — ✓abst | 19.37 (L) ⊘ | — ✓abst | 19.37 (L) ⊘ |
| `plot.depth` | must abstain | — ✓abst | 4.40 (L) ⊘ | — ✓abst | 4.40 (L) ⊘ | — ✓abst | 4.40 (L) ⊘ |
| `plot.area` | 111.41 (unverified) | — | 85.29 (L) | — | 85.29 (L) | — | 85.29 (L) |
| `building.width` | must abstain | — ✓abst | 5.82 (L) ⊘ | 9.14 (H) ⊘ | 9.14 (H) ⊘ | — ✓abst | 5.82 (L) ⊘ |
| `building.depth` | must abstain | — ✓abst | 13.83 (L) ⊘ | — ✓abst | 13.83 (L) ⊘ | — ✓abst | 13.83 (L) ⊘ |
| `building.footprint_area` | 69.10 (unverified) | — | 80.49 (L) | — | withheld (C) · | — | 80.49 (L) |
| `building.floor_count` | 3 (inspected) | — miss | — miss | — miss | 4 (M) ✗ | — miss | — miss |
| `road.width` | 9.14 (printed) | — miss | — miss | 9.14 (H) ✓ | 9.14 (H) ✓ | — miss | — miss |
| `setbacks.front` | must abstain | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst |
| `setbacks.rear` | must abstain | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst |
| `setbacks.left` | must abstain | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst |
| `setbacks.right` | must abstain | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst | — ✓abst |
| `coverage` | — | — | 94.38 (L) | — | — | — | 94.38 (L) |
| `far` | — | — | 0.944 (L) | — | — | — | 0.944 (L) |
| `building_use` | residential (inspected) | — miss | — miss | — miss | — miss | — miss | — miss |

## PLAN8 (paper sheet S7)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 13.71 (printed) | 11.28 (L) ✗ | 11.28 (L) ✗ | — miss | 11.28 (L) ✗ | 7.11 (L) ✗ | 11.28 (L) ⚑ ✗ |
| `plot.depth` | 13.71 (printed) | 16.50 (L) ✗ | 16.50 (L) ✗ | — miss | 16.50 (L) ✗ | 11.89 (L) ✗ | 16.50 (L) ⚑ ✗ |
| `plot.area` | 183.69 (printed) | 183.69 (H) ✓ | 183.69 (H) ✓ | — miss | 183.69 (H) ✓ | 84.55 (L) ✗ | 183.69 (H) ⚑ ✓ |
| `building.width` | 10.95 (unverified) | 9.55 (L) | 9.55 (L) | — | 9.55 (L) | 4.31 (L) | 9.55 (L) ⚑ |
| `building.depth` | 10.70 (unverified) | 11.99 (L) | 11.99 (L) | — | 11.99 (L) | 4.19 (L) | 11.99 (L) ⚑ |
| `building.footprint_area` | 115.42 (printed) | 115.42 (H) ✓ | 115.42 (H) ✓ | — miss | 115.42 (H) ✓ | 17.88 (L) ✗ | 115.42 (H) ⚑ ✓ |
| `building.floor_count` | 3 (printed) | — miss | 3 (M) ✓ | — miss | 3 (M) ✓ | — miss | 3 (M) ✓ |
| `road.width` | 7.60 (unverified) | — | — | 7.60 (H) | 7.60 (H) | — | — |
| `setbacks.front` | 1.65 (unverified) | 0.00 (L) | 0.00 (L) | 1.00 (H) | withheld (C) · | 1.16 (L) | 0.00 (L) ⚑ |
| `setbacks.rear` | 1.36 (unverified) | 1.74 (L) | 1.74 (L) | 1.00 (H) | withheld (C) · | 6.39 (L) | 1.74 (L) ⚑ |
| `setbacks.left` | 1.66 (unverified) | 4.51 (L) | 4.51 (L) | 1.00 (H) | withheld (C) · | 0.24 (L) | 4.51 (L) ⚑ |
| `setbacks.right` | 1.10 (unverified) | 0.00 (L) | 0.00 (L) | 1.00 (H) | withheld (C) · | 2.42 (L) | 0.00 (L) ⚑ |
| `coverage` | — | 62.83 (H) | 62.83 (H) | 100.00 (H) | withheld (C) · | 62.83 (H) | 62.83 (H) |
| `far` | — | 1.640 (H) | 1.640 (H) | — | 1.640 (H) | 1.640 (H) | 1.640 (H) |
| `building_use` | — | — | residential (H) | — | residential (H) | — | residential (H) |

## PLAN9 (paper sheet S8)

| Field | Truth (tier) | CV | PDF | VLM | FUSION | DXF | PDF+DXF |
|---|---|---|---|---|---|---|---|
| `plot.width` | 10.65 (printed) | 10.65 (H) ✓ | 10.65 (H) ✓ | 10.65 (H) ✓ | 10.65 (H) ✓ | 6.59 (L) ✗ | 10.65 (H) ⚑ ✓ |
| `plot.depth` | 15.25 (printed) | 15.21 (H) ✓ | 15.21 (H) ✓ | 15.00 (H) ✓ | withheld (C) | 15.51 (L) ✓ | 15.21 (H) ✓ |
| `plot.area` | 147.99 (printed) | — miss | 147.99 (M) ✓ | 147.99 (H) ✓ | 147.99 (H) ✓ | 81.95 (L) ✗ | 147.99 (M) ⚑ ✓ |
| `building.width` | 7.14 (printed) | 7.14 (H) ✓ | 7.14 (H) ✓ | — miss | 7.14 (H) ✓ | 6.38 (L) ✗ | 7.14 (H) ⚑ ✓ |
| `building.depth` | 12.11 (printed) | 12.10 (H) ✓ | 12.10 (H) ✓ | — miss | 12.10 (H) ✓ | 4.94 (L) ✗ | 12.10 (H) ⚑ ✓ |
| `building.footprint_area` | 94.42 (printed) | — miss | 94.42 (M) ✓ | — miss | 94.42 (M) ✓ | 31.50 (L) ✗ | 94.42 (M) ⚑ ✓ |
| `building.floor_count` | 3 (inspected) | — miss | — miss | — miss | 1 (M) ✗ | — miss | — miss |
| `road.width` | must abstain | — ✓abst | — ✓abst | 8.92 (H) ⊘ | 8.92 (H) ⊘ | — ✓abst | — ✓abst |
| `setbacks.front` | 1.50 (printed) | 1.50 (H) ✓ | 1.50 (H) ✓ | — miss | 1.50 (H) ✓ | 3.80 (L) ✗ | 1.50 (H) ⚑ ✓ |
| `setbacks.rear` | 1.50 (printed) | 1.38 (H) ✓ | 1.38 (H) ✓ | — miss | 1.38 (H) ✓ | 5.76 (L) ✗ | 1.38 (H) ⚑ ✓ |
| `setbacks.left` | 1.00 (printed) | 1.00 (H) ✓ | 1.00 (H) ✓ | — miss | 1.00 (H) ✓ | 0.24 (L) ✗ | 1.00 (H) ⚑ ✓ |
| `setbacks.right` | 1.00 (printed) | 1.00 (H) ✓ | 1.00 (H) ✓ | — miss | 1.00 (H) ✓ | 0.00 (L) ✗ | 1.00 (H) ⚑ ✓ |
| `coverage` | 63.80 (derived) | — miss | 63.80 (M) ✓ | — miss | 63.80 (M) ✓ | 38.44 (L) ✗ | 63.80 (M) ⚑ ✓ |
| `far` | — | — | 0.638 (L) | — | 0.638 (M) | 0.384 (L) | 0.638 (L) ⚑ |
| `building_use` | residential (printed) | — miss | residential (H) ✓ | — miss | residential (H) ✓ | — miss | residential (H) ✓ |
