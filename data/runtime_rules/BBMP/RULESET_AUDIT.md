# BBMP Runtime Rules Audit

- Total rule records: **288**
- ACTIVE: **277**
- DRAFT: **11**
- Runtime authority: **ACTIVE records in `data/runtime_rules/BBMP/rules.json` only**
- Runtime LLM rule generation: **disabled**

## Active rule families

- Table 6 — Coverage: 75 rules
- Table 6 — FAR: 75 rules
- Table 4 — depth of site: 54 rules
- Table 4 — width of site; Note 3(b): 53 rules
- Table 5 — height of building: 12 rules
- Bye-law 9.10.2: 5 rules
- Bye-law 9.3.2: 2 rules
- Bye-law 9.3.3: 1 rules

## Draft rule families

- Table 8(i), Sl. No. 3: 4 rules
- Table 8(i): 2 rules
- Table 8(i), Sl. No. 2: 2 rules
- Table 8(ii): 1 rules
- Table 8(i), Sl. No. 1: 1 rules
- Note 1 after Table 9: 1 rules

## Deterministic context fields

- `building_use`: selects Residential / Commercial / Public-Semi-Public / T&T / Public Utility or named special-use rules.
- `development_area`: A/B/C selects the Table 6 row group.
- `plot.area`: selects the Table 6 plot-area band.
- `plot.depth`: selects Table 4 front/rear setback band.
- `plot.width`: selects Table 4 side setback band.
- `building_height_estimated`: selects Table 5 height band.
- `building.floor_count`: identifies 5+ floor high-rise buildings under Bye-law 2.46.
- `building_height_excluding_stilt`: used only by the inactive 2025 draft Table 8/Note 1 rules.

## Important non-inferences

- The engine does not infer A/B/C from geometry.
- The engine does not infer building use from legal text at runtime.
- The engine does not convert a missing retrieval result into a legal rule.
- Draft regulations never affect a live result.
- A missing applicability input produces `INSUFFICIENT_DATA`, not PASS/FAIL.
