# BBMP deterministic runtime rules

This directory is the **authoritative runtime compliance ruleset** for BBMP.
Runtime compliance does **not** generate rules from RAG text and does **not** call an LLM.

## Sources represented

### ACTIVE — `bangalore_dcr.pdf`

The bundled Bangalore Mahanagara Palike Building Bye-Laws 2003 came into operation
on 5 June 2004. The runtime rules transcribe the clauses that can be evaluated from
the current `NormalizedPlan` contract:

- Table 4 / Bye-law 9.2: front/rear setbacks by site depth and left/right setbacks
  by site width for Residential, Commercial and the combined T&T/P.U./Public &
  Semi-Public category.
- Table 4 Note 3(c): for buildings above 9.5 m, Table 4 and Table 5 are both
  evaluated; the higher applicable setback therefore controls deterministically.
- Table 5 / Bye-law 9.2: all-round setbacks for height bands above 9.5 m.
- Bye-law 9.3.2: high-rise minimum site width and depth of 21 m.
- Bye-law 9.3.3: high-rise minimum road width of 12 m.
- Table 6 / Bye-laws 9.2 and 9.10: coverage and FAR for Residential and Commercial
  buildings by development area A/B/C and plot-area band; and the combined
  Public/Semi-Public/T&T/Public Utility category with its printed road-width bands.
- Bye-law 9.10.2: special maximum coverage for hospital, health centre/nursing home,
  nursery/primary school, secondary school and college.

### DRAFT — `Revised setback gazette copy.pdf`

The 11-Nov-2025 notification is stored as **DRAFT** rules only. Because the supplied PDF is image-only, an OCR companion `Revised setback gazette copy_ocr.txt` is also included for RAG ingestion. Its Table 8 values
are present in `rules.json` for provenance/review, but `JsonFileRuleEngine` ignores
all DRAFT rules. They cannot affect a compliance result until a final applicable
notification is formally promoted to ACTIVE.

The draft Table 8 rules represented are:

- up to 60 sq.m: 0.75 m front and 0.60 m on any one side;
- above 60 to 150 sq.m: 0.90 m front, 0.70 m rear, 0.70 m on any one side;
- above 150 to 4000 sq.m: 12% of site depth front, 8% of site depth rear, and
  8% of site width on each side;
- above 4000 sq.m: 5.0 m minimum on all sides.

### `10963 Greater Bengaluru Area (Parking)Rules, 2026..pdf`

This is a draft parking notification. No runtime parking rule is generated from it
yet because the current `NormalizedPlan` has no parking/driveway/mechanical-parking
measurements. Mapping its driveway requirements onto `road.width` would be legally
incorrect.

## Required regulatory context

The engine never guesses regulatory classification. For rules that require it, the
plan must provide:

- `building_use` — for example `residential`, `commercial`, `public_semi_public`,
  `traffic_transportation`, `public_utility`, or the named special uses;
- `development_area` — `A`, `B`, or `C` for Table 6;
- `building_height_estimated` where height-tiered rules apply;
- `building.floor_count` for the 2003 high-rise definition (ground floor plus four
  or more floors above = 5 or more total floors).

Missing context produces `INSUFFICIENT_DATA`; it is never inferred by an LLM.

## Runtime architecture

```text
NormalizedPlan
      |
      v
JsonFileRuleEngine
      |
      +--> select ACTIVE rules whose applicability is determinable
      |
      v
DeterministicRuleEvaluator
      |
      +--> PASS
      +--> FAIL
      +--> NOT_APPLICABLE
      +--> INSUFFICIENT_DATA
      +--> CONFLICTING_EVIDENCE
      +--> REQUIRES_REVIEW
```
