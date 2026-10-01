# PMC_DEMO -- illustrative second ruleset

**This is NOT a verified transcription of any real Pune Municipal
Corporation bye-law.** It exists to demonstrate that the deterministic
rule engine (`backend.compliance.interfaces.JsonFileRuleEngine`) is
genuinely multi-municipality: swapping which ruleset a plan is checked
against is a `municipality` string argument, not a code change, and this
second, structurally-independent JSON file is the demonstration of that,
the same way `data/runtime_rules/BBMP/rules.json` is the real one.

Every rule here follows the exact same schema as the BBMP ruleset
(`RuntimeRuleDefinition` / RASE `applies_when` + `threshold` conditions --
see `backend/rase/schema.py`) and covers the same rule categories
(setback-by-plot-dimension bands, coverage-by-plot-area bands, FAR, and a
high-rise minimum road width rule) with simplified, round-number
thresholds, so it exercises the same code paths as BBMP without
duplicating BBMP's own numbers.

Before this could back a real compliance check for Pune (or any other
municipality), it would need the same treatment `BBMP/README.md` and
`BBMP/RULESET_AUDIT.md` document for the BBMP ruleset: transcription from
the actual gazetted bye-laws, page/clause citations, and an audit pass.
Until then, treat any PASS/FAIL produced against `PMC_DEMO` as a
demonstration of the mechanism, not a real compliance verdict.
