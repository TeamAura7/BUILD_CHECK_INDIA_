# Evaluation corpus

Two splits, and the difference between them is the whole point.

| split | what it is | may be used to |
|---|---|---|
| `dev` | plans the extractor was developed and tuned against (PLAN1-PLAN9) | debug, regress, tune |
| `heldout` | plans no one has looked at while building or tuning | report generalisation, once per release |

Every result on `dev` is an optimistic estimate: those plans shaped the code.
**Only `heldout` numbers may be quoted as accuracy.** There are no held-out
plans yet; the dev set cannot become one.

## Commands

```
python -m backend.tools.run_corpus validate
python -m backend.tools.run_corpus eval                      # dev, all modalities
python -m backend.tools.run_corpus add --id HO-001 --split heldout --pdf a.pdf --dxf a.dxf --source "..." --licence "..." --redacted
python -m backend.tools.run_corpus freeze                    # after annotation
python -m backend.tools.run_corpus eval --split heldout
```

## Adding a held-out plan (the rules)

1. **Choose plans blind.** Take real sanctioned plans (PDF, and the DXF/DWG export if it exists) you have not run through the tool. Aim for variety, not similarity to PLAN1-9: other architects, sheet layouts, scales, roads on any side, scanned and CAD-exported, Kannada text, stilt/basement levels, irregular plots.
2. **`add` never runs an extractor.** Do not run the tool on a held-out plan before its truth is frozen.
3. **Annotate from the sheet, not from the tool.** Fill `data/corpus/truth/<id>.json` by hand. Definitions of every field are at the top of `backend/corpus/schema.py`. Per value set `verification`:
   `printed` (a number printed for exactly this quantity), `derived` (arithmetic on printed numbers), `inspected` (read by eye, nothing printed), or `must_abstain` (the sheet does not determine it, e.g. a road drawn with no width; a system that answers here is fabricating).
4. **Two annotators** if you can. Record disagreements in `discrepancies`; resolve against the sheet.
5. Set `annotation.human_verified` to `true` only when a person has checked every value.
6. `freeze` records sha256 of every file and truth. After that any change fails the evaluation. Adding another held-out plan later is refused; start a new corpus version instead.
7. **Once per release.** Looking at held-out errors and then fixing the code turns those plans into dev plans. Fix from dev; report from held-out; if you must learn from a held-out failure, move the plan to `dev` and collect a replacement.

## Metrics

- **Confident-wrong rate** (headline): of answers shown without a warning (HIGH/MEDIUM confidence, no `.conflict`), the fraction that are wrong. Target: as close to zero as the sample can show.
- **Answer rate / accuracy when answered**, and **risk-coverage** at HIGH, HIGH+MEDIUM and all confidence.
- **Wrong answers caught**: fraction of wrong answers that were flagged or LOW.
- **Spurious**: answers on `must_abstain` fields.
- **PDF-vs-DXF disagreement as an error detector**: whether the independent pipelines disagreeing predicts a wrong value.

`plot.width/depth` and `building.width/depth` are scored as unordered pairs. Truth of tier `unverified` (migrated from a source that itself called it insufficient) is excluded unless `--include-unverified`.

## Corrections (growing `dev` from real usage)

When a reviewer edits a field in the app (`POST /api/jobs/{id}/edit`), the
edit is recorded as a `CorrectionEvent` -- what the pipeline predicted, what
the reviewer typed, and a copy of the source file -- in `data/corpus/corrections/`.
Recording is automatic; nothing here is scored until a human explicitly
promotes it:

```
python -m backend.tools.run_corpus corrections list             # pending (un-promoted)
python -m backend.tools.run_corpus corrections promote <id>      # -> a dev plan, verification="inspected"
```

Two rules, enforced in code (`backend/corpus/corrections.py`), not just here:

- **A correction can never be promoted into `heldout`.** The reviewer saw the
  model's prediction before correcting it, so the correction is anchored on
  it -- exactly the bias the held-out intake's blank-truth-template rule
  (above) exists to avoid.
- **Promotion only fills the field that was actually corrected.** Every other
  field on that plan stays `unannotated` until someone independently verifies
  it from the sheet; promoting a correction never fabricates truth for values
  no one looked at.

This is how the corpus should grow between held-out plan-sourcing efforts,
not a substitute for them (see `STRESS_AND_ABSTENTION_REPORT.md` section 5
for why both are needed). It is also not a training signal for any model --
see that report and `backend/gnn_extraction`'s own findings for why a learned
model trained on this volume of corrections would just overfit; corrections
feed the *evaluation* corpus, nothing is retrained from them automatically.

## Privacy

Plans carry applicant names, survey numbers and addresses. Do not share the corpus or publish figures from it without redaction and permission from the plan owners; `provenance.licence` and `provenance.redacted` exist to track that.
