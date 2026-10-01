"""
Run the production extraction path over corpus plans and score it.

Extraction is slow (a PDF ~40 s, a DXF ~1-2 min) and scoring is instant, so
NormalizedPlans are cached, keyed by the plan file's sha256 AND a fingerprint
of the backend source. Editing any backend file therefore invalidates the
cache: a stale prediction can never be scored against new code.

The three modalities are the ones the app can actually produce:
  pdf         PDFHybridExtractor(cv only) -> build_normalized_plan
  dxf         DXFHybridExtractor          -> build_normalized_plan
  reconciled  reconcile_pdf_dxf(pdf, dxf) (the dual-upload result)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from backend.corpus.schema import CANONICAL_FIELDS, PlanEntry
from backend.corpus.scoring import aggregate, disagreement_analysis, score_plan
from backend.corpus.store import (
    REPO_ROOT, assert_heldout_intact, corpus_dir, load_manifest, load_truth, plans_in_split,
)

EXTRACTION_TIMEOUT_S = 600.0
MODALITIES = ("pdf", "dxf", "reconciled")


def code_fingerprint(root: Path = REPO_ROOT) -> str:
    h = hashlib.sha256()
    for path in sorted((root / "backend").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def plan_to_predictions(plan) -> dict[str, dict[str, Any]]:
    """The scored fields of a NormalizedPlan as plain dicts."""
    preds: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_FIELDS:
        if name == "building_use":
            vf = getattr(plan, "building_use", None)
        else:
            obj: Any = plan
            for part in name.split("."):
                obj = getattr(obj, part, None) if obj is not None else None
            vf = obj
        if vf is None:
            preds[name] = {"value": None, "confidence": None, "flagged": False}
            continue
        preds[name] = {
            "value": vf.value,
            "confidence": vf.confidence.level.value if vf.confidence else None,
            "flagged": vf.conflict is not None,
        }
    return preds


def _extract(entry: PlanEntry, kind: str, root: Path):
    from backend.spatial_reasoning.pipeline import build_normalized_plan

    path = root / getattr(entry, kind)
    if kind == "pdf":
        from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
        extraction = PDFHybridExtractor(enable_vision=False).extract(path, entry.id)
    else:
        from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
        extraction = DXFHybridExtractor().extract(path, entry.id)
    return build_normalized_plan(extraction, plan_id=f"{entry.id}-{kind}")


def get_plan(entry: PlanEntry, kind: str, *, root: Path = REPO_ROOT, use_cache: bool = True):
    """(NormalizedPlan | None, note). None means no answer (missing file or
    timeout/failure), which scores as MISSING rather than being skipped."""
    from backend.schemas.normalized_plan import NormalizedPlan
    from backend.tools.bounded_execution import ExtractionTimeoutError, run_with_timeout

    if not getattr(entry, kind):
        return None, f"no {kind} file"
    cache = corpus_dir(root) / ".cache" / f"{entry.id}__{kind}__{entry.sha256.get(kind, '')[:12]}__{code_fingerprint(root)}.json"
    if use_cache and cache.exists():
        return NormalizedPlan.model_validate_json(cache.read_text(encoding="utf-8")), "cached"
    started = time.time()
    try:
        plan = run_with_timeout(lambda: _extract(entry, kind, root), EXTRACTION_TIMEOUT_S, f"{entry.id} {kind}")
    except ExtractionTimeoutError:
        return None, f"timed out after {EXTRACTION_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001 - a crash is a result, not a reason to stop the run
        return None, f"failed: {type(exc).__name__}: {exc}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(plan.model_dump_json(), encoding="utf-8")
    return plan, f"extracted in {time.time() - started:.0f}s"


def predictions_for(entry: PlanEntry, modality: str, *, root: Path = REPO_ROOT, use_cache: bool = True):
    if modality == "reconciled":
        pdf_plan, _ = get_plan(entry, "pdf", root=root, use_cache=use_cache)
        dxf_plan, note = get_plan(entry, "dxf", root=root, use_cache=use_cache)
        if pdf_plan is None or dxf_plan is None:
            return None, "needs both a pdf and a dxf"
        from backend.spatial_reasoning.pdf_dxf_reconciliation import reconcile_pdf_dxf
        reconciled, _report = reconcile_pdf_dxf(pdf_plan, dxf_plan)
        return plan_to_predictions(reconciled), "reconciled"
    plan, note = get_plan(entry, modality, root=root, use_cache=use_cache)
    if plan is None:
        return {}, note   # no answer: every field scores MISSING
    return plan_to_predictions(plan), note


def run_eval(
    split: str, *, modalities=MODALITIES, verifications=None, use_cache: bool = True,
    root: Path = REPO_ROOT, log=print,
) -> dict[str, Any]:
    from backend.corpus.schema import SCORABLE_DEFAULT

    if split == "heldout":
        assert_heldout_intact(root)
    manifest = load_manifest(root)
    entries = plans_in_split(manifest, split)
    verifications = tuple(verifications or SCORABLE_DEFAULT)
    truths = {e.id: load_truth(e, root) for e in entries}
    out: dict[str, Any] = {"split": split, "verifications": verifications, "modalities": {}, "notes": {}}
    all_preds: dict[str, dict[str, dict]] = {m: {} for m in modalities}
    for entry in entries:
        for modality in modalities:
            if modality == "reconciled" and not (entry.pdf and entry.dxf):
                continue
            preds, note = predictions_for(entry, modality, root=root, use_cache=use_cache)
            if preds is None:
                continue
            all_preds[modality][entry.id] = preds
            out["notes"][f"{entry.id}/{modality}"] = note
            log(f"  {entry.id:7s} {modality:10s} {note}")
    for modality in modalities:
        per_plan, every = {}, []
        for pid, preds in all_preds[modality].items():
            scores = score_plan(truths[pid], preds, verifications=verifications)
            per_plan[pid] = {"aggregate": aggregate(scores), "scores": [s.__dict__ for s in scores]}
            every += scores
        by_field = {}
        for name in CANONICAL_FIELDS:
            fs = [s for s in every if s.field == name]
            if fs:
                by_field[name] = aggregate(fs)
        out["modalities"][modality] = {"overall": aggregate(every), "by_field": by_field, "per_plan": per_plan}
    if "pdf" in modalities and "dxf" in modalities:
        out["disagreement"] = disagreement_analysis(
            truths, all_preds["pdf"], all_preds["dxf"], verifications=verifications
        )
    return out


def format_report(result: dict[str, Any]) -> str:
    def pct(x: Optional[float]) -> str:
        return "  n/a" if x is None else f"{100 * x:5.1f}%"

    lines = [f"CORPUS EVALUATION -- split={result['split']}  truth tiers={', '.join(result['verifications'])}", ""]
    for modality, data in result["modalities"].items():
        o = data["overall"]
        lines += [
            f"[{modality}]  scored fields={o['n_truth_valued']}  correct={o['correct']}  wrong={o['wrong']}  "
            f"missing={o['missing']}  spurious={o['spurious']}  abstained_ok={o['abstained_ok']}",
            f"    answer rate {pct(o['answer_rate'])} | accuracy when answered {pct(o['accuracy_when_answered'])} | "
            f"CONFIDENT-WRONG {o['confident_wrong']}/{o['confident_answers']} = {pct(o['confident_wrong_rate'])} | "
            f"wrong answers caught (flagged/LOW) {pct(o['wrong_answers_caught'])} | axis-swapped {o['axis_swapped']}",
            "    risk-coverage: " + "; ".join(
                f"{r['confidence_at_least']}: cov {pct(r['coverage'])} risk {pct(r['risk'])} (n={r['answers']})"
                for r in o["risk_coverage"]),
        ]
        lines.append("    per field (correct/wrong/missing/spurious):")
        for name, a in data["by_field"].items():
            lines.append(f"      {name:24s} {a['correct']:2d}/{a['wrong']:2d}/{a['missing']:2d}/{a['spurious']:2d}"
                         f"   confident-wrong {a['confident_wrong']}/{a['confident_answers']}")
        lines.append("")
    d = result.get("disagreement")
    if d and d["pairs"]:
        lines.append(f"[pdf-vs-dxf disagreement as an error detector]  fields both produced: {d['pairs']}")
        for label, key in (("DXF wrong", "detects_dxf_wrong"), ("either wrong", "detects_any_wrong")):
            c = d[key]
            au = "n/a" if c["auroc"] is None else f"{c['auroc']:.2f}"
            lines.append(
                f"    detecting {label:12s}: flagged&wrong {c['flagged_and_wrong']}, flagged&right {c['flagged_but_right']}, "
                f"missed {c['missed_wrong']}, quiet&right {c['quiet_and_right']} | "
                f"precision {pct(c['precision'])} recall {pct(c['recall'])} AUROC {au}")
    return "\n".join(lines)


__all__ = ["MODALITIES", "code_fingerprint", "format_report", "get_plan", "plan_to_predictions",
           "predictions_for", "run_eval"]
