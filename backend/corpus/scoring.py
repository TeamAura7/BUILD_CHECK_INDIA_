"""
Scoring against corpus truth. Pure functions: no extractor is imported, so
predictions can be cached and re-scored instantly.

The headline quantity is not accuracy but the CONFIDENT-WRONG rate: of the
values the system presents without a warning (HIGH/MEDIUM confidence, no
`.conflict`), how many are wrong. A wrong number that looks trustworthy is
what a compliance tool must not ship; a missing or flagged one is safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from backend.corpus.schema import AXIS_PAIRS, CANONICAL_FIELDS, SCORABLE_DEFAULT, PlanTruth

# Same regime as `backend/tools/eval_harness.py` (pinned by a test so the two
# cannot drift apart).
ABS_TOL = 0.15
REL_TOL = 0.05

CONFIDENT_LEVELS = ("HIGH", "MEDIUM")
_LEVEL_SETS = (("HIGH", ("HIGH",)), ("HIGH+MEDIUM", ("HIGH", "MEDIUM")), ("ALL", ("HIGH", "MEDIUM", "LOW")))


def within_tolerance(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(ABS_TOL, abs(expected) * REL_TOL)


def _matches(field_name: str, predicted: Any, expected: Any) -> bool:
    if predicted is None or expected is None:
        return False
    unit = CANONICAL_FIELDS[field_name]
    if unit == "category":
        return str(predicted).strip().lower() == str(expected).strip().lower()
    if unit == "count":
        return int(round(float(predicted))) == int(expected)
    return within_tolerance(float(predicted), float(expected))


@dataclass
class FieldScore:
    field: str
    status: str  # CORRECT | WRONG | MISSING | SPURIOUS | ABSTAINED_OK
    truth: Any = None
    predicted: Any = None
    confidence: Optional[str] = None
    flagged: bool = False
    axis_swapped: bool = False

    @property
    def answered(self) -> bool:
        return self.status in ("CORRECT", "WRONG", "SPURIOUS")

    @property
    def confident(self) -> bool:
        return self.answered and self.confidence in CONFIDENT_LEVELS and not self.flagged


def _pred(preds: dict[str, dict], name: str) -> tuple[Any, Optional[str], bool]:
    p = preds.get(name) or {}
    return p.get("value"), p.get("confidence"), bool(p.get("flagged"))


def _score_one(name: str, truth_value: Any, must_abstain: bool, preds: dict[str, dict]) -> FieldScore:
    value, level, flagged = _pred(preds, name)
    if must_abstain:
        status = "ABSTAINED_OK" if value is None else "SPURIOUS"
        return FieldScore(name, status, None, value, level, flagged)
    if value is None:
        return FieldScore(name, "MISSING", truth_value, None, level, flagged)
    status = "CORRECT" if _matches(name, value, truth_value) else "WRONG"
    return FieldScore(name, status, truth_value, value, level, flagged)


def _score_pair(
    a: str, b: str, truth: dict[str, Any], preds: dict[str, dict]
) -> list[FieldScore]:
    ta, tb = truth[a], truth[b]
    (pa, la, fa), (pb, lb, fb) = _pred(preds, a), _pred(preds, b)
    if pa is not None and pb is not None:
        direct = int(_matches(a, pa, ta)) + int(_matches(b, pb, tb))
        swapped = int(_matches(a, pa, tb)) + int(_matches(b, pb, ta))
        use_swap = swapped > direct
        ea, eb = (tb, ta) if use_swap else (ta, tb)
        return [
            FieldScore(a, "CORRECT" if _matches(a, pa, ea) else "WRONG", ta, pa, la, fa, use_swap),
            FieldScore(b, "CORRECT" if _matches(b, pb, eb) else "WRONG", tb, pb, lb, fb, use_swap),
        ]
    out = []
    for name, own, other, value, level, flagged in ((a, ta, tb, pa, la, fa), (b, tb, ta, pb, lb, fb)):
        if value is None:
            out.append(FieldScore(name, "MISSING", own, None, level, flagged))
        else:
            ok = _matches(name, value, own) or _matches(name, value, other)
            out.append(FieldScore(name, "CORRECT" if ok else "WRONG", own, value, level, flagged))
    return out


def score_plan(
    truth: PlanTruth,
    preds: dict[str, dict],
    *,
    verifications: Iterable[str] = SCORABLE_DEFAULT,
) -> list[FieldScore]:
    """Score one plan's predictions. Fields whose truth tier is not allowed
    (e.g. `unverified`) or that are unannotated are left out entirely."""
    allowed = set(verifications)
    numeric: dict[str, Any] = {}
    abstain: set[str] = set()
    for name, tf in truth.fields.items():
        if name not in CANONICAL_FIELDS:
            continue
        if tf.verification == "must_abstain":
            abstain.add(name)
        elif tf.verification in allowed and tf.value is not None:
            numeric[name] = tf.value

    scores: list[FieldScore] = []
    done: set[str] = set()
    for a, b in AXIS_PAIRS:
        if a in numeric and b in numeric:
            scores += _score_pair(a, b, numeric, preds)
            done |= {a, b}
    for name in numeric:
        if name not in done:
            scores.append(_score_one(name, numeric[name], False, preds))
    for name in sorted(abstain):
        scores.append(_score_one(name, None, True, preds))
    return scores


def aggregate(scores: Iterable[FieldScore]) -> dict[str, Any]:
    scores = list(scores)
    valued = [s for s in scores if s.status in ("CORRECT", "WRONG", "MISSING")]
    answered = [s for s in scores if s.answered]
    wrong = [s for s in answered if s.status != "CORRECT"]
    confident = [s for s in answered if s.confident]
    confident_wrong = [s for s in confident if s.status != "CORRECT"]
    correct = sum(1 for s in scores if s.status == "CORRECT")

    risk_coverage = []
    for label, levels in _LEVEL_SETS:
        at = [s for s in answered if s.confidence in levels]
        at_valued = [s for s in at if s.status in ("CORRECT", "WRONG")]
        risk_coverage.append({
            "confidence_at_least": label,
            "coverage": (len(at_valued) / len(valued)) if valued else None,
            "risk": (sum(1 for s in at if s.status != "CORRECT") / len(at)) if at else None,
            "answers": len(at),
        })
    return {
        "n_truth_valued": len(valued),
        "correct": correct,
        "wrong": sum(1 for s in scores if s.status == "WRONG"),
        "missing": sum(1 for s in scores if s.status == "MISSING"),
        "spurious": sum(1 for s in scores if s.status == "SPURIOUS"),
        "abstained_ok": sum(1 for s in scores if s.status == "ABSTAINED_OK"),
        "answer_rate": (sum(1 for s in valued if s.answered) / len(valued)) if valued else None,
        "accuracy_when_answered": (
            sum(1 for s in valued if s.status == "CORRECT") / max(1, sum(1 for s in valued if s.answered))
            if any(s.answered for s in valued) else None
        ),
        "confident_answers": len(confident),
        "confident_wrong": len(confident_wrong),
        "confident_wrong_rate": (len(confident_wrong) / len(confident)) if confident else None,
        "wrong_answers_caught": (
            sum(1 for s in wrong if not s.confident) / len(wrong) if wrong else None
        ),
        "axis_swapped": sum(1 for s in scores if s.axis_swapped),
        "risk_coverage": risk_coverage,
    }


def _auroc(scores: list[float], positive: list[bool]) -> Optional[float]:
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def disagreement_analysis(
    truth_by_plan: dict[str, PlanTruth],
    pdf_preds: dict[str, dict[str, dict]],
    dxf_preds: dict[str, dict[str, dict]],
    *,
    verifications: Iterable[str] = SCORABLE_DEFAULT,
) -> dict[str, Any]:
    """Does disagreement between the independent PDF and DXF pipelines predict
    error? For each field both produced (and truth exists), `disagree` is
    "differ beyond tolerance"; it is scored as a detector of "at least one
    modality is wrong" and of "the DXF is wrong". Independence is what makes
    this informative: neither pipeline ever sees the other's output."""
    rows = []
    for plan_id, truth in truth_by_plan.items():
        if plan_id not in pdf_preds or plan_id not in dxf_preds:
            continue
        pdf_scores = {s.field: s for s in score_plan(truth, pdf_preds[plan_id], verifications=verifications)}
        dxf_scores = {s.field: s for s in score_plan(truth, dxf_preds[plan_id], verifications=verifications)}
        for name, ps in pdf_scores.items():
            ds = dxf_scores.get(name)
            if ds is None or ps.predicted is None or ds.predicted is None:
                continue
            if CANONICAL_FIELDS[name] == "category":
                continue
            a, b = float(ps.predicted), float(ds.predicted)
            rel = abs(a - b) / max(abs(a), abs(b), 1e-9)
            rows.append({
                "plan": plan_id, "field": name, "relative_difference": rel,
                "disagree": not within_tolerance(b, a),
                "pdf_wrong": ps.status != "CORRECT", "dxf_wrong": ds.status != "CORRECT",
            })
    def confusion(target: str) -> dict[str, Any]:
        tp = sum(1 for r in rows if r["disagree"] and r[target])
        fp = sum(1 for r in rows if r["disagree"] and not r[target])
        fn = sum(1 for r in rows if not r["disagree"] and r[target])
        tn = sum(1 for r in rows if not r["disagree"] and not r[target])
        return {
            "flagged_and_wrong": tp, "flagged_but_right": fp, "missed_wrong": fn, "quiet_and_right": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "auroc": _auroc([r["relative_difference"] for r in rows], [r[target] for r in rows]),
        }
    for r in rows:
        r["any_wrong"] = r["pdf_wrong"] or r["dxf_wrong"]
    return {"pairs": len(rows), "detects_dxf_wrong": confusion("dxf_wrong"),
            "detects_any_wrong": confusion("any_wrong"), "rows": rows}


__all__ = [
    "ABS_TOL", "REL_TOL", "FieldScore", "aggregate", "disagreement_analysis",
    "score_plan", "within_tolerance",
]
