"""
Correction-capture: turns a reviewer's in-app edit (`POST /api/jobs/{id}/edit`)
into a durable, provenance-tagged record, instead of letting it vanish with the
job's in-memory result. This is how the corpus is meant to grow from real usage
once BBMP users start correcting the pipeline's answers, alongside (not instead
of) plans sourced and annotated directly -- see STRESS_AND_ABSTENTION_REPORT.md
section 5 for why the pipeline still needs both.

Two hard rules, enforced here rather than left to convention:

1. A correction is written to an APPEND-ONLY log
   (`data/corpus/corrections/log.jsonl`), never straight into the corpus
   `manifest.json` or the `dev`/`heldout` trees. Turning one into a scored
   corpus plan (`promote_correction`) is a separate, explicit, human step --
   nothing here does it silently, and nothing here retrains anything.
2. A correction can NEVER be promoted into the held-out split. A reviewer
   correcting a value has already seen the model's prediction, so the
   correction is anchored on it -- exactly the bias `store.add_plan`'s
   held-out path (blank truth template, never pre-filled from an extractor)
   exists to avoid. Silently letting a correction become held-out truth would
   reintroduce that bias into the one split whose entire point is to be free
   of it. `promote_correction` raises `CorpusError` if asked for "heldout";
   there is no override.

Promoting several corrections against the SAME source document accumulates
into one corpus plan (matched by the sha256 of the copied file), rather than
minting a new near-duplicate plan id per correction.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field

from backend.corpus.schema import CANONICAL_FIELDS, Annotation, PlanTruth, Provenance, TruthField
from backend.corpus.store import (
    CorpusError,
    REPO_ROOT,
    corpus_dir,
    load_manifest,
    load_truth,
    save_manifest,
    save_truth,
    sha256_file,
)
from backend.corpus.schema import PlanEntry

CORRECTIONS_DIRNAME = "corrections"
LOG_NAME = "log.jsonl"
PROMOTIONS_NAME = "promotions.json"


class CorrectionEvent(BaseModel):
    """One reviewer edit of one field, as captured at edit time.

    `predicted_*` is what the pipeline shipped before the edit; `corrected_value`
    is what the reviewer typed. Both are kept, deliberately -- an eval could
    later ask "how often was HIGH actually wrong," which needs the prediction,
    not just the correction.
    """

    correction_id: str
    recorded_at: str
    job_id: str
    document_id: str
    field: str
    unit: str = ""
    predicted_value: Optional[Union[float, int, str]] = None
    predicted_confidence: Optional[str] = None
    predicted_source: Optional[str] = None
    corrected_value: Union[float, int, str]
    corrected_by: str = ""
    note: str = ""
    source_sha256: dict[str, str] = Field(default_factory=dict)
    files_dir: Optional[str] = None  # relative to repo root, once files are copied


def _corrections_dir(root: Path = REPO_ROOT) -> Path:
    return corpus_dir(root) / CORRECTIONS_DIRNAME


def _log_path(root: Path = REPO_ROOT) -> Path:
    return _corrections_dir(root) / LOG_NAME


def _promotions_path(root: Path = REPO_ROOT) -> Path:
    return _corrections_dir(root) / PROMOTIONS_NAME


def _load_promotions(root: Path = REPO_ROOT) -> dict[str, str]:
    """correction_id -> promoted-to plan_id."""
    path = _promotions_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_promotions(promotions: dict[str, str], root: Path = REPO_ROOT) -> None:
    path = _promotions_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(promotions, indent=2, sort_keys=True), encoding="utf-8")


def record_correction(
    *,
    job_id: str,
    document_id: str,
    field: str,
    unit: str,
    predicted_value: Optional[Union[float, int, str]],
    predicted_confidence: Optional[str],
    predicted_source: Optional[str],
    corrected_value: Union[float, int, str],
    corrected_by: str = "",
    note: str = "",
    source_files: Optional[dict[str, Path]] = None,
    root: Path = REPO_ROOT,
) -> CorrectionEvent:
    """Append one correction to the log. `source_files` (e.g. {"pdf": path} or
    {"dxf": path}, or both for a dual upload) is copied alongside the log entry
    so the correction survives the upload directory being cleared -- pass it
    (from the job's own saved paths) whenever the files still exist; omit it
    (e.g. a sample-plan job, whose file already lives in the repo) to skip the
    copy and record only the field values.
    """
    correction_id = uuid.uuid4().hex
    source_sha256: dict[str, str] = {}
    files_dir: Optional[str] = None
    if source_files:
        dest_dir = _corrections_dir(root) / "files" / correction_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        for kind, src in source_files.items():
            if src is None or not Path(src).exists():
                continue
            dest = dest_dir / f"{document_id}.{kind}"
            shutil.copyfile(src, dest)
            source_sha256[kind] = sha256_file(dest)
        if source_sha256:
            files_dir = dest_dir.relative_to(root).as_posix()

    event = CorrectionEvent(
        correction_id=correction_id,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        job_id=job_id,
        document_id=document_id,
        field=field,
        unit=unit,
        predicted_value=predicted_value,
        predicted_confidence=predicted_confidence,
        predicted_source=predicted_source,
        corrected_value=corrected_value,
        corrected_by=corrected_by,
        note=note,
        source_sha256=source_sha256,
        files_dir=files_dir,
    )
    log_path = _log_path(root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(event.model_dump_json() + "\n")
    return event


def load_corrections(root: Path = REPO_ROOT) -> list[CorrectionEvent]:
    path = _log_path(root)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(CorrectionEvent.model_validate_json(line))
    return events


def pending_corrections(root: Path = REPO_ROOT) -> list[CorrectionEvent]:
    promoted = _load_promotions(root)
    return [e for e in load_corrections(root) if e.correction_id not in promoted]


def load_promotions(root: Path = REPO_ROOT) -> dict[str, str]:
    """correction_id -> promoted-to plan_id, for every correction promoted so far."""
    return _load_promotions(root)


def promote_correction(
    correction_id: str,
    *,
    split: str = "dev",
    annotator: str = "",
    root: Path = REPO_ROOT,
) -> PlanEntry:
    """Fold one recorded correction into the scored corpus as an `inspected`
    truth field. Never allowed into "heldout" -- see the module docstring.

    If a plan already exists for this correction's source document (matched by
    the sha256 of a previously-promoted correction's copied file, i.e. another
    correction on the SAME upload was promoted earlier), the field is added to
    that plan's truth instead of creating a duplicate plan id.
    """
    if split == "heldout":
        raise CorpusError(
            "a correction can never be promoted into 'heldout': the reviewer already saw "
            "the model's prediction before correcting it, which is exactly the anchoring "
            "bias the held-out split's blank-template intake exists to avoid"
        )
    if split != "dev":
        raise CorpusError(f"split must be 'dev' (or, never, 'heldout'), got {split!r}")

    events = {e.correction_id: e for e in load_corrections(root)}
    event = events.get(correction_id)
    if event is None:
        raise CorpusError(f"no correction recorded with id {correction_id!r}")
    promotions = _load_promotions(root)
    if correction_id in promotions:
        raise CorpusError(f"correction {correction_id!r} was already promoted to {promotions[correction_id]!r}")
    if event.field not in CANONICAL_FIELDS:
        raise CorpusError(
            f"{event.field!r} is not one of the corpus's canonical fields ({sorted(CANONICAL_FIELDS)}); "
            "this correction was recorded for audit purposes but cannot be scored"
        )
    if not event.files_dir:
        raise CorpusError(f"correction {correction_id!r} has no copied source file to promote into the corpus")

    manifest = load_manifest(root)
    existing = _entry_for_same_source(manifest, event, root)

    if existing is not None:
        truth = load_truth(existing, root)
        entry = existing
    else:
        plan_id = f"correction-{event.document_id}-{correction_id[:8]}"
        target_dir = corpus_dir(root) / split / plan_id
        target_dir.mkdir(parents=True, exist_ok=True)
        entry = PlanEntry(
            id=plan_id, split=split, truth=f"data/corpus/truth/{plan_id}.json",
            provenance=Provenance(source=f"user correction on job {event.job_id}"),
            tags=["user_correction"],
        )
        src_dir = root / event.files_dir
        for src in src_dir.iterdir():
            kind = src.suffix.lstrip(".").lower()
            if kind not in ("pdf", "dxf"):
                continue
            dest = target_dir / f"{plan_id}.{kind}"
            shutil.copyfile(src, dest)
            setattr(entry, kind, dest.relative_to(root).as_posix())
            entry.sha256[kind] = sha256_file(dest)
        truth = PlanTruth(
            plan_id=plan_id,
            annotation=Annotation(
                annotator=annotator or event.corrected_by, annotated_on=datetime.now(timezone.utc).date().isoformat(),
                human_verified=True,
                notes="Seeded from a reviewer's in-app correction; verification tier is 'inspected' "
                      "because the reviewer saw the model's prediction before correcting it, not blind "
                      "annotation. Other fields are left unannotated until independently reviewed.",
            ),
            fields={name: TruthField() for name in CANONICAL_FIELDS},
        )
        manifest.plans.append(entry)

    truth.fields[event.field] = TruthField(
        value=event.corrected_value, verification="inspected",
        evidence=f"Reviewer correction, job {event.job_id} ({event.recorded_at}): {event.note}".strip(": "),
    )
    save_truth(truth, root / entry.truth)
    save_manifest(manifest, root)

    promotions[correction_id] = entry.id
    _save_promotions(promotions, root)
    return entry


def _entry_for_same_source(manifest, event: CorrectionEvent, root: Path) -> Optional[PlanEntry]:
    if not event.source_sha256:
        return None
    for entry in manifest.plans:
        if "user_correction" not in entry.tags:
            continue
        if any(entry.sha256.get(kind) == digest for kind, digest in event.source_sha256.items()):
            return entry
    return None


__all__ = [
    "CorrectionEvent", "load_corrections", "load_promotions", "pending_corrections",
    "promote_correction", "record_correction",
]
