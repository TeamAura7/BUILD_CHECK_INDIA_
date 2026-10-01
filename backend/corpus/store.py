"""
Corpus storage, validation, intake and the held-out freeze lock.

Held-out protocol (enforced here, not just documented):

  1. `add_plan(split="heldout")` copies the files and writes a BLANK truth
     template. It never runs an extractor, and never pre-fills a value from
     one: annotating against a model's output anchors the annotator on the
     model's mistakes.
  2. A human fills the truth file in and sets `annotation.human_verified`.
  3. `freeze_heldout` refuses unless every held-out plan is annotated and
     human-verified, then records the sha256 of every plan file and truth
     file in `heldout.lock.json`.
  4. `assert_heldout_intact` (called before any held-out evaluation) fails if
     a plan is unfrozen, or a file or truth changed after the freeze.

Changing a held-out plan after seeing results is what turns a test set into a
development set, so the lock makes that a loud failure, not a quiet one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.corpus.schema import (
    CANONICAL_FIELDS,
    Manifest,
    PlanEntry,
    PlanTruth,
    Provenance,
    TruthField,
    VERIFICATION_ORDER,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCK_NAME = "heldout.lock.json"


class CorpusError(Exception):
    pass


def corpus_dir(root: Path = REPO_ROOT) -> Path:
    return root / "data" / "corpus"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(root: Path = REPO_ROOT) -> Manifest:
    path = corpus_dir(root) / "manifest.json"
    if not path.exists():
        return Manifest()
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: Manifest, root: Path = REPO_ROOT) -> None:
    path = corpus_dir(root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_truth(entry: PlanEntry, root: Path = REPO_ROOT) -> PlanTruth:
    return PlanTruth.model_validate_json((root / entry.truth).read_text(encoding="utf-8"))


def save_truth(truth: PlanTruth, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(truth.model_dump_json(indent=2), encoding="utf-8")


def plans_in_split(manifest: Manifest, split: str) -> list[PlanEntry]:
    return [p for p in manifest.plans if p.split == split]


def _lock_path(root: Path) -> Path:
    return corpus_dir(root) / LOCK_NAME


def _entry_hashes(entry: PlanEntry, root: Path) -> dict[str, str]:
    hashes = {}
    for kind in ("pdf", "dxf"):
        rel = getattr(entry, kind)
        if rel:
            hashes[kind] = sha256_file(root / rel)
    hashes["truth"] = sha256_file(root / entry.truth)
    return hashes


def validate_corpus(root: Path = REPO_ROOT) -> list[str]:
    """Every problem found, as a list of strings (empty means valid)."""
    problems: list[str] = []
    manifest = load_manifest(root)
    seen: set[str] = set()
    for entry in manifest.plans:
        tag = f"{entry.id} ({entry.split})"
        if entry.id in seen:
            problems.append(f"{tag}: duplicate plan id")
        seen.add(entry.id)
        if not entry.pdf and not entry.dxf:
            problems.append(f"{tag}: has neither a pdf nor a dxf")
        for kind in ("pdf", "dxf"):
            rel = getattr(entry, kind)
            if not rel:
                continue
            path = root / rel
            if not path.exists():
                problems.append(f"{tag}: {kind} file missing: {rel}")
                continue
            recorded = entry.sha256.get(kind)
            if recorded and recorded != sha256_file(path):
                problems.append(f"{tag}: {kind} file changed since it was registered ({rel})")
        truth_path = root / entry.truth
        if not truth_path.exists():
            problems.append(f"{tag}: truth file missing: {entry.truth}")
            continue
        try:
            truth = load_truth(entry, root)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{tag}: truth file does not parse: {exc}")
            continue
        if truth.plan_id != entry.id:
            problems.append(f"{tag}: truth file is for {truth.plan_id!r}")
        for name, tf in truth.fields.items():
            if name not in CANONICAL_FIELDS:
                problems.append(f"{tag}: unknown field {name!r}")
            if tf.verification == "must_abstain" and tf.value is not None:
                problems.append(f"{tag}: {name} is must_abstain but has a value")
            if tf.verification in VERIFICATION_ORDER and tf.value is None:
                problems.append(f"{tag}: {name} is {tf.verification} but has no value")
            if CANONICAL_FIELDS.get(name) not in (None, "category") and isinstance(tf.value, str):
                problems.append(f"{tag}: {name} must be numeric, got {tf.value!r}")
    return problems


def add_plan(
    plan_id: str,
    split: str,
    *,
    pdf: Optional[Path] = None,
    dxf: Optional[Path] = None,
    provenance: Optional[Provenance] = None,
    tags: Optional[list[str]] = None,
    root: Path = REPO_ROOT,
) -> PlanEntry:
    """Register a new plan with a BLANK truth template. Never runs an extractor."""
    if split not in ("dev", "heldout"):
        raise CorpusError(f"split must be 'dev' or 'heldout', got {split!r}")
    if not pdf and not dxf:
        raise CorpusError("provide at least one of pdf / dxf")
    manifest = load_manifest(root)
    if any(p.id == plan_id for p in manifest.plans):
        raise CorpusError(f"plan id {plan_id!r} already exists")
    if split == "heldout" and _lock_path(root).exists():
        raise CorpusError("the held-out set is frozen; adding to it would silently change the test set")

    target_dir = corpus_dir(root) / split / plan_id
    target_dir.mkdir(parents=True, exist_ok=True)
    entry = PlanEntry(
        id=plan_id, split=split, truth=f"data/corpus/truth/{plan_id}.json",
        provenance=provenance or Provenance(), tags=tags or [],
    )
    for kind, src in (("pdf", pdf), ("dxf", dxf)):
        if src is None:
            continue
        dest = target_dir / f"{plan_id}.{kind}"
        shutil.copyfile(src, dest)
        setattr(entry, kind, dest.relative_to(root).as_posix())
        entry.sha256[kind] = sha256_file(dest)

    truth = PlanTruth(plan_id=plan_id, fields={name: TruthField() for name in CANONICAL_FIELDS})
    save_truth(truth, root / entry.truth)
    manifest.plans.append(entry)
    save_manifest(manifest, root)
    return entry


def freeze_heldout(root: Path = REPO_ROOT) -> dict:
    manifest = load_manifest(root)
    heldout = plans_in_split(manifest, "heldout")
    if not heldout:
        raise CorpusError("there are no held-out plans to freeze")
    if _lock_path(root).exists():
        raise CorpusError("the held-out set is already frozen")
    for entry in heldout:
        truth = load_truth(entry, root)
        if not truth.annotation.human_verified:
            raise CorpusError(f"{entry.id}: truth is not marked human_verified")
        blank = [n for n, f in truth.fields.items() if f.verification == "unannotated"]
        if blank:
            raise CorpusError(f"{entry.id}: unannotated fields remain: {', '.join(blank)}")
    lock = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "plans": {e.id: _entry_hashes(e, root) for e in heldout},
    }
    _lock_path(root).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def assert_heldout_intact(root: Path = REPO_ROOT) -> None:
    manifest = load_manifest(root)
    heldout = plans_in_split(manifest, "heldout")
    if not heldout:
        raise CorpusError("there are no held-out plans")
    if not _lock_path(root).exists():
        raise CorpusError("held-out plans exist but are not frozen; run `freeze` first")
    lock = json.loads(_lock_path(root).read_text(encoding="utf-8"))["plans"]
    for entry in heldout:
        if entry.id not in lock:
            raise CorpusError(f"{entry.id} was added after the freeze")
        if _entry_hashes(entry, root) != lock[entry.id]:
            raise CorpusError(f"{entry.id}: files or truth changed after the freeze")
    for plan_id in lock:
        if not any(e.id == plan_id for e in heldout):
            raise CorpusError(f"{plan_id} was removed after the freeze")


__all__ = [
    "CorpusError", "add_plan", "assert_heldout_intact", "corpus_dir", "freeze_heldout",
    "load_manifest", "load_truth", "plans_in_split", "save_manifest", "save_truth",
    "sha256_file", "validate_corpus", "REPO_ROOT",
]
