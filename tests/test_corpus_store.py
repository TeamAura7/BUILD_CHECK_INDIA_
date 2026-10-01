"""Tests for `backend.corpus.store`: validation, intake and the held-out freeze."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.corpus import runner, store
from backend.corpus.schema import CANONICAL_FIELDS, Provenance, TruthField
from backend.corpus.store import CorpusError, REPO_ROOT


def _file(tmp_path: Path, name: str, payload: bytes = b"%PDF-1.4 fake") -> Path:
    path = tmp_path / "incoming" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _annotate(root: Path, plan_id: str, *, verified: bool = True) -> None:
    entry = next(e for e in store.load_manifest(root).plans if e.id == plan_id)
    truth = store.load_truth(entry, root)
    truth.fields = {n: TruthField(value=1.0 if n != "building_use" else "residential", verification="printed")
                    for n in CANONICAL_FIELDS}
    truth.annotation.human_verified = verified
    store.save_truth(truth, root / entry.truth)


def test_the_real_corpus_is_valid_and_is_all_development_data():
    assert store.validate_corpus() == []
    manifest = store.load_manifest()
    assert len(manifest.plans) >= 8
    assert {p.split for p in manifest.plans} == {"dev"}, (
        "plans the extractor was developed against must never be labelled held-out"
    )


def test_every_registered_plan_file_still_matches_its_recorded_hash():
    for entry in store.load_manifest().plans:
        for kind in ("pdf", "dxf"):
            rel = getattr(entry, kind)
            if rel:
                assert store.sha256_file(REPO_ROOT / rel) == entry.sha256[kind], rel


def test_heldout_ids_never_appear_in_backend_or_test_code():
    """Leakage guard: no extraction logic or test may name a held-out plan."""
    heldout = [p for p in store.load_manifest().plans if p.split == "heldout"]
    pattern = [re.escape(p.id) for p in heldout]
    if not pattern:
        return
    regex = re.compile("|".join(pattern), re.I)
    for folder in ("backend", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if path.name in ("test_corpus_store.py",):
                continue
            assert not regex.search(path.read_text(encoding="utf-8", errors="ignore")), path


def test_adding_a_heldout_plan_writes_a_blank_template_and_runs_no_extractor(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("intake must never run an extractor")

    monkeypatch.setattr(runner, "_extract", boom)
    entry = store.add_plan(
        "HO-001", "heldout", pdf=_file(tmp_path, "a.pdf"), root=tmp_path,
        provenance=Provenance(source="test", licence="test"),
    )
    truth = store.load_truth(entry, tmp_path)
    assert set(truth.fields) == set(CANONICAL_FIELDS)
    assert all(f.value is None and f.verification == "unannotated" for f in truth.fields.values())
    assert truth.annotation.human_verified is False
    assert (tmp_path / entry.pdf).exists()


def test_freeze_requires_a_human_verified_fully_annotated_truth(tmp_path):
    store.add_plan("HO-001", "heldout", pdf=_file(tmp_path, "a.pdf"), root=tmp_path)
    with pytest.raises(CorpusError, match="human_verified"):
        store.freeze_heldout(tmp_path)
    _annotate(tmp_path, "HO-001", verified=True)
    entry = store.load_manifest(tmp_path).plans[0]
    truth = store.load_truth(entry, tmp_path)
    truth.fields["road.width"] = TruthField()      # one field left blank
    store.save_truth(truth, tmp_path / entry.truth)
    with pytest.raises(CorpusError, match="unannotated"):
        store.freeze_heldout(tmp_path)


def test_a_frozen_heldout_set_detects_any_later_change(tmp_path):
    store.add_plan("HO-001", "heldout", pdf=_file(tmp_path, "a.pdf"), root=tmp_path)
    _annotate(tmp_path, "HO-001")
    with pytest.raises(CorpusError, match="not frozen"):
        store.assert_heldout_intact(tmp_path)
    store.freeze_heldout(tmp_path)
    store.assert_heldout_intact(tmp_path)          # intact right after the freeze

    entry = store.load_manifest(tmp_path).plans[0]
    truth_path = tmp_path / entry.truth
    original = truth_path.read_text(encoding="utf-8")
    truth_path.write_text(original.replace('"value": 1.0', '"value": 2.0', 1), encoding="utf-8")
    with pytest.raises(CorpusError, match="changed after the freeze"):
        store.assert_heldout_intact(tmp_path)
    truth_path.write_text(original, encoding="utf-8")
    store.assert_heldout_intact(tmp_path)

    (tmp_path / entry.pdf).write_bytes(b"tampered")
    with pytest.raises(CorpusError, match="changed after the freeze"):
        store.assert_heldout_intact(tmp_path)


def test_plans_cannot_be_added_to_a_frozen_heldout_set(tmp_path):
    store.add_plan("HO-001", "heldout", pdf=_file(tmp_path, "a.pdf"), root=tmp_path)
    _annotate(tmp_path, "HO-001")
    store.freeze_heldout(tmp_path)
    with pytest.raises(CorpusError, match="frozen"):
        store.add_plan("HO-002", "heldout", pdf=_file(tmp_path, "b.pdf"), root=tmp_path)
    store.add_plan("DEV-001", "dev", pdf=_file(tmp_path, "c.pdf"), root=tmp_path)   # dev is unaffected


def test_evaluating_an_unfrozen_heldout_set_is_refused_before_any_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_extract", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not extract")))
    store.add_plan("HO-001", "heldout", pdf=_file(tmp_path, "a.pdf"), root=tmp_path)
    with pytest.raises(CorpusError, match="not frozen"):
        runner.run_eval("heldout", root=tmp_path, log=lambda *_: None)


def test_duplicate_ids_and_bad_splits_are_rejected(tmp_path):
    store.add_plan("X", "dev", pdf=_file(tmp_path, "a.pdf"), root=tmp_path)
    with pytest.raises(CorpusError, match="already exists"):
        store.add_plan("X", "dev", pdf=_file(tmp_path, "b.pdf"), root=tmp_path)
    with pytest.raises(CorpusError, match="split"):
        store.add_plan("Y", "test", pdf=_file(tmp_path, "c.pdf"), root=tmp_path)
    with pytest.raises(CorpusError, match="at least one"):
        store.add_plan("Z", "dev", root=tmp_path)


def test_validation_reports_missing_files_unknown_fields_and_inconsistent_truth(tmp_path):
    entry = store.add_plan("V", "dev", pdf=_file(tmp_path, "a.pdf"), root=tmp_path)
    truth = store.load_truth(entry, tmp_path)
    truth.fields["not.a.field"] = TruthField(value=1.0, verification="printed")
    truth.fields["road.width"] = TruthField(value=5.0, verification="must_abstain")
    truth.fields["plot.width"] = TruthField(value=None, verification="printed")
    store.save_truth(truth, tmp_path / entry.truth)
    (tmp_path / entry.pdf).unlink()
    problems = "\n".join(store.validate_corpus(tmp_path))
    assert "file missing" in problems
    assert "unknown field" in problems
    assert "must_abstain but has a value" in problems
    assert "printed but has no value" in problems


def test_the_cache_is_invalidated_by_any_backend_source_change(tmp_path):
    (tmp_path / "backend").mkdir()
    module = tmp_path / "backend" / "m.py"
    module.write_text("x = 1", encoding="utf-8")
    before = runner.code_fingerprint(tmp_path)
    assert runner.code_fingerprint(tmp_path) == before
    module.write_text("x = 2", encoding="utf-8")
    assert runner.code_fingerprint(tmp_path) != before
