"""Tests for `backend.corpus.corrections`: recording, listing and promoting a
reviewer's in-app field edit into the scored corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.corpus import corrections as corr
from backend.corpus import store
from backend.corpus.store import CorpusError


def _dxf(tmp_path: Path, name: str = "plan.dxf", payload: bytes = b"0\nSECTION\nfake dxf\n") -> Path:
    path = tmp_path / "incoming" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _record(tmp_path, *, field="plot.width", document_id="doc1", job_id="job1", source=None, **kw):
    return corr.record_correction(
        job_id=job_id, document_id=document_id, field=field, unit=kw.pop("unit", "m"),
        predicted_value=kw.pop("predicted_value", 4.7), predicted_confidence=kw.pop("predicted_confidence", "LOW"),
        predicted_source=kw.pop("predicted_source", "VECTOR_GEOMETRY"),
        corrected_value=kw.pop("corrected_value", 15.0), corrected_by=kw.pop("corrected_by", "reviewer@example.com"),
        note=kw.pop("note", "measured on the printed dimension line"),
        source_files={"dxf": source} if source else None, root=tmp_path, **kw,
    )


def test_recording_a_correction_appends_to_the_log_and_copies_the_source_file(tmp_path):
    src = _dxf(tmp_path)
    event = _record(tmp_path, source=src)
    events = corr.load_corrections(tmp_path)
    assert len(events) == 1 and events[0].correction_id == event.correction_id
    assert events[0].corrected_value == 15.0 and events[0].predicted_value == 4.7
    assert event.files_dir is not None
    assert (tmp_path / event.files_dir / "doc1.dxf").exists()
    assert event.source_sha256["dxf"] == store.sha256_file(src)


def test_recording_without_source_files_still_logs_the_field_values(tmp_path):
    event = _record(tmp_path, source=None)
    assert event.files_dir is None and event.source_sha256 == {}
    assert corr.load_corrections(tmp_path)[0].correction_id == event.correction_id


def test_pending_corrections_excludes_already_promoted_ones(tmp_path):
    e1 = _record(tmp_path, source=_dxf(tmp_path, "a.dxf"), field="plot.width")
    e2 = _record(tmp_path, source=_dxf(tmp_path, "b.dxf"), field="plot.depth", document_id="doc2", job_id="job2")
    assert {e.correction_id for e in corr.pending_corrections(tmp_path)} == {e1.correction_id, e2.correction_id}
    corr.promote_correction(e1.correction_id, root=tmp_path)
    assert {e.correction_id for e in corr.pending_corrections(tmp_path)} == {e2.correction_id}
    assert corr.load_promotions(tmp_path) == {e1.correction_id: corr.load_promotions(tmp_path)[e1.correction_id]}


def test_promoting_a_correction_creates_a_dev_plan_with_an_inspected_field(tmp_path):
    src = _dxf(tmp_path)
    event = _record(tmp_path, source=src, field="plot.width", corrected_value=15.0)
    entry = corr.promote_correction(event.correction_id, root=tmp_path)
    assert entry.split == "dev" and "user_correction" in entry.tags
    truth = store.load_truth(entry, tmp_path)
    assert truth.fields["plot.width"].value == 15.0
    assert truth.fields["plot.width"].verification == "inspected"
    assert truth.annotation.human_verified is True
    # every OTHER field is left unannotated -- promotion never fabricates truth
    # for fields the reviewer never actually corrected
    assert truth.fields["plot.depth"].verification == "unannotated"
    manifest = store.load_manifest(tmp_path)
    assert any(p.id == entry.id for p in manifest.plans)
    assert store.validate_corpus(tmp_path) == []


def test_promoting_into_heldout_is_always_refused(tmp_path):
    event = _record(tmp_path, source=_dxf(tmp_path))
    with pytest.raises(CorpusError, match="heldout"):
        corr.promote_correction(event.correction_id, split="heldout", root=tmp_path)
    # the correction must still be pending -- the refused attempt didn't half-apply
    assert event.correction_id in {e.correction_id for e in corr.pending_corrections(tmp_path)}


def test_promoting_twice_is_refused(tmp_path):
    event = _record(tmp_path, source=_dxf(tmp_path))
    corr.promote_correction(event.correction_id, root=tmp_path)
    with pytest.raises(CorpusError, match="already promoted"):
        corr.promote_correction(event.correction_id, root=tmp_path)


def test_promoting_an_unrecorded_id_is_refused(tmp_path):
    with pytest.raises(CorpusError, match="no correction"):
        corr.promote_correction("does-not-exist", root=tmp_path)


def test_a_non_canonical_field_cannot_be_promoted_but_stays_on_record(tmp_path):
    """building.height_estimated is editable in the UI but is not one of the
    corpus's canonical scored fields -- it should be recorded for audit
    purposes, and refused at promotion time, not silently dropped earlier."""
    event = _record(tmp_path, source=_dxf(tmp_path), field="building.height_estimated", corrected_value=9.5, unit="m")
    assert event.field == "building.height_estimated"
    with pytest.raises(CorpusError, match="canonical"):
        corr.promote_correction(event.correction_id, root=tmp_path)
    assert event.correction_id in {e.correction_id for e in corr.pending_corrections(tmp_path)}


def test_a_correction_with_no_copied_source_file_cannot_be_promoted(tmp_path):
    event = _record(tmp_path, source=None)
    with pytest.raises(CorpusError, match="no copied source file"):
        corr.promote_correction(event.correction_id, root=tmp_path)


def test_two_corrections_on_the_same_source_document_accumulate_into_one_plan(tmp_path):
    src = _dxf(tmp_path, "same.dxf")
    e1 = _record(tmp_path, source=src, field="plot.width", corrected_value=15.0, document_id="docA", job_id="jobA")
    e2 = _record(tmp_path, source=src, field="plot.depth", corrected_value=20.0, document_id="docA", job_id="jobB")
    entry1 = corr.promote_correction(e1.correction_id, root=tmp_path)
    entry2 = corr.promote_correction(e2.correction_id, root=tmp_path)
    assert entry1.id == entry2.id, "a second correction on the same uploaded document should not mint a duplicate plan"
    truth = store.load_truth(entry1, tmp_path)
    assert truth.fields["plot.width"].value == 15.0 and truth.fields["plot.depth"].value == 20.0
    assert store.validate_corpus(tmp_path) == []


def test_split_other_than_dev_or_heldout_is_refused(tmp_path):
    event = _record(tmp_path, source=_dxf(tmp_path))
    with pytest.raises(CorpusError, match="'dev'"):
        corr.promote_correction(event.correction_id, split="bogus", root=tmp_path)
