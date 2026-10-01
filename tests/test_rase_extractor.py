from __future__ import annotations

import json

import pytest

from backend.config import Settings
import backend.rase.extractor as extractor_mod
from backend.rase.extractor import draft_rule, load_drafts, promote_draft, save_draft


FAKE_CHUNKS = [
    {
        "chunk_id": "BBMP::syn.txt::Regulation 5.::0",
        "clause_ref": "Regulation 5.",
        "text": "Regulation 5. Front Setback ... minimum front setback shall be 3.0 metres.",
    },
    {
        "chunk_id": "BBMP::syn.txt::Regulation 5.::1",
        "clause_ref": "Regulation 5.",
        "text": "...applies to plots exceeding 200 square metres.",
    },
    {
        "chunk_id": "BBMP::syn.txt::Regulation 6.::0",
        "clause_ref": "Regulation 6.",
        "text": "Regulation 6. Rear Setback ... minimum rear setback shall be 2.0 metres.",
    },
]

FAKE_DRAFT_JSON = json.dumps(
    {
        "rule_id": "bbmp-front-setback-residential",
        "citation": "Regulation 5",
        "description": "Minimum front setback for residential plots over 200 sq m",
        "applies_when": {"field": "plot.area", "op": ">", "value": 200},
        "threshold": {"field": "setbacks.front", "op": ">=", "value": 3.0, "unit": "m"},
    }
)


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        groq_api_key="fake-key",
        runtime_rules_dir=tmp_path / "runtime_rules",
        rase_min_retrieval_chunks=1,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_draft_rule_success(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor_mod, "hybrid_search", lambda *a, **k: FAKE_CHUNKS)
    monkeypatch.setattr(extractor_mod, "_call_groq", lambda *a, **k: FAKE_DRAFT_JSON)

    settings = _settings(tmp_path)
    draft = draft_rule("front setback", "BBMP", settings=settings)

    assert draft is not None
    assert draft.status == "DRAFT"
    assert draft.rule.rule_id == "bbmp-front-setback-residential"
    assert draft.rule.threshold["field"] == "setbacks.front"
    assert set(draft.source_chunk_ids) == {c["chunk_id"] for c in FAKE_CHUNKS}


def test_draft_rule_refuses_on_insufficient_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor_mod, "hybrid_search", lambda *a, **k: FAKE_CHUNKS[:1])
    monkeypatch.setattr(extractor_mod, "_call_groq", lambda *a, **k: FAKE_DRAFT_JSON)

    settings = _settings(tmp_path, rase_min_retrieval_chunks=5)
    draft = draft_rule("front setback", "BBMP", settings=settings)
    assert draft is None


def test_draft_rule_respects_model_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor_mod, "hybrid_search", lambda *a, **k: FAKE_CHUNKS)
    monkeypatch.setattr(
        extractor_mod, "_call_groq", lambda *a, **k: json.dumps({"refused": True, "reason": "no threshold found"})
    )
    settings = _settings(tmp_path)
    draft = draft_rule("something unrelated", "BBMP", settings=settings)
    assert draft is None


def test_draft_rule_requires_groq_key(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor_mod, "hybrid_search", lambda *a, **k: FAKE_CHUNKS)
    settings = _settings(tmp_path, groq_api_key="")
    with pytest.raises(RuntimeError):
        draft_rule("front setback", "BBMP", settings=settings)


def test_save_load_promote_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(extractor_mod, "hybrid_search", lambda *a, **k: FAKE_CHUNKS)
    monkeypatch.setattr(extractor_mod, "_call_groq", lambda *a, **k: FAKE_DRAFT_JSON)

    settings = _settings(tmp_path)
    draft = draft_rule("front setback", "BBMP", settings=settings)
    save_draft(draft, settings=settings)

    loaded = load_drafts("BBMP", settings=settings)
    assert len(loaded) == 1
    assert loaded[0].rule.rule_id == draft.rule.rule_id

    promoted = promote_draft("bbmp-front-setback-residential", "BBMP", settings=settings)
    assert promoted.version == "1.0.0"

    live_path = settings.runtime_rules_path("BBMP")
    live_rules = json.loads(live_path.read_text())
    assert len(live_rules) == 1
    assert live_rules[0]["rule_id"] == "bbmp-front-setback-residential"


def test_promote_unknown_draft_raises(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(ValueError):
        promote_draft("does-not-exist", "BBMP", settings=settings)
