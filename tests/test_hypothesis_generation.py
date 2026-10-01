"""
Architecture V2, Phase 4/5 -- unit tests for
`backend.spatial_reasoning.hypothesis_generation.grow_clusters`, the
format-agnostic extraction of
`backend.cv_extraction.dxf_extractor._merge_candidate_region_pairs`'s
agglomerative growth algorithm (see ARCHITECTURE_V2.md Deliverable C.2 item 3).

These exercise the generic algorithm directly, with simple integer/string
payloads standing in for whatever a real caller's fragments/resolved
candidates would be -- `test_dxf_extractor.py`'s real-fixture tests
(including `test_merge_accepts_two_regions_that_together_resolve_plot_and_
building`'s exact-warning-text assertion) are the integration-level
regression floor proving `_merge_candidate_region_pairs` itself, which now
delegates to `grow_clusters`, still produces identical merge decisions.
"""

from __future__ import annotations

import time

from backend.spatial_reasoning.hypothesis_generation import HypothesisCluster, grow_clusters


def _make_id_counter(start: int):
    box = [start]

    def _next() -> int:
        value = box[0]
        box[0] += 1
        return value

    return _next


def test_fewer_than_two_clusters_grows_nothing():
    assert grow_clusters([HypothesisCluster(id=0, score=1.0, resolved="a", payload="a")],
                          pool_fn=lambda a, b: a + b, resolve_fn=lambda p: p, score_fn=lambda r: 1.0,
                          next_id_fn=_make_id_counter(100)) == []


def test_a_pair_that_improves_on_both_inputs_is_merged():
    # Two fragments "A" and "B" whose combined score (5.0) beats both
    # their own scores (1.0, 2.0) -- pool/resolve/score are simple string
    # concatenation/length stand-ins for real geometry.
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved="A", payload="A"),
        HypothesisCluster(id=1, score=2.0, resolved="B", payload="B"),
    ]
    grown = grow_clusters(
        clusters,
        pool_fn=lambda a, b: a + b,
        resolve_fn=lambda payload: payload,
        score_fn=lambda resolved: 5.0 if resolved == "AB" else 0.0,
        next_id_fn=_make_id_counter(100),
    )
    assert len(grown) == 1
    assert grown[0].payload == "AB"
    assert grown[0].score == 5.0
    assert grown[0].id == 100


def test_a_pair_that_does_not_improve_on_both_inputs_is_not_merged():
    clusters = [
        HypothesisCluster(id=0, score=5.0, resolved="A", payload="A"),
        HypothesisCluster(id=1, score=2.0, resolved="B", payload="B"),
    ]
    grown = grow_clusters(
        clusters,
        pool_fn=lambda a, b: a + b,
        # merged score (3.0) beats B (2.0) but not A (5.0) -- must not merge
        resolve_fn=lambda payload: payload,
        score_fn=lambda resolved: 3.0 if resolved == "AB" else 0.0,
        next_id_fn=_make_id_counter(100),
    )
    assert grown == []


def test_area_sane_gate_blocks_an_otherwise_winning_merge():
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved=10.0, payload="A"),
        HypothesisCluster(id=1, score=2.0, resolved=10.0, payload="B"),
    ]
    grown = grow_clusters(
        clusters,
        pool_fn=lambda a, b: a + b,
        resolve_fn=lambda payload: 100.0,  # implausibly large merged "area"
        score_fn=lambda resolved: 5.0,
        next_id_fn=_make_id_counter(100),
        area_sane_fn=lambda ra, rb, rm: rm <= (ra + rb) * 1.15,  # 100 > (10+10)*1.15 -- insane
    )
    assert grown == []


def test_veto_blocks_an_otherwise_winning_merge_and_fires_the_callback():
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved="A", payload="A"),
        HypothesisCluster(id=1, score=2.0, resolved="B", payload="B"),
    ]
    vetoed_calls = []
    grown = grow_clusters(
        clusters,
        pool_fn=lambda a, b: a + b,
        resolve_fn=lambda payload: payload,
        score_fn=lambda resolved: 5.0 if resolved == "AB" else 0.0,
        next_id_fn=_make_id_counter(100),
        veto_fn=lambda a, b, merged: "moves away from confirmed evidence",
        on_merge_vetoed=lambda a, b, score, reason: vetoed_calls.append((a.id, b.id, score, reason)),
    )
    assert grown == []
    assert vetoed_calls == [(0, 1, 5.0, "moves away from confirmed evidence")]


def test_growth_continues_across_multiple_rounds():
    # A+B merges first (score 5), then (A+B)+C merges again (score 9),
    # matching _merge_candidate_region_pairs's "no fixed combination size"
    # design -- growth keeps going as long as improvement keeps happening.
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved="A", payload="A"),
        HypothesisCluster(id=1, score=2.0, resolved="B", payload="B"),
        HypothesisCluster(id=2, score=1.0, resolved="C", payload="C"),
    ]

    def _score(resolved: str) -> float:
        return {"A": 1.0, "B": 2.0, "C": 1.0, "AB": 5.0, "AC": 0.0, "BC": 0.0, "ABC": 9.0}.get(resolved, -100.0)

    grown = grow_clusters(
        clusters,
        pool_fn=lambda a, b: "".join(sorted(a + b, key=lambda ch: "ABC".index(ch))),
        resolve_fn=lambda payload: payload,
        score_fn=_score,
        next_id_fn=_make_id_counter(100),
    )
    assert [c.payload for c in grown] == ["AB", "ABC"]
    assert grown[-1].score == 9.0


def test_priority_pairs_are_tried_before_pairs_involving_an_extra():
    # If an "extra" (is_priority=False) cluster's pair is evaluated first,
    # it would win (score 5); but the priority-only pair must be tried
    # first per _pair_sort_key, and it also wins (score 6) -- growth must
    # accept the priority pair, not the extra one, on the very first round.
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved="A", payload="A", is_priority=True),
        HypothesisCluster(id=1, score=1.0, resolved="B", payload="B", is_priority=True),
        HypothesisCluster(id=2, score=1.0, resolved="C", payload="C", is_priority=False),
    ]

    def _score(resolved: str) -> float:
        return {"A": 1.0, "B": 1.0, "C": 1.0, "AB": 6.0, "AC": 5.0, "BC": 5.0}.get(resolved, -100.0)

    accepted = []
    grow_clusters(
        clusters,
        pool_fn=lambda a, b: "".join(sorted(a + b)),
        resolve_fn=lambda payload: payload,
        score_fn=_score,
        next_id_fn=_make_id_counter(100),
        on_merge_accepted=lambda a, b, new: accepted.append(new.payload),
    )
    assert accepted[0] == "AB"


def test_time_budget_stops_growth_and_fires_the_callback():
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved="A", payload="A"),
        HypothesisCluster(id=1, score=1.0, resolved="B", payload="B"),
    ]
    exhausted = []

    def _slow_resolve(payload):
        time.sleep(0.05)
        return payload

    grown = grow_clusters(
        clusters,
        pool_fn=lambda a, b: a + b,
        resolve_fn=_slow_resolve,
        score_fn=lambda resolved: 5.0 if resolved == "AB" else 0.0,
        next_id_fn=_make_id_counter(100),
        time_budget_seconds=0.0,  # already exhausted before the first pair is even tried
        on_budget_exhausted=lambda: exhausted.append(True),
    )
    assert grown == []
    assert exhausted == [True]


def test_a_newly_merged_cluster_is_always_priority_for_later_rounds():
    # Mirrors the original: extra_ids is frozen at the start and never
    # grows to include a synthetic merged id, so a merged cluster always
    # counts as priority in subsequent pairing decisions.
    clusters = [
        HypothesisCluster(id=0, score=1.0, resolved="A", payload="A", is_priority=True),
        HypothesisCluster(id=1, score=1.0, resolved="B", payload="B", is_priority=True),
    ]

    def _score(resolved: str) -> float:
        return {"A": 1.0, "B": 1.0, "AB": 5.0}.get(resolved, -100.0)

    grown = grow_clusters(
        clusters, pool_fn=lambda a, b: "".join(sorted(a + b)), resolve_fn=lambda p: p, score_fn=_score,
        next_id_fn=_make_id_counter(100),
    )
    assert grown[0].is_priority is True
