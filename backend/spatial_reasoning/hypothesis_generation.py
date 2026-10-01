"""
Format-agnostic agglomerative hypothesis growth (Architecture V2, Phase 4/5).

See ARCHITECTURE_V2.md, Deliverable C.2 item 3 and Deliverable E's migration
table entry for `dxf_extractor.py`. This module is a faithful EXTRACTION of
`backend.cv_extraction.dxf_extractor._merge_candidate_region_pairs`'s
algorithm: single-linkage agglomerative growth where "does re-resolving the
pooled fragments score better than BOTH of its own inputs" is the merge
criterion, never spatial proximity/bbox-overlap alone (that was tried and
reverted in this project's own history -- see `_merge_candidate_region_pairs`'s
docstring). The control flow (budget-bounded rounds, priority-vs-extra pair
ordering, "keep the single best pair per round") is copied exactly; what
changed is that the four DXF-specific decisions (how to pool two fragments,
how to resolve a pooled fragment into a candidate, how to score it, whether
an area-sanity/evidence-veto gate blocks an otherwise-winning merge) are
injected as callables instead of being hardcoded calls to
`_resolve_plot_building_road`/`_score_region_resolution`/inline area and
evidence checks. This is what makes the same growth algorithm usable by any
future caller (a PDF-side hypothesis generator, in a later phase) without a
second, independently-drifting copy of the loop itself.

`dxf_extractor._merge_candidate_region_pairs` now delegates to `grow_clusters`
below, translating its own DXF-specific region/evidence bookkeeping into the
generic callables -- a behavior-preserving refactor (move, not rewrite): the
existing `test_dxf_extractor.py` regression suite (including the exact
warning-text assertion in `test_merge_accepts_two_regions_that_together_
resolve_plot_and_building`) is the proof the extraction changed nothing
observable.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

Pool = Callable[[Any, Any], Any]
Resolve = Callable[[Any], Any]
Score = Callable[[Any], float]
AreaSane = Callable[[Any, Any, Any], bool]
Veto = Callable[["HypothesisCluster", "HypothesisCluster", Any], Optional[str]]
OnMergeAccepted = Callable[["HypothesisCluster", "HypothesisCluster", "HypothesisCluster"], None]
OnMergeVetoed = Callable[["HypothesisCluster", "HypothesisCluster", float, str], None]
OnBudgetExhausted = Callable[[], None]


@dataclass
class HypothesisCluster:
    """
    One node in the growth process: either an original candidate or an
    already-merged pool of earlier ones.

    `payload` and `resolved` are deliberately opaque (`Any`) -- this module
    has no opinion on what a "fragment" or a "resolved candidate" looks
    like for a given caller; DXF's wrapper supplies its own
    `_RawPoly`/region-index tuples, a future PDF caller could supply
    something else entirely.

    `is_priority` mirrors `_merge_candidate_region_pairs`'s "top-N by own
    score, plus evidence-carrying extras appended after" split: priority
    pairs are always tried before any pair touching a non-priority
    cluster, so an extra can never starve out an already-working
    priority-only merge. A newly merged cluster is always priority for
    subsequent rounds (mirrors the original: `extra_ids` is frozen at the
    start and never grows to include a synthetic merged id).
    """

    id: int
    score: float
    resolved: Any
    payload: Any
    is_priority: bool = True


def _pair_sort_key(clusters: list[HypothesisCluster], i: int, j: int) -> tuple[bool, float]:
    a, b = clusters[i], clusters[j]
    return (a.is_priority and b.is_priority, a.score + b.score)


def grow_clusters(
    initial_clusters: Sequence[HypothesisCluster],
    *,
    pool_fn: Pool,
    resolve_fn: Resolve,
    score_fn: Score,
    next_id_fn: Callable[[], int],
    area_sane_fn: Optional[AreaSane] = None,
    veto_fn: Optional[Veto] = None,
    time_budget_seconds: float = float("inf"),
    on_merge_accepted: Optional[OnMergeAccepted] = None,
    on_merge_vetoed: Optional[OnMergeVetoed] = None,
    on_budget_exhausted: Optional[OnBudgetExhausted] = None,
) -> list[HypothesisCluster]:
    """
    Grow `initial_clusters` by repeatedly pooling whichever pair improves on
    BOTH of its own inputs, for as many rounds as improvement keeps
    happening, and return every accepted merge along the way (not just the
    final one) -- callers compare these against the original candidates to
    pick an overall winner, exactly as `_resolve_via_regions` already does
    with `_merge_candidate_region_pairs`'s output today.

    A pair is accepted only when ALL of the following hold, matching
    `_merge_candidate_region_pairs` exactly:
      1. `resolve_fn(pool_fn(a.payload, b.payload))`, then `score_fn(...)`,
         produces a score strictly greater than both `a.score` and `b.score`.
      2. `area_sane_fn(a.resolved, b.resolved, merged_resolved)` is True, if
         `area_sane_fn` was given (omit it if the caller's resolved shape
         has no notion of "area").
      3. `veto_fn(a, b, merged_resolved)` returns None (no veto reason), if
         `veto_fn` was given.

    Stops when no pair improves on both its inputs, when fewer than two
    clusters remain, or when `time_budget_seconds` of wall-clock time has
    been spent testing pairs -- whatever has already been found by then is
    kept, never discarded for running out of time.
    """
    clusters = list(initial_clusters)
    if len(clusters) < 2:
        return []

    merged: list[HypothesisCluster] = []
    budget_start = _time.time()
    budget_exhausted = False

    while len(clusters) >= 2 and not budget_exhausted:
        import itertools

        pairs = sorted(
            itertools.combinations(range(len(clusters)), 2),
            key=lambda ij: _pair_sort_key(clusters, *ij),
            reverse=True,
        )
        best_this_round: Optional[tuple[float, int, int, HypothesisCluster]] = None

        for i, j in pairs:
            if _time.time() - budget_start >= time_budget_seconds:
                if on_budget_exhausted is not None:
                    on_budget_exhausted()
                budget_exhausted = True
                break

            a, b = clusters[i], clusters[j]
            pooled_payload = pool_fn(a.payload, b.payload)
            merged_resolved = resolve_fn(pooled_payload)
            merged_score = score_fn(merged_resolved)

            area_sane = area_sane_fn(a.resolved, b.resolved, merged_resolved) if area_sane_fn is not None else True

            if merged_score > a.score and merged_score > b.score and area_sane:
                veto_reason = veto_fn(a, b, merged_resolved) if veto_fn is not None else None
                if veto_reason is not None:
                    if on_merge_vetoed is not None:
                        on_merge_vetoed(a, b, merged_score, veto_reason)
                    continue
                if best_this_round is None or merged_score > best_this_round[0]:
                    new_cluster = HypothesisCluster(
                        id=next_id_fn(), score=merged_score, resolved=merged_resolved,
                        payload=pooled_payload, is_priority=True,
                    )
                    best_this_round = (merged_score, i, j, new_cluster)

        if best_this_round is None:
            break  # no pair improved on both its inputs this round -- growth is done

        _score, i, j, new_cluster = best_this_round
        a, b = clusters[i], clusters[j]
        if on_merge_accepted is not None:
            on_merge_accepted(a, b, new_cluster)
        merged.append(new_cluster)
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)] + [new_cluster]

    return merged


__all__ = ["HypothesisCluster", "grow_clusters"]
