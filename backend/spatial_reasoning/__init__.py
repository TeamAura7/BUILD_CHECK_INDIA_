"""
Phase 3 — Computational geometry & spatial reasoning.

Turns Phase 2's `ExtractionResult` (raw candidates + unresolved
dimensions, in page-space) into the single, resolved Phase 1
`NormalizedPlan` contract (schemas/normalized_plan.py), in metric
plan space.

Entry point: `backend.spatial_reasoning.pipeline.build_normalized_plan`.

Sub-modules (see ARCHITECTURE.md's "SPATIAL REASONING" stage):

    geometry_utils          generic polygon/segment geometry helpers
    scale                   page-points -> metres scale estimation
    dimension_classification semantic labeling of raw dimension candidates
    plot_resolution          plot candidate scoring/selection
    building_resolution      building candidate scoring/selection
    road_access               road + access/gate/entry evidence
    front_side               front/rear/left/right edge resolution
    setbacks                  setback derivation (geometry + dimension evidence)
    areas                     plot/building/coverage/FAR computation
    evidence_reconciliation   cross-source agreement/conflict detection
    consistency                physical plausibility checks
    pipeline                   orchestration -> NormalizedPlan
"""
