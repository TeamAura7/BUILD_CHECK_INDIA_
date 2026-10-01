"""Tests for `backend.spatial_reasoning.scale`."""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.geometry import Line, Point
from backend.spatial_reasoning.scale import dimension_length_metres, estimate_scale
from tests.fixtures.geometry_builders import dim


def test_dimension_length_metres_converts_units():
    assert dimension_length_metres(dim(1000, "mm")) == 1.0
    assert dimension_length_metres(dim(100, "cm")) == 1.0
    assert dimension_length_metres(dim(1, "ft")) is not None
    assert abs(dimension_length_metres(dim(1, "ft")) - 0.3048) < 1e-6
    assert dimension_length_metres(dim(5, "m")) == 5.0


def test_dimension_length_metres_unknown_unit_is_none():
    assert dimension_length_metres(dim(5, "sqm")) is None


def test_estimate_scale_falls_back_with_low_confidence_when_no_samples():
    est = estimate_scale([])
    assert est.confidence == ConfidenceLevel.LOW
    assert est.samples_used == 0
    assert est.points_per_metre > 0


def test_estimate_scale_high_confidence_with_consistent_samples():
    # Three dimensions, all implying the same points-per-metre (40).
    dims = [
        dim(10.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=400, y=0))),
        dim(5.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=200, y=0))),
        dim(2.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=80, y=0))),
    ]
    est = estimate_scale(dims)
    assert est.confidence == ConfidenceLevel.HIGH
    assert abs(est.points_per_metre - 40.0) < 0.5
    assert est.samples_used == 3


def test_estimate_scale_rejects_outlier_sample():
    dims = [
        dim(10.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=400, y=0))),  # 40 ppm
        dim(5.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=201, y=0))),  # ~40.2 ppm
        dim(4.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=159, y=0))),  # ~39.75 ppm
        dim(1.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=900, y=0))),  # 900 ppm, wild outlier
    ]
    est = estimate_scale(dims)
    assert est.samples_rejected >= 1
    assert abs(est.points_per_metre - 40.0) < 2.0


def test_estimate_scale_ignores_dimensions_without_geometry_line():
    dims = [dim(10.0, "m", line=None)]
    est = estimate_scale(dims)
    assert est.samples_used == 0
    assert est.confidence == ConfidenceLevel.LOW
