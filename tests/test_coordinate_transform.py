from __future__ import annotations

import pytest

from backend.cv_extraction import coordinates


def test_points_per_pixel_at_72_dpi_is_1():
    assert coordinates.points_per_pixel(72.0) == pytest.approx(1.0)


def test_points_per_pixel_at_300_dpi():
    assert coordinates.points_per_pixel(300.0) == pytest.approx(72.0 / 300.0)


def test_points_per_pixel_rejects_non_positive_dpi():
    with pytest.raises(ValueError):
        coordinates.points_per_pixel(0)


def test_pixel_to_page_point_roundtrip():
    dpi = 150.0
    point = coordinates.pixel_to_page_point(150, 300, dpi)
    assert point.x == pytest.approx(150 * (72.0 / dpi))
    assert point.y == pytest.approx(300 * (72.0 / dpi))


def test_pixel_bbox_to_page_bbox_orders_min_max():
    bbox = coordinates.pixel_bbox_to_page_bbox(100, 200, 10, 20, dpi=72.0)
    assert bbox.min_x == 10
    assert bbox.min_y == 20
    assert bbox.max_x == 100
    assert bbox.max_y == 200


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 0), (90, 90), (135, 90), (180, 180), (270, 270), (359, 270), (450, 90), (-90, 270)],
)
def test_normalize_rotation_folds_onto_quadrants(raw, expected):
    assert coordinates.normalize_rotation(raw) == expected


def test_build_transform_record_uses_settings_fallback():
    record = coordinates.build_transform_record(
        page=0, dpi=200.0, rotation_degrees=90, page_width_pts=612, page_height_pts=792
    )
    assert record.raster_dpi == 200.0
    assert record.rotation_degrees == 90
    assert record.points_per_pixel == pytest.approx(72.0 / 200.0)
    assert record.provisional_points_per_metre > 0


def test_build_transform_record_honors_explicit_scale():
    record = coordinates.build_transform_record(
        page=1,
        dpi=150.0,
        rotation_degrees=0,
        page_width_pts=595,
        page_height_pts=842,
        provisional_points_per_metre=100.0,
    )
    assert record.provisional_points_per_metre == 100.0
