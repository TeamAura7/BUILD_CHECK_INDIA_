from __future__ import annotations

import pytest

from backend.schemas.units import AreaUnit, FeetInches, LengthUnit, UnitValue
from backend.tools.unit_conversion import (
    normalize,
    normalize_area,
    normalize_feet_inches,
    normalize_length,
)


def test_mm_to_metres():
    assert normalize_length(UnitValue(magnitude=1000, unit="mm")).magnitude == pytest.approx(1.0)


def test_cm_to_metres():
    assert normalize_length(UnitValue(magnitude=250, unit="cm")).magnitude == pytest.approx(2.5)


def test_feet_to_metres():
    result = normalize_length(UnitValue(magnitude=10, unit="ft"))
    assert result.magnitude == pytest.approx(3.048)


def test_inches_to_metres():
    result = normalize_length(UnitValue(magnitude=12, unit="in"))
    assert result.magnitude == pytest.approx(0.3048)


def test_feet_inches_to_metres():
    result = normalize_feet_inches(feet=12, inches=6)
    # 12'6" = 12.5 ft = 3.81 m
    assert result.magnitude == pytest.approx(3.81, abs=1e-2)


def test_feet_inches_rejects_out_of_range_inches():
    with pytest.raises(Exception):
        FeetInches(feet=5, inches=12)  # inches must be < 12


def test_sqft_to_sqm():
    result = normalize_area(UnitValue(magnitude=1000, unit="sq_ft"))
    assert result.magnitude == pytest.approx(92.903, abs=1e-2)


def test_sqm_passthrough():
    result = normalize_area(UnitValue(magnitude=50, unit="sq_m"))
    assert result.magnitude == pytest.approx(50.0)


def test_normalize_dispatches_length():
    result = normalize(UnitValue(magnitude=100, unit="cm"))
    assert result.unit == "m"
    assert result.magnitude == pytest.approx(1.0)


def test_normalize_dispatches_area():
    result = normalize(UnitValue(magnitude=100, unit="sq_ft"))
    assert result.unit == "sq_m"


def test_normalize_rejects_unknown_unit():
    with pytest.raises(ValueError):
        normalize(UnitValue(magnitude=5, unit="furlongs"))


def test_unit_value_rejects_nan():
    with pytest.raises(Exception):
        UnitValue(magnitude=float("nan"), unit="m")
