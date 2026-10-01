"""
Convenience helpers that sit on top of backend.schemas.units, for code
that needs to go straight from a raw extracted measurement to a
normalized UnitValue in canonical units.

Keep actual conversion math in backend.schemas.units — this module is
just ergonomics (dispatch by unit family + FeetInches convenience).
"""

from __future__ import annotations

from backend.schemas.units import (
    AreaUnit,
    CanonicalUnit,
    FeetInches,
    LengthUnit,
    UnitValue,
    area_to_sqm,
    feet_inches_to_metres,
    length_to_metres,
)

_LENGTH_UNIT_VALUES = {u.value for u in LengthUnit}
_AREA_UNIT_VALUES = {u.value for u in AreaUnit}


def normalize_length(raw: UnitValue) -> UnitValue:
    """Convert a raw length UnitValue to canonical metres."""
    if raw.unit not in _LENGTH_UNIT_VALUES:
        raise ValueError(f"'{raw.unit}' is not a recognized length unit")
    metres = length_to_metres(raw.magnitude, LengthUnit(raw.unit))
    return UnitValue(magnitude=metres, unit=CanonicalUnit.METRE.value)


def normalize_feet_inches(feet: float, inches: float) -> UnitValue:
    """Convert a feet+inches measurement directly to canonical metres."""
    fi = FeetInches(feet=feet, inches=inches)
    return UnitValue(magnitude=feet_inches_to_metres(fi), unit=CanonicalUnit.METRE.value)


def normalize_area(raw: UnitValue) -> UnitValue:
    """Convert a raw area UnitValue to canonical square metres."""
    if raw.unit not in _AREA_UNIT_VALUES:
        raise ValueError(f"'{raw.unit}' is not a recognized area unit")
    sqm = area_to_sqm(raw.magnitude, AreaUnit(raw.unit))
    return UnitValue(magnitude=sqm, unit=CanonicalUnit.SQUARE_METRE.value)


def normalize(raw: UnitValue) -> UnitValue:
    """Dispatch to normalize_length or normalize_area based on raw.unit."""
    if raw.unit in _LENGTH_UNIT_VALUES:
        return normalize_length(raw)
    if raw.unit in _AREA_UNIT_VALUES:
        return normalize_area(raw)
    raise ValueError(f"Cannot normalize unrecognized unit '{raw.unit}'")
