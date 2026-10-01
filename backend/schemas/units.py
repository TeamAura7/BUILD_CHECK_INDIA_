"""
Unit contract for BUILDCheck India.

Canonical (normalized) units used EVERYWHERE downstream of extraction:

    length      -> metres (m)
    area        -> square metres (sq_m)
    percentage  -> % (0-100 float)
    FAR         -> ratio (dimensionless float)

Raw extracted values may arrive in any of the LengthUnit / AreaUnit members
below. Conversion to canonical units happens ONCE, at the normalization
boundary, and both the raw and normalized values are preserved (see
schemas/evidence.py -> ValueField).

Feet+inches values (e.g. 12' 6") are represented as a FeetInches struct
before conversion so no information is lost in the raw record.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

MM_PER_M = 1000.0
CM_PER_M = 100.0
INCHES_PER_FOOT = 12.0
M_PER_FOOT = 0.3048
SQFT_PER_SQM = 10.7639104167


class LengthUnit(str, Enum):
    MM = "mm"
    CM = "cm"
    M = "m"
    FT = "ft"
    INCH = "in"
    FT_IN = "ft_in"  # composite feet+inches, see FeetInches


class AreaUnit(str, Enum):
    SQ_FT = "sq_ft"
    SQ_M = "sq_m"


class CanonicalUnit(str, Enum):
    """Units every NormalizedPlan field is expressed in after normalization."""

    METRE = "m"
    SQUARE_METRE = "sq_m"
    PERCENTAGE = "%"
    RATIO = "ratio"


class FeetInches(BaseModel):
    """Raw representation of a feet+inches measurement, e.g. 12' 6"."""

    feet: float = Field(..., ge=0)
    inches: float = Field(..., ge=0, lt=12)

    def to_metres(self) -> float:
        total_feet = self.feet + (self.inches / INCHES_PER_FOOT)
        return round(total_feet * M_PER_FOOT, 6)

    def __str__(self) -> str:  # human-readable, useful in evidence text
        return f"{self.feet:g}' {self.inches:g}\""


def length_to_metres(value: float, unit: LengthUnit) -> float:
    """Convert a scalar length in `unit` to canonical metres."""
    if unit == LengthUnit.MM:
        return round(value / MM_PER_M, 6)
    if unit == LengthUnit.CM:
        return round(value / CM_PER_M, 6)
    if unit == LengthUnit.M:
        return round(value, 6)
    if unit == LengthUnit.FT:
        return round(value * M_PER_FOOT, 6)
    if unit == LengthUnit.INCH:
        return round((value / INCHES_PER_FOOT) * M_PER_FOOT, 6)
    if unit == LengthUnit.FT_IN:
        raise ValueError(
            "LengthUnit.FT_IN requires a FeetInches value; use feet_inches_to_metres()."
        )
    raise ValueError(f"Unsupported length unit: {unit}")


def feet_inches_to_metres(fi: FeetInches) -> float:
    return fi.to_metres()


def area_to_sqm(value: float, unit: AreaUnit) -> float:
    """Convert a scalar area in `unit` to canonical square metres."""
    if unit == AreaUnit.SQ_M:
        return round(value, 6)
    if unit == AreaUnit.SQ_FT:
        return round(value / SQFT_PER_SQM, 6)
    raise ValueError(f"Unsupported area unit: {unit}")


class UnitValue(BaseModel):
    """
    A single scalar measurement paired with its unit.

    Used as the `raw_value` / `normalized_value` payload inside ValueField
    (see schemas/evidence.py). `normalized_value.unit` is always one of
    CanonicalUnit; `raw_value.unit` may be any LengthUnit / AreaUnit / str.
    """

    magnitude: float
    unit: str  # LengthUnit | AreaUnit | CanonicalUnit value, kept as str for flexibility

    @field_validator("magnitude")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf guard
            raise ValueError("magnitude must be a finite number")
        return v
