"""Reference-ellipsoid constants used by terrestrial-frame geometry."""
from __future__ import annotations

WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_FIRST_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)

__all__ = [
    "WGS84_FIRST_ECCENTRICITY_SQUARED",
    "WGS84_FLATTENING",
    "WGS84_SEMI_MAJOR_AXIS_M",
]
