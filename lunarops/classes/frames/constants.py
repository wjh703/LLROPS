"""Reference-ellipsoid constants used by terrestrial-frame geometry."""
from __future__ import annotations

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

__all__ = ["WGS84_A_M", "WGS84_F", "WGS84_E2"]
