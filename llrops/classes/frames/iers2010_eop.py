"""IERS 2010 high-frequency Earth-orientation corrections."""
from __future__ import annotations

from dataclasses import dataclass

from llrops import _iers2010


_MICRO = 1.0e-6


@dataclass(frozen=True, slots=True)
class HighFrequencyEopCorrection:
    xp_arcsec: float
    yp_arcsec: float
    ut1_sec: float


def ocean_tide_correction(mjd: float) -> HighFrequencyEopCorrection:
    """Return official ORTHO_EOP ocean-tide corrections."""
    delta_xp, delta_yp, delta_ut1 = _iers2010.ortho_eop(float(mjd))
    return HighFrequencyEopCorrection(
        delta_xp * _MICRO,
        delta_yp * _MICRO,
        delta_ut1 * _MICRO,
    )


def libration_correction(mjd: float) -> HighFrequencyEopCorrection:
    """Return official PMSDNUT2 and UTLIBR corrections."""
    delta_xp, delta_yp = _iers2010.pmsdnut2(float(mjd))
    delta_ut1, _delta_lod = _iers2010.utlibr(float(mjd))
    return HighFrequencyEopCorrection(
        delta_xp * _MICRO,
        delta_yp * _MICRO,
        delta_ut1 * _MICRO,
    )


def high_frequency_eop_correction(mjd: float) -> HighFrequencyEopCorrection:
    ocean = ocean_tide_correction(mjd)
    libration = libration_correction(mjd)
    return HighFrequencyEopCorrection(
        ocean.xp_arcsec + libration.xp_arcsec,
        ocean.yp_arcsec + libration.yp_arcsec,
        ocean.ut1_sec + libration.ut1_sec,
    )


__all__ = [
    "HighFrequencyEopCorrection",
    "high_frequency_eop_correction",
    "libration_correction",
    "ocean_tide_correction",
]
