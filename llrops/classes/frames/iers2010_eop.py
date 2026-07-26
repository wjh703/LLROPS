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


def _native_ortho_eop(mjd: float) -> tuple[float, float, float]:
    """Return ORTHO_EOP values converted from micro-units to SI units."""
    delta_xp, delta_yp, delta_ut1 = _iers2010.llrops_ortho_eop(float(mjd))
    return delta_xp * _MICRO, delta_yp * _MICRO, delta_ut1 * _MICRO


def _native_pmsdnut2(mjd: float) -> tuple[float, float]:
    delta_xp, delta_yp = _iers2010.llrops_pmsdnut2(float(mjd))
    return delta_xp * _MICRO, delta_yp * _MICRO


def _native_utlibr(mjd: float) -> tuple[float, float]:
    delta_ut1, delta_lod = _iers2010.llrops_utlibr(float(mjd))
    return delta_ut1 * _MICRO, delta_lod * _MICRO


def _fundamental_arguments(mjd: float) -> tuple[float, float, float, float, float]:
    """Return official FUNDARG values for an MJD expressed in TT-like time."""
    t = (float(mjd) - 51_544.5) / 36_525.0
    return tuple(float(value) for value in _iers2010.fundarg(t))


def ocean_tide_correction(mjd: float) -> HighFrequencyEopCorrection:
    """Return official ORTHO_EOP ocean-tide corrections."""
    delta_xp, delta_yp, delta_ut1 = _native_ortho_eop(mjd)
    return HighFrequencyEopCorrection(delta_xp, delta_yp, delta_ut1)


def libration_correction(mjd: float) -> HighFrequencyEopCorrection:
    """Return official PMSDNUT2 and UTLIBR corrections."""
    delta_xp, delta_yp = _native_pmsdnut2(mjd)
    delta_ut1, _delta_lod = _native_utlibr(mjd)
    return HighFrequencyEopCorrection(delta_xp, delta_yp, delta_ut1)


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
