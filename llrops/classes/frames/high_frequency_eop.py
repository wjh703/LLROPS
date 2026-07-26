"""IERS 2010 high-frequency Earth-orientation corrections."""
from __future__ import annotations

from dataclasses import dataclass

from llrops.base.epoch import Epoch, TimeScale
from llrops import _iers2010


_MICROARCSECOND_TO_ARCSECOND = 1.0e-6
_MICROSECOND_TO_SECOND = 1.0e-6


@dataclass(frozen=True, slots=True)
class HighFrequencyEopCorrection:
    ocean_delta_xp_arcsec: float = 0.0
    ocean_delta_yp_arcsec: float = 0.0
    ocean_delta_ut1_sec: float = 0.0
    libration_delta_xp_arcsec: float = 0.0
    libration_delta_yp_arcsec: float = 0.0
    libration_delta_ut1_sec: float = 0.0
    libration_delta_lod_sec_per_day: float = 0.0

    @property
    def xp_arcsec(self) -> float:
        return self.ocean_delta_xp_arcsec + self.libration_delta_xp_arcsec

    @property
    def yp_arcsec(self) -> float:
        return self.ocean_delta_yp_arcsec + self.libration_delta_yp_arcsec

    @property
    def ut1_sec(self) -> float:
        return self.ocean_delta_ut1_sec + self.libration_delta_ut1_sec


def _utc_mjd(epoch: Epoch) -> float:
    if not isinstance(epoch, Epoch):
        raise TypeError("High-frequency EOP requires an Epoch.")
    return epoch.require_scale(TimeScale.UTC, name="epoch_utc").mjd


def ocean_tide_correction(epoch: Epoch) -> HighFrequencyEopCorrection:
    """Return official ORTHO_EOP ocean-tide corrections."""
    delta_xp, delta_yp, delta_ut1 = _iers2010.ortho_eop(_utc_mjd(epoch))
    return HighFrequencyEopCorrection(
        ocean_delta_xp_arcsec=delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        ocean_delta_yp_arcsec=delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        ocean_delta_ut1_sec=delta_ut1 * _MICROSECOND_TO_SECOND,
    )


def libration_correction(epoch: Epoch) -> HighFrequencyEopCorrection:
    """Return official PMSDNUT2 and UTLIBR corrections."""
    mjd = _utc_mjd(epoch)
    delta_xp, delta_yp = _iers2010.pmsdnut2(mjd)
    delta_ut1, delta_lod = _iers2010.utlibr(mjd)
    return HighFrequencyEopCorrection(
        libration_delta_xp_arcsec=delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        libration_delta_yp_arcsec=delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        libration_delta_ut1_sec=delta_ut1 * _MICROSECOND_TO_SECOND,
        libration_delta_lod_sec_per_day=delta_lod * _MICROSECOND_TO_SECOND,
    )


def high_frequency_eop_correction(epoch: Epoch) -> HighFrequencyEopCorrection:
    ocean = ocean_tide_correction(epoch)
    libration = libration_correction(epoch)
    return HighFrequencyEopCorrection(
        ocean_delta_xp_arcsec=ocean.ocean_delta_xp_arcsec,
        ocean_delta_yp_arcsec=ocean.ocean_delta_yp_arcsec,
        ocean_delta_ut1_sec=ocean.ocean_delta_ut1_sec,
        libration_delta_xp_arcsec=libration.libration_delta_xp_arcsec,
        libration_delta_yp_arcsec=libration.libration_delta_yp_arcsec,
        libration_delta_ut1_sec=libration.libration_delta_ut1_sec,
        libration_delta_lod_sec_per_day=libration.libration_delta_lod_sec_per_day,
    )


__all__ = [
    "HighFrequencyEopCorrection",
    "high_frequency_eop_correction",
    "libration_correction",
    "ocean_tide_correction",
]
