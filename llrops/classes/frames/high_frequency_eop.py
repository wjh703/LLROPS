"""IERS 2010 high-frequency Earth-orientation corrections."""
from __future__ import annotations

from dataclasses import dataclass
import math

from llrops.base.epoch import Epoch, TimeScale
from llrops.base.epoch import utc2tt
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


def _require_utc(epoch: Epoch) -> Epoch:
    if not isinstance(epoch, Epoch):
        raise TypeError("High-frequency EOP requires an Epoch.")
    return epoch.require_scale(TimeScale.UTC, name="epoch_utc")


def _ut1_mjd(epoch_utc: Epoch, ut1_minus_utc_sec: float) -> float:
    epoch = _require_utc(epoch_utc)
    if isinstance(ut1_minus_utc_sec, bool) or not isinstance(ut1_minus_utc_sec, (int, float)):
        raise TypeError("ut1_minus_utc_sec must be a real scalar.")
    value = float(ut1_minus_utc_sec)
    if not math.isfinite(value):
        raise ValueError("ut1_minus_utc_sec must be finite.")
    return epoch.mjd + value / 86_400.0


def _tt_like_mjd(epoch: Epoch) -> float:
    if not isinstance(epoch, Epoch):
        raise TypeError("Libration EOP requires a TT or TDB Epoch.")
    if epoch.scale not in (TimeScale.TT, TimeScale.TDB):
        raise ValueError("libration EOP requires a TT or TDB Epoch.")
    return epoch.mjd


def ocean_tide_correction(
    epoch_utc: Epoch,
    *,
    ut1_minus_utc_sec: float = 0.0,
) -> HighFrequencyEopCorrection:
    """Return ORTHO_EOP corrections at the corresponding UT1 epoch."""
    delta_xp, delta_yp, delta_ut1 = _iers2010.ortho_eop(
        _ut1_mjd(epoch_utc, ut1_minus_utc_sec)
    )
    return HighFrequencyEopCorrection(
        ocean_delta_xp_arcsec=delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        ocean_delta_yp_arcsec=delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        ocean_delta_ut1_sec=delta_ut1 * _MICROSECOND_TO_SECOND,
    )


def libration_correction(epoch_tt: Epoch) -> HighFrequencyEopCorrection:
    """Return PMSDNUT2 and UTLIBR corrections at a TT/TDB epoch."""
    mjd = _tt_like_mjd(epoch_tt)
    delta_xp, delta_yp = _iers2010.pmsdnut2(mjd)
    delta_ut1, delta_lod = _iers2010.utlibr(mjd)
    return HighFrequencyEopCorrection(
        libration_delta_xp_arcsec=delta_xp * _MICROARCSECOND_TO_ARCSECOND,
        libration_delta_yp_arcsec=delta_yp * _MICROARCSECOND_TO_ARCSECOND,
        libration_delta_ut1_sec=delta_ut1 * _MICROSECOND_TO_SECOND,
        libration_delta_lod_sec_per_day=delta_lod * _MICROSECOND_TO_SECOND,
    )


def high_frequency_eop_correction(
    epoch_utc: Epoch,
    *,
    ut1_minus_utc_sec: float = 0.0,
) -> HighFrequencyEopCorrection:
    """Return combined EOP corrections for an explicit UTC observation epoch."""
    epoch_utc = _require_utc(epoch_utc)
    ocean = ocean_tide_correction(
        epoch_utc,
        ut1_minus_utc_sec=ut1_minus_utc_sec,
    )
    libration = libration_correction(utc2tt(epoch_utc))
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
