"""ITRF/ITRS and GCRS transforms driven by explicit EOP and ERFA."""
from __future__ import annotations

from typing import Sequence

import erfa
import numpy as np

from llrops.base.constants import SECONDS_PER_DAY
from llrops.base.epoch import Epoch, TimeScale, utc2tt
from llrops.base.validation import vector3

from .earth_orientation import EarthOrientation

_ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)

class TerrestrialFrameTransform:
    def __init__(self, earth_orientation: EarthOrientation) -> None:
        if not isinstance(earth_orientation, EarthOrientation):
            raise TypeError("earth_orientation must implement EarthOrientation.")
        self.earth_orientation = earth_orientation

    @staticmethod
    def _utc(value: Epoch) -> Epoch:
        if not isinstance(value, Epoch):
            raise TypeError("Frame transforms require an Epoch.")
        return value.require_scale(TimeScale.UTC, name="epoch_utc")

    def celestial_to_terrestrial_matrix(self, epoch_utc: Epoch) -> np.ndarray:
        """Return ERFA's GCRS-to-ITRF rotation matrix for a UTC epoch."""
        epoch = self._utc(epoch_utc)
        tt = utc2tt(epoch)
        dut1_s = self.earth_orientation.ut1_minus_utc_sec(epoch)
        ut1_jd1 = epoch.jd1
        ut1_jd2 = epoch.jd2 + dut1_s / SECONDS_PER_DAY
        pole = self.earth_orientation.polar_motion(epoch)
        xp = pole.xp_arcsec * _ARCSEC_TO_RAD
        yp = pole.yp_arcsec * _ARCSEC_TO_RAD
        matrix = np.asarray(erfa.c2t06a(tt.jd1, tt.jd2, ut1_jd1, ut1_jd2, xp, yp), dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("ERFA c2t06a returned an invalid rotation matrix.")
        return matrix

    def gcrs2itrf(self, position_gcrs_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        matrix = self.celestial_to_terrestrial_matrix(epoch_utc)
        return matrix @ vector3(position_gcrs_m, name="position_gcrs_m")

    def itrf2gcrs(self, position_itrf_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        matrix = self.celestial_to_terrestrial_matrix(epoch_utc)
        return matrix.T @ vector3(position_itrf_m, name="position_itrf_m")


__all__ = ["TerrestrialFrameTransform"]
