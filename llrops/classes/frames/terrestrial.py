"""ITRF/ITRS and GCRS transforms driven by explicit EOP and ERFA."""
from __future__ import annotations

from typing import Sequence

import erfa
import numpy as np

from llrops.base.epoch import Epoch, TimeScale, utc2tt
from llrops.base.array_validation import vector3

from .earth_orientation import EarthOrientation
from .iers2010_eop import high_frequency_eop_correction

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
        """Return an IERS 2010 GCRS-to-ITRF rotation matrix."""
        epoch = self._utc(epoch_utc)
        tt = utc2tt(epoch)

        high_frequency = high_frequency_eop_correction(epoch.mjd)
        dut1_s = self.earth_orientation.ut1_minus_utc_sec(epoch) + high_frequency.ut1_sec
        ut1_jd1, ut1_jd2 = erfa.utcut1(epoch.jd1, epoch.jd2, dut1_s)

        pole = self.earth_orientation.polar_motion(epoch)
        xp = (pole.xp_arcsec + high_frequency.xp_arcsec) * _ARCSEC_TO_RAD
        yp = (pole.yp_arcsec + high_frequency.yp_arcsec) * _ARCSEC_TO_RAD

        offsets = self.earth_orientation.celestial_pole_offsets(epoch)
        x, y, s = erfa.xys06a(tt.jd1, tt.jd2)
        x += offsets.dx_arcsec * _ARCSEC_TO_RAD
        y += offsets.dy_arcsec * _ARCSEC_TO_RAD
        celestial_to_intermediate = erfa.c2ixys(x, y, s)

        era = erfa.era00(ut1_jd1, ut1_jd2)
        tio_locator = erfa.sp00(tt.jd1, tt.jd2)
        polar_motion = erfa.pom00(xp, yp, tio_locator)
        matrix = np.asarray(
            erfa.c2tcio(celestial_to_intermediate, era, polar_motion),
            dtype=float,
        )
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("ERFA returned an invalid celestial-to-terrestrial matrix.")
        return matrix

    def gcrs2itrf(self, position_gcrs_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        matrix = self.celestial_to_terrestrial_matrix(epoch_utc)
        return matrix @ vector3(position_gcrs_m, name="position_gcrs_m")

    def itrf2gcrs(self, position_itrf_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        matrix = self.celestial_to_terrestrial_matrix(epoch_utc)
        return matrix.T @ vector3(position_itrf_m, name="position_itrf_m")


__all__ = ["TerrestrialFrameTransform"]
