"""IERS 2010 solid-Earth tide station displacement."""
from __future__ import annotations

import numpy as np

from lunarops import _iers2010
from lunarops.base.epoch import Epoch, TimeScale
from lunarops.classes.frames import ReferenceFrameSystem

from .base import StationDisplacementInput


class Iers2010SolidEarthTide:
    """Solid-Earth tide displacement from the official IERS routine.

    ``DEHANTTIDEINEL`` requires geocentric ITRF vectors for the station, Sun,
    and Moon. The celestial vectors are derived from the configured ephemeris
    and the same frame system used by the observation model.
    """

    def __init__(self, frames: ReferenceFrameSystem) -> None:
        if not isinstance(frames, ReferenceFrameSystem):
            raise TypeError("frames must be a ReferenceFrameSystem.")
        self.frames = frames

    @staticmethod
    def _utc_calendar(epoch: Epoch) -> tuple[int, int, int, float]:
        epoch.require_scale(TimeScale.UTC, name="epoch_utc")
        civil = epoch.to_datetime()
        fractional_hour = (
            civil.hour
            + civil.minute / 60.0
            + (civil.second + civil.microsecond / 1.0e6) / 3600.0
        )
        return civil.year, civil.month, civil.day, fractional_hour

    def _body_itrf_m(self, body: str, epoch_utc: Epoch, epoch_tdb: Epoch) -> np.ndarray:
        position_bcrs_m = self.frames.ephemeris.body_position_bcrs(body, epoch_tdb)
        position_gcrs_m = self.frames.bcrs2gcrs(position_bcrs_m, epoch_tdb)
        position_itrf_m = self.frames.gcrs2itrf(position_gcrs_m, epoch_utc)
        value = np.asarray(position_itrf_m, dtype=float).reshape(3)
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Ephemeris/frame conversion returned non-finite {body} coordinates.")
        return value

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        epoch_utc = data.epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        epoch_tdb = self.frames.time_converter.convert(epoch_utc, TimeScale.TDB)
        sun_itrf_m = self._body_itrf_m("SUN", epoch_utc, epoch_tdb)
        moon_itrf_m = self._body_itrf_m("MOON", epoch_utc, epoch_tdb)
        year, month, day, fractional_hour = self._utc_calendar(epoch_utc)

        displacement = np.asarray(
            _iers2010.dehanttideinel(
                data.station_itrf_m,
                year,
                month,
                day,
                fractional_hour,
                sun_itrf_m,
                moon_itrf_m,
            ),
            dtype=float,
        )
        if displacement.size != 3 or not np.all(np.isfinite(displacement)):
            raise RuntimeError("DEHANTTIDEINEL returned an invalid displacement vector.")
        return displacement.reshape(3)
