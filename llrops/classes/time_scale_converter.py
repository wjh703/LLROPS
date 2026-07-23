"""Explicit UTC/TT/TDB conversion services.

UTC<->TT is handled by the ERFA-backed routines in
``llrops.base.epoch``.  TT<->TDB remains in ``classes`` because it depends on the
configured ephemeris target-16 table and, optionally, the topocentric
``v_E dot X / c^2`` term.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from llrops.base.constants import C2
from llrops.base.epoch import Epoch, TimeScale, tt2utc as _tt2utc, utc2tt as _utc2tt
from llrops.base.array_validation import vector3


class TimeScaleConverter:
    def __init__(
        self,
        ephemeris: object | None = None,
        max_iterations: int = 6,
        tolerance_s: float = 1.0e-12,
    ) -> None:
        iterations = int(max_iterations)
        tolerance = float(tolerance_s)
        if iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if tolerance <= 0.0:
            raise ValueError("tolerance_s must be positive.")
        self.ephemeris = ephemeris
        self.max_iterations = iterations
        self.tolerance_s = tolerance

    def _require_ephemeris(self):
        if self.ephemeris is None:
            raise RuntimeError("TT/TDB conversion requires an ephemeris.")
        return self.ephemeris

    def utc2tt(self, epoch: Epoch) -> Epoch:
        epoch.require_scale(TimeScale.UTC)
        return _utc2tt(epoch)

    def tt2utc(self, epoch: Epoch) -> Epoch:
        epoch.require_scale(TimeScale.TT)
        return _tt2utc(epoch)

    def tdb_minus_tt_sec(
        self,
        epoch_tdb: Epoch,
        *,
        station_gcrs_m: Sequence[float] | None = None,
    ) -> float:
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        ephemeris = self._require_ephemeris()
        geocentric = ephemeris.tdb_minus_tt_sec(epoch_tdb)
        if geocentric is None:
            raise RuntimeError(
                "The configured ephemeris does not provide a TDB-TT table."
            )
        correction = 0.0
        if station_gcrs_m is not None:
            station = vector3(station_gcrs_m, name="station_gcrs_m")
            earth_velocity = ephemeris.body_state_bcrs(
                "EARTH",
                epoch_tdb,
            ).velocity_mps
            correction = float(np.dot(earth_velocity, station)) / C2
        return float(geocentric) + correction

    def tdb2tt(
        self,
        epoch_tdb: Epoch,
        *,
        station_gcrs_m: Sequence[float] | None = None,
    ) -> Epoch:
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        delta_s = self.tdb_minus_tt_sec(
            epoch_tdb,
            station_gcrs_m=station_gcrs_m,
        )
        shifted = epoch_tdb.shifted(-delta_s)
        return Epoch(shifted.jd1, shifted.jd2, TimeScale.TT)

    def tt2tdb(
        self,
        epoch_tt: Epoch,
        *,
        station_gcrs_m: Sequence[float] | None = None,
    ) -> Epoch:
        epoch_tt.require_scale(TimeScale.TT, name="epoch_tt")
        current = Epoch(epoch_tt.jd1, epoch_tt.jd2, TimeScale.TDB)
        for _ in range(self.max_iterations):
            delta_s = self.tdb_minus_tt_sec(
                current,
                station_gcrs_m=station_gcrs_m,
            )
            shifted = epoch_tt.shifted(delta_s)
            updated = Epoch(shifted.jd1, shifted.jd2, TimeScale.TDB)
            if abs(current.seconds_until(updated)) < self.tolerance_s:
                return updated
            current = updated
        return current

    def convert(
        self,
        epoch: Epoch,
        scale: TimeScale | str,
        *,
        station_gcrs_m: Sequence[float] | None = None,
    ) -> Epoch:
        target = TimeScale.parse(scale)
        if epoch.scale is target:
            return epoch
        if epoch.scale is TimeScale.UTC and target is TimeScale.TT:
            return self.utc2tt(epoch)
        if epoch.scale is TimeScale.TT and target is TimeScale.UTC:
            return self.tt2utc(epoch)
        if epoch.scale is TimeScale.TT and target is TimeScale.TDB:
            return self.tt2tdb(epoch, station_gcrs_m=station_gcrs_m)
        if epoch.scale is TimeScale.TDB and target is TimeScale.TT:
            return self.tdb2tt(epoch, station_gcrs_m=station_gcrs_m)
        if epoch.scale is TimeScale.UTC and target is TimeScale.TDB:
            return self.tt2tdb(
                self.utc2tt(epoch),
                station_gcrs_m=station_gcrs_m,
            )
        if epoch.scale is TimeScale.TDB and target is TimeScale.UTC:
            return self.tt2utc(
                self.tdb2tt(epoch, station_gcrs_m=station_gcrs_m)
            )
        raise AssertionError("Unhandled time-scale conversion.")

    def isot(
        self,
        epoch: Epoch,
        *,
        scale: TimeScale | str = TimeScale.UTC,
        precision: int = 9,
        station_gcrs_m: Sequence[float] | None = None,
    ) -> str:
        target = TimeScale.parse(scale)
        if target is TimeScale.TDB:
            raise ValueError(
                "ISOT output is limited to UTC or TT. TDB epochs are serialized "
                "as two-part Julian dates."
            )
        converted = self.convert(
            epoch,
            target,
            station_gcrs_m=station_gcrs_m,
        )
        return converted.isot(scale=target, precision=precision)


__all__ = ["TimeScaleConverter"]
