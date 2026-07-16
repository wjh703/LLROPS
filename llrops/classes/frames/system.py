"""Facade combining time conversion and terrestrial/lunar/relativistic frames."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from llrops.base.epoch import Epoch
from llrops.classes.time import TimeScaleConverter
from llrops.classes.ephemerides import Ephemeris

from .earth_orientation import EarthOrientation
from .lunar import LunarFrameTransform
from .relativistic import RelativisticFrameTransform
from .terrestrial import TerrestrialFrameTransform


class ReferenceFrameSystem:
    def __init__(
        self,
        ephemeris: Ephemeris,
        earth_orientation: EarthOrientation,
        time_converter: TimeScaleConverter | None = None,
        owns_ephemeris: bool = False,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if not isinstance(earth_orientation, EarthOrientation):
            raise TypeError("earth_orientation must implement EarthOrientation.")
        if time_converter is None:
            time_converter = TimeScaleConverter(ephemeris)
        elif time_converter.ephemeris is not ephemeris:
            raise ValueError("time_converter must use the same ephemeris as the frame system.")
        self.ephemeris = ephemeris
        self.earth_orientation = earth_orientation
        self.time_converter = time_converter
        self.owns_ephemeris = owns_ephemeris
        self.terrestrial = TerrestrialFrameTransform(earth_orientation)
        self.lunar = LunarFrameTransform(ephemeris)
        self.relativistic = RelativisticFrameTransform(ephemeris)

    @property
    def ephemeris_file(self):
        return self.ephemeris.source_file

    def close(self) -> None:
        if self.owns_ephemeris:
            self.ephemeris.close()

    def itrf2gcrs(self, position_itrf_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        return self.terrestrial.itrf2gcrs(position_itrf_m, epoch_utc)

    def gcrs2itrf(self, position_gcrs_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        return self.terrestrial.gcrs2itrf(position_gcrs_m, epoch_utc)

    def pa2lcrs(self, position_pa_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.lunar.pa2lcrs(position_pa_m, epoch_tdb)

    def lcrs2pa(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.lunar.lcrs2pa(position_lcrs_m, epoch_tdb)

    def gcrs2bcrs(self, position_gcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic.gcrs2bcrs(position_gcrs_m, epoch_tdb)

    def bcrs2gcrs(self, position_bcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic.bcrs2gcrs(position_bcrs_m, epoch_tdb)

    def lcrs2bcrs(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic.lcrs2bcrs(position_lcrs_m, epoch_tdb)

    def bcrs2lcrs(self, position_bcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic.bcrs2lcrs(position_bcrs_m, epoch_tdb)

    def lcrs2gcrs(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic.lcrs2gcrs(position_lcrs_m, epoch_tdb)

    def external_potential(
        self,
        center: str,
        epoch_tdb: Epoch,
        bodies: Iterable[str],
    ) -> float:
        return self.relativistic.external_potential(center, epoch_tdb, bodies)


__all__ = ["ReferenceFrameSystem"]
