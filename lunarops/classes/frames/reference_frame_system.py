"""Facade combining time conversion and terrestrial/lunar/relativistic frames."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from lunarops.base.epoch import Epoch
from lunarops.classes.time_scale_converter import TimeScaleConverter
from lunarops.classes.ephemerides import Ephemeris

from .earth_orientation import EarthOrientationProvider
from .lunar import LunarFrameTransform
from .relativistic import RelativisticFrameTransform
from .terrestrial import TerrestrialFrameTransform


class ReferenceFrameSystem:
    def __init__(
        self,
        ephemeris: Ephemeris,
        earth_orientation_provider: EarthOrientationProvider | None = None,
        time_scale_converter: TimeScaleConverter | None = None,
        *,
        earth_orientation: EarthOrientationProvider | None = None,
        time_converter: TimeScaleConverter | None = None,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if earth_orientation_provider is not None and earth_orientation is not None:
            raise ValueError(
                "Specify only one of earth_orientation_provider or earth_orientation."
            )
        if earth_orientation_provider is None:
            earth_orientation_provider = earth_orientation
        if earth_orientation_provider is None:
            raise TypeError("earth_orientation_provider is required.")
        if not isinstance(earth_orientation_provider, EarthOrientationProvider):
            raise TypeError(
                "earth_orientation_provider must be an EarthOrientationProvider instance."
            )
        if time_scale_converter is not None and time_converter is not None:
            raise ValueError(
                "Specify only one of time_scale_converter or time_converter."
            )
        if time_scale_converter is None:
            time_scale_converter = time_converter
        if time_scale_converter is None:
            time_scale_converter = TimeScaleConverter(ephemeris)
        elif time_scale_converter.ephemeris is not ephemeris:
            raise ValueError(
                "time_scale_converter must use the same ephemeris as the frame system."
            )
        self.ephemeris = ephemeris
        self.earth_orientation_provider = earth_orientation_provider
        self.time_scale_converter = time_scale_converter
        self.terrestrial_transform = TerrestrialFrameTransform(earth_orientation_provider)
        self.lunar_transform = LunarFrameTransform(ephemeris)
        self.relativistic_transform = RelativisticFrameTransform(ephemeris)

        # Legacy attribute aliases; new code should use the explicit names above.
        self.earth_orientation = earth_orientation_provider
        self.time_converter = time_scale_converter
        self.terrestrial = self.terrestrial_transform
        self.lunar = self.lunar_transform
        self.relativistic = self.relativistic_transform

    @property
    def ephemeris_file_path(self) -> Path | None:
        return self.ephemeris.source_file_path

    @property
    def ephemeris_file(self) -> Path | None:
        """Backward-compatible alias for :attr:`ephemeris_file_path`."""
        return self.ephemeris_file_path

    def itrf2gcrs(self, position_itrf_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        return self.terrestrial_transform.itrf2gcrs(position_itrf_m, epoch_utc)

    def gcrs2itrf(self, position_gcrs_m: Sequence[float], epoch_utc: Epoch) -> np.ndarray:
        return self.terrestrial_transform.gcrs2itrf(position_gcrs_m, epoch_utc)

    def pa2lcrs(self, position_pa_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.lunar_transform.pa2lcrs(position_pa_m, epoch_tdb)

    def lcrs2pa(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.lunar_transform.lcrs2pa(position_lcrs_m, epoch_tdb)

    def gcrs2bcrs(self, position_gcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.gcrs2bcrs(position_gcrs_m, epoch_tdb)

    def bcrs2gcrs(self, position_bcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.bcrs2gcrs(position_bcrs_m, epoch_tdb)

    def lcrs2bcrs(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.lcrs2bcrs(position_lcrs_m, epoch_tdb)

    def bcrs2lcrs(self, position_bcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.bcrs2lcrs(position_bcrs_m, epoch_tdb)

    def lcrs2gcrs(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.lcrs2gcrs(position_lcrs_m, epoch_tdb)

    def gcrs2lcrs(self, position_gcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.relativistic_transform.gcrs2lcrs(position_gcrs_m, epoch_tdb)

    def external_gravitational_potential_m2_s2(
        self,
        center_body_name: str,
        epoch_tdb: Epoch,
        perturbing_body_names: Iterable[str],
    ) -> float:
        return self.relativistic_transform.external_gravitational_potential_m2_s2(
            center_body_name,
            epoch_tdb,
            perturbing_body_names,
        )

    def external_potential(
        self,
        center: str,
        epoch_tdb: Epoch,
        bodies: Iterable[str],
    ) -> float:
        """Backward-compatible alias for the explicit gravitational-potential API."""
        return self.external_gravitational_potential_m2_s2(
            center,
            epoch_tdb,
            bodies,
        )


__all__ = ["ReferenceFrameSystem"]
