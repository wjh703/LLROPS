from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from llrops.base.epoch import Epoch


class GravitationalDelay(ABC):
    """Interface for a one-way gravitational path delay model."""

    @abstractmethod
    def path_delay_m(
        self,
        transmitter_bcrs_m: Sequence[float],
        receiver_bcrs_m: Sequence[float],
        epoch_tdb: Epoch,
    ) -> float:
        """Return the one-way equivalent path delay in meters."""


class ZeroGravitationalDelay(GravitationalDelay):
    """Gravitational delay model that always returns zero."""

    def path_delay_m(self, transmitter_bcrs_m, receiver_bcrs_m, epoch_tdb: Epoch) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class TroposphereInput:
    """Inputs required to evaluate an optical tropospheric slant delay."""

    elevation_rad: float
    pressure_hpa: float
    temperature_k: float
    relative_humidity_percent: float
    latitude_rad: float
    height_m: float
    wavelength_um: float


class TroposphereDelay(ABC):
    """Interface for a one-way tropospheric slant-delay model."""

    @abstractmethod
    def slant_delay_m(self, data: TroposphereInput) -> float:
        """Return the one-way tropospheric slant delay in meters."""


class ZeroTroposphereDelay(TroposphereDelay):
    """Tropospheric delay model that always returns zero."""

    def slant_delay_m(self, data: TroposphereInput) -> float:
        return 0.0
