"""Minimal linearized observation equations used by estimation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Mapping

import numpy as np

from llrops.base.epoch import Epoch, TimeScale


class ObservationOutputLevel(str, Enum):
    STANDARD = "standard"
    FULL = "full"

    @classmethod
    def parse(cls, value: object) -> "ObservationOutputLevel":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or cls.STANDARD.value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unknown observation output level {value!r}.") from exc


STANDARD_OUTPUT_FIELDS = (
    "obs_time_utc",
    "normal_point_index",
    "station_id",
    "station_name",
    "reflector_id",
    "reflector_name",
    "observed_rtt_s",
    "computed_rtt_s",
    "oc_one_way_m",
    "fit_sigma_one_way_m",
    "elevation_up_deg",
    "converged",
    "status",
)

REFLECTOR_DESIGN_OUTPUT_FIELDS = (
    "design_reflector_dx",
    "design_reflector_dy",
    "design_reflector_dz",
)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationEquation:
    observed_minus_computed_m: float
    sigma_m: float
    partials: Mapping[str, np.ndarray]
    identity: Hashable
    station_key: str
    reflector_key: str
    epoch: Epoch
    converged: bool = True
    wavelength_nm: float | None = None

    def __post_init__(self) -> None:
        residual = float(self.observed_minus_computed_m)
        sigma = float(self.sigma_m)
        if not np.isfinite(residual):
            raise ValueError("observed_minus_computed_m must be finite.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_m must be positive and finite.")
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch.")
        self.epoch.require_scale(TimeScale.UTC, name="epoch")
        try:
            hash(self.identity)
        except TypeError as exc:
            raise TypeError("identity must be hashable.") from exc
        if not isinstance(self.station_key, str) or not self.station_key:
            raise TypeError("station_key must be a non-empty string.")
        if not isinstance(self.reflector_key, str) or not self.reflector_key:
            raise TypeError("reflector_key must be a non-empty string.")
        normalized: dict[str, np.ndarray] = {}
        for name, values in self.partials.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Partial-block names must be non-empty strings.")
            array = np.array(values, dtype=float, copy=True).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Partial block {name!r} contains non-finite values.")
            array.setflags(write=False)
            normalized[name] = array
        wavelength = self.wavelength_nm
        if wavelength is not None:
            wavelength = float(wavelength)
            if not np.isfinite(wavelength) or wavelength <= 0.0:
                raise ValueError("wavelength_nm must be positive and finite.")
        object.__setattr__(self, "observed_minus_computed_m", residual)
        object.__setattr__(self, "sigma_m", sigma)
        object.__setattr__(self, "partials", normalized)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "wavelength_nm", wavelength)

    @property
    def normal_point_index(self) -> Hashable:
        return self.identity


__all__ = [
    "ObservationEquation",
    "ObservationOutputLevel",
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
]
