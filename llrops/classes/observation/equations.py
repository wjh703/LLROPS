"""Typed linearized observation equations consumed by parametrizations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from llrops.base.epoch import Epoch, TimeScale

from .containers import FrozenMapping


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
    metadata: Mapping[str, object] | None = None

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
        normalized: dict[str, np.ndarray] = {}
        for name, values in dict(self.partials).items():
            array = np.array(values, dtype=float, copy=True).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Partial block {name!r} contains non-finite values.")
            array.setflags(write=False)
            normalized[str(name)] = array
        object.__setattr__(self, "observed_minus_computed_m", residual)
        object.__setattr__(self, "sigma_m", sigma)
        object.__setattr__(self, "partials", FrozenMapping(normalized))
        object.__setattr__(self, "station_key", str(self.station_key))
        object.__setattr__(self, "reflector_key", str(self.reflector_key))
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata or {}))


__all__ = ["ObservationEquation"]
