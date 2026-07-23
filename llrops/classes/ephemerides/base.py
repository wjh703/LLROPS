"""Core ephemeris interfaces and immutable query/result objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llrops.base.epoch import Epoch, TimeScale
from llrops.base.array_validation import readonly_matrix3x3, finite_array



def require_tdb_epoch(epoch: Epoch, *, name: str = "epoch") -> Epoch:
    if not isinstance(epoch, Epoch):
        raise TypeError(f"{name} must be an Epoch.")
    return epoch.require_scale(TimeScale.TDB, name=name)


@dataclass(frozen=True, slots=True, eq=False)
class BodyState:
    """BCRS position and velocity of one body relative to the SSB."""

    position_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_m",
            finite_array(self.position_m, size=3, name="position_m", copy=True, readonly=True),
        )
        object.__setattr__(
            self,
            "velocity_mps",
            finite_array(self.velocity_mps, size=3, name="velocity_mps", copy=True, readonly=True),
        )


class Ephemeris(ABC):
    """Abstract ephemeris used by the LLR physical models."""

    @property
    @abstractmethod
    def source_file(self) -> Path:
        ...

    @abstractmethod
    def body_state_bcrs(self, body: str, epoch: Epoch) -> BodyState:
        """Return a body's SSB-relative BCRS state at a TDB epoch."""

    def body_position_bcrs(self, body: str, epoch: Epoch) -> np.ndarray:
        return np.array(self.body_state_bcrs(body, epoch).position_m, copy=True)

    @abstractmethod
    def pa2lcrs_matrix(self, epoch: Epoch) -> np.ndarray:
        """Return the passive rotation from lunar PA axes to LCRS axes at TDB."""

    def tdb_minus_tt_sec(self, epoch: Epoch) -> float | None:
        """Return geocentric TDB-TT in seconds at a TDB epoch."""
        require_tdb_epoch(epoch)
        return None

    @property
    def longitude_libration_model(self) -> str:
        return "none"

    def longitude_libration_correction_rad(self, epoch: Epoch) -> float:
        require_tdb_epoch(epoch)
        return 0.0

    @property
    def lb_minus_ll(self) -> float:
        return 0.0

    def close(self) -> None:
        """Release resources; the default implementation owns none."""
        return None

    def __enter__(self) -> "Ephemeris":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "BodyState",
    "Ephemeris",
    "require_tdb_epoch",
    "readonly_matrix3x3",
]
