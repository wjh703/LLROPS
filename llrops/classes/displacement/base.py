"""Core displacement interfaces and immutable evaluation inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from llrops.base.epoch import Epoch, TimeScale
from llrops.base.validation import readonly_vector3


def _epoch(value: Epoch, *, scale: TimeScale, name: str) -> Epoch:
    if not isinstance(value, Epoch):
        raise TypeError(f"{name} must be an Epoch.")
    return value.require_scale(scale, name=name)


@dataclass(frozen=True, slots=True, eq=False)
class StationDisplacementInput:
    station_itrf_m: np.ndarray
    epoch_utc: Epoch

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "station_itrf_m",
            readonly_vector3(self.station_itrf_m, name="station_itrf_m"),
        )
        object.__setattr__(
            self,
            "epoch_utc",
            _epoch(self.epoch_utc, scale=TimeScale.UTC, name="epoch_utc"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ReflectorDisplacementInput:
    reflector_lcrs_m: np.ndarray
    epoch_tdb: Epoch

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reflector_lcrs_m",
            readonly_vector3(
                self.reflector_lcrs_m,
                name="reflector_lcrs_m",
            ),
        )
        object.__setattr__(
            self,
            "epoch_tdb",
            _epoch(self.epoch_tdb, scale=TimeScale.TDB, name="epoch_tdb"),
        )


@runtime_checkable
class StationDisplacement(Protocol):
    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        ...


@runtime_checkable
class ReflectorDisplacement(Protocol):
    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        ...


class ZeroStationDisplacement:
    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        return np.zeros(3, dtype=float)


class ZeroReflectorDisplacement:
    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        return np.zeros(3, dtype=float)


class CompositeStationDisplacement:
    def __init__(self, components: Sequence[StationDisplacement] = ()) -> None:
        normalized = tuple(components)
        for index, component in enumerate(normalized):
            if component is None:
                raise TypeError(
                    "CompositeStationDisplacement components cannot contain None; "
                    f"component {index} is invalid."
                )
            if not callable(getattr(component, "displacement_itrf_m", None)):
                raise TypeError(
                    "CompositeStationDisplacement components must implement "
                    f"displacement_itrf_m(data); component {index} is {type(component)!r}."
                )
        self.components = normalized

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        total = np.zeros(3, dtype=float)
        for component in self.components:
            value = np.asarray(component.displacement_itrf_m(data), dtype=float)
            if value.size != 3:
                raise ValueError(
                    f"{type(component).__name__}.displacement_itrf_m() must return three values."
                )
            value = value.reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{type(component).__name__}.displacement_itrf_m() returned non-finite values."
                )
            total += value
        return total


__all__ = [
    "CompositeStationDisplacement",
    "ReflectorDisplacement",
    "ReflectorDisplacementInput",
    "StationDisplacement",
    "StationDisplacementInput",
    "ZeroReflectorDisplacement",
    "ZeroStationDisplacement",
]
