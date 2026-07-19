"""Robust observation-weight models for iterative LLR adjustment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Protocol, Sequence

import numpy as np

ObsKey = Hashable


@dataclass(frozen=True)
class RobustWeightUpdate:
    target_factors: dict[ObsKey, float]
    applied_factors: dict[ObsKey, float]
    target_change_quantile: float
    active_set_change_fraction: float
    maximum_applied_change: float


class RobustWeightModel(Protocol):
    def initial_factors(self, keys: Sequence[ObsKey]) -> dict[ObsKey, float]: ...

    def update(
        self,
        standardized_residuals: Mapping[ObsKey, float],
        current_factors: Mapping[ObsKey, float],
        previous_target_factors: Mapping[ObsKey, float],
        keys: Sequence[ObsKey],
    ) -> RobustWeightUpdate: ...


@dataclass(frozen=True)
class Igg3WeightModel:
    """IGGIII IRLS model with immediate zero-target rejection."""

    k0: float = 1.5
    k1: float = 6.0
    active_threshold: float = 1.0e-12
    convergence_floor: float = 1.0e-3
    change_quantile: float = 0.999

    def __post_init__(self) -> None:
        if not np.isfinite(self.k0) or not np.isfinite(self.k1):
            raise ValueError("IGGIII thresholds must be finite.")
        if not 0.0 < self.k0 < self.k1:
            raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
        if not 0.0 < self.change_quantile <= 1.0:
            raise ValueError("Robust factor change quantile must be in (0, 1].")

    def initial_factors(self, keys: Sequence[ObsKey]) -> dict[ObsKey, float]:
        return {key: 1.0 for key in keys}

    def target_factors(
        self,
        standardized_residuals: Mapping[ObsKey, float],
        keys: Sequence[ObsKey],
    ) -> dict[ObsKey, float]:
        values = np.asarray([standardized_residuals[key] for key in keys], dtype=float)
        factors = igg3_factors(values, k0=self.k0, k1=self.k1)
        return {key: float(value) for key, value in zip(keys, factors)}

    def update(
        self,
        standardized_residuals: Mapping[ObsKey, float],
        current_factors: Mapping[ObsKey, float],
        previous_target_factors: Mapping[ObsKey, float],
        keys: Sequence[ObsKey],
    ) -> RobustWeightUpdate:
        targets = self.target_factors(standardized_residuals, keys)
        applied = dict(targets)
        previous_targets = {
            key: previous_target_factors.get(key, current_factors.get(key, 1.0))
            for key in keys
        }
        return RobustWeightUpdate(
            target_factors=targets,
            applied_factors=applied,
            target_change_quantile=robust_factor_change_quantile(
                current_factors,
                targets,
                keys,
                quantile=self.change_quantile,
                significance_floor=self.convergence_floor,
            ),
            active_set_change_fraction=active_set_change_fraction(
                previous_targets,
                targets,
                keys,
                active_threshold=self.active_threshold,
            ),
            maximum_applied_change=maximum_robust_factor_change(
                current_factors,
                applied,
                keys,
                significance_floor=self.convergence_floor,
            ),
        )


def igg3_factors(values: np.ndarray, *, k0: float, k1: float) -> np.ndarray:
    if not np.isfinite(k0) or not np.isfinite(k1) or not 0.0 < k0 < k1:
        raise ValueError("IGGIII thresholds must satisfy 0 < k0 < k1.")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Standardized residuals must be finite.")
    magnitude = np.abs(values)
    result = np.zeros_like(magnitude)
    full = magnitude <= k0
    middle = (magnitude > k0) & (magnitude <= k1)
    result[full] = 1.0
    result[middle] = k0 / magnitude[middle] * ((k1 - magnitude[middle]) / (k1 - k0)) ** 2
    return result


def maximum_robust_factor_change(old_factors, new_factors, keys, *, significance_floor=0.0):
    return max((abs(new_factors[key] - old_factors[key]) for key in keys if max(abs(old_factors[key]), abs(new_factors[key])) >= significance_floor), default=0.0)


def robust_factor_change_quantile(old_factors, target_factors, keys, *, quantile, significance_floor=0.0):
    changes = np.asarray([abs(target_factors[key] - old_factors[key]) for key in keys if max(abs(old_factors[key]), abs(target_factors[key])) >= significance_floor], dtype=float)
    return 0.0 if not len(changes) else float(np.quantile(changes, quantile, method="higher"))


def active_set_change_fraction(old_factors, new_factors, keys, *, active_threshold):
    if not keys:
        return 0.0
    changed = sum((old_factors[key] > active_threshold) != (new_factors[key] > active_threshold) for key in keys)
    return float(changed / len(keys))


__all__ = [
    "Igg3WeightModel", "RobustWeightModel", "RobustWeightUpdate",
    "active_set_change_fraction", "igg3_factors",
    "maximum_robust_factor_change",
    "robust_factor_change_quantile",
]
