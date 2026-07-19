"""Convergence policies for nonlinear parameter updates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ParameterConvergenceEvaluation:
    converged: bool
    tolerances_m: dict[str, float]
    normalized_updates: dict[str, float]


@dataclass(frozen=True)
class ParameterConvergencePolicy:
    """Evaluate each parametrization block against its own metric tolerance."""

    default_tolerance_m: float
    tolerance_by_block_m: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default_tolerance_m < 0.0:
            raise ValueError("Default parameter tolerance must be non-negative.")
        if any(float(value) < 0.0 for value in self.tolerance_by_block_m.values()):
            raise ValueError("Block parameter tolerances must be non-negative.")

    @staticmethod
    def _block_type(label: str) -> str:
        return label.split(":", 1)[-1]

    def tolerance_for(self, label: str) -> float:
        block_type = self._block_type(label)
        return float(
            self.tolerance_by_block_m.get(
                label,
                self.tolerance_by_block_m.get(block_type, self.default_tolerance_m),
            )
        )

    def evaluate(self, updates_m: Mapping[str, float]) -> ParameterConvergenceEvaluation:
        tolerances = {label: self.tolerance_for(label) for label in updates_m}
        ratios = {
            label: (0.0 if value == 0.0 else float("inf")) if tolerances[label] == 0.0 else float(value) / tolerances[label]
            for label, value in updates_m.items()
        }
        return ParameterConvergenceEvaluation(
            converged=all(ratio <= 1.0 for ratio in ratios.values()),
            tolerances_m=tolerances,
            normalized_updates=ratios,
        )


__all__ = ["ParameterConvergenceEvaluation", "ParameterConvergencePolicy"]
