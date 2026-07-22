"""Helmert variance-component estimation for LLR observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import ParametrizationList
from llrops.estimation.normal_equation_engine import solve_normal_equations
from llrops.estimation.variance_components import VarianceComponentDefinition
from llrops.fileio.normal_equations import NormalEquations

ObsKey = Hashable


@dataclass(frozen=True)
class VarianceComponentEstimate:
    scales: dict[str, float]
    diagnostics: dict[str, dict[str, object]]


@dataclass(frozen=True)
class HelmertVceEstimator:
    """Helmert trace VCE using exact component effective redundancies."""

    components: tuple[VarianceComponentDefinition, ...]
    minimum_nonzero_factor: float = 1.0e-12
    minimum_effective_redundancy: float = 20.0
    minimum_variance_ratio_per_iteration: float = 0.25
    maximum_variance_ratio_per_iteration: float = 4.0

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("Helmert VCE requires at least one component.")
        component_ids = [component.id for component in self.components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("Helmert VCE component IDs must be unique.")
        values = (
            self.minimum_nonzero_factor,
            self.minimum_effective_redundancy,
            self.minimum_variance_ratio_per_iteration,
            self.maximum_variance_ratio_per_iteration,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Helmert VCE thresholds must be finite.")
        if not 0.0 < self.minimum_nonzero_factor < 1.0:
            raise ValueError("Helmert VCE active-factor threshold must be in (0, 1).")
        if self.minimum_effective_redundancy < 0.0:
            raise ValueError(
                "Helmert VCE minimum effective redundancy must be non-negative."
            )
        if not (
            0.0
            < self.minimum_variance_ratio_per_iteration
            <= self.maximum_variance_ratio_per_iteration
        ):
            raise ValueError("Helmert VCE variance-ratio limits are invalid.")

    def estimate(
        self,
        *,
        equations: Sequence[ObservationEquation],
        residuals: Mapping[ObsKey, float],
        normals: NormalEquations,
        parametrization: ParametrizationList,
        parameter_names: Sequence[ParameterName],
        assignments: Mapping[ObsKey, str],
        factors: Mapping[ObsKey, float],
        scales: Mapping[str, float],
        covariance: Optional[np.ndarray] = None,
    ) -> VarianceComponentEstimate:
        active = [
            equation
            for equation in equations
            if factors[equation.identity] > self.minimum_nonzero_factor
        ]
        covariance = (
            solve_normal_equations(normals).covariance
            if covariance is None
            else np.asarray(covariance, dtype=float)
        )
        if covariance is None:
            raise RuntimeError(
                "Equivalent-weight normal equation covariance is unavailable."
            )
        if normals.obs_count != len(active):
            raise RuntimeError(
                "Equivalent-weight normal equations and active observation set "
                "are inconsistent."
            )
        component_normals = {
            component.id: NormalEquations.zeros(parameter_names)
            for component in self.components
        }
        counts = {component.id: 0 for component in self.components}
        numerators = {component.id: 0.0 for component in self.components}
        for equation in active:
            component_id = assignments[equation.identity]
            weight = factors[equation.identity] / (
                scales[component_id] ** 2 * equation.sigma_m**2
            )
            component_normals[component_id].accumulate_sparse_row(
                parametrization.design_entries(equation),
                0.0,
                weight=weight,
            )
            counts[component_id] += 1
            numerators[component_id] += (
                factors[equation.identity]
                * residuals[equation.identity] ** 2
                / equation.sigma_m**2
            )
        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for component in self.components:
            consumed = float(np.trace(covariance @ component_normals[component.id].N))
            active_count = counts[component.id]
            redundancy = float(active_count - consumed)
            current_variance = scales[component.id] ** 2
            if redundancy < self.minimum_effective_redundancy:
                raw_variance = current_variance
                raw_ratio = limited_ratio = 1.0
                next_variance = current_variance
                status = "INSUFFICIENT_REDUNDANCY"
            else:
                raw_variance = numerators[component.id] / redundancy
                if not np.isfinite(raw_variance) or raw_variance <= 0.0:
                    raise RuntimeError(f"Invalid Helmert VCE estimate for component {component.id!r}: {raw_variance!r}.")
                raw_ratio = raw_variance / current_variance
                limited_ratio = float(np.clip(raw_ratio, self.minimum_variance_ratio_per_iteration, self.maximum_variance_ratio_per_iteration))
                next_variance = current_variance * limited_ratio
                status = "UPDATED"
            updates[component.id] = float(np.sqrt(next_variance))
            diagnostics[component.id] = {
                "active_count": float(active_count),
                "consumed_dof": consumed,
                "effective_redundancy": redundancy,
                "current_variance": float(current_variance),
                "estimated_variance": float(raw_variance),
                "estimated_variance_ratio": float(raw_ratio),
                "bounded_variance_ratio": float(limited_ratio),
                "target_scale_log_change": float(abs(np.log(raw_ratio))),
                "proposed_variance": float(next_variance),
                "proposed_scale": float(np.sqrt(next_variance)),
                "update_status": status,
            }
        expected = float(len(active) - np.linalg.matrix_rank(normals.N))
        actual = sum(float(item["effective_redundancy"]) for item in diagnostics.values())
        if not np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(f"Helmert redundancy check failed: {actual:.12g} != {expected:.12g}.")
        return VarianceComponentEstimate(updates, diagnostics)


    def estimate_dense(
        self,
        *,
        design,
        sigmas,
        residuals,
        component_ids,
        factors,
        scales,
        normals,
        covariance,
    ):
        design = np.asarray(design, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        residuals = np.asarray(residuals, dtype=float)
        factors = np.asarray(factors, dtype=float)
        component_ids = np.asarray(component_ids, dtype=object)
        active = factors > self.minimum_nonzero_factor
        covariance = np.asarray(covariance, dtype=float)
        counts: dict[str, int] = {}
        numerators: dict[str, float] = {}
        component_normal_matrices: dict[str, np.ndarray] = {}
        for component in self.components:
            mask = active & (component_ids == component.id)
            A = design[mask]
            factor = factors[mask]
            sigma = sigmas[mask]
            weight = factor / (float(scales[component.id]) ** 2 * sigma**2)
            component_normal_matrices[component.id] = A.T @ (weight[:, None] * A)
            counts[component.id] = int(np.count_nonzero(mask))
            numerators[component.id] = float(
                np.sum(factor * residuals[mask] ** 2 / sigma**2)
            )

        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for component in self.components:
            consumed = float(
                np.trace(covariance @ component_normal_matrices[component.id])
            )
            active_count = counts[component.id]
            redundancy = float(active_count - consumed)
            current_variance = float(scales[component.id]) ** 2
            if redundancy < self.minimum_effective_redundancy:
                raw_variance = current_variance
                raw_ratio = limited_ratio = 1.0
                next_variance = current_variance
                status = "INSUFFICIENT_REDUNDANCY"
            else:
                raw_variance = numerators[component.id] / redundancy
                if not np.isfinite(raw_variance) or raw_variance <= 0.0:
                    raise RuntimeError(
                        f"Invalid Helmert VCE estimate for component "
                        f"{component.id!r}: {raw_variance!r}."
                    )
                raw_ratio = raw_variance / current_variance
                limited_ratio = float(
                    np.clip(
                        raw_ratio,
                        self.minimum_variance_ratio_per_iteration,
                        self.maximum_variance_ratio_per_iteration,
                    )
                )
                next_variance = current_variance * limited_ratio
                status = "UPDATED"
            updates[component.id] = float(np.sqrt(next_variance))
            diagnostics[component.id] = {
                "active_count": float(active_count),
                "consumed_dof": consumed,
                "effective_redundancy": redundancy,
                "current_variance": float(current_variance),
                "estimated_variance": float(raw_variance),
                "estimated_variance_ratio": float(raw_ratio),
                "bounded_variance_ratio": float(limited_ratio),
                "target_scale_log_change": float(abs(np.log(raw_ratio))),
                "proposed_variance": float(next_variance),
                "proposed_scale": float(np.sqrt(next_variance)),
                "update_status": status,
            }
        expected = float(np.count_nonzero(active) - np.linalg.matrix_rank(normals.N))
        actual = sum(
            float(item["effective_redundancy"]) for item in diagnostics.values()
        )
        if not np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(
                f"Helmert redundancy check failed: {actual:.12g} != {expected:.12g}."
            )
        return VarianceComponentEstimate(updates, diagnostics)


__all__ = ["HelmertVceEstimator", "VarianceComponentEstimate"]
