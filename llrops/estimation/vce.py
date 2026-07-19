"""Variance-component estimation strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

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
    normals: NormalEquations


class VarianceComponentEstimator(ABC):
    """Base API for Helmert, simplified Helmert, and least-squares VCE."""

    method: str

    @abstractmethod
    def estimate(self, *, equations: Sequence[ObservationEquation], residuals: Mapping[ObsKey, float], normals: NormalEquations, parametrization: ParametrizationList, parameter_names: Sequence[ParameterName], assignments: Mapping[ObsKey, str], factors: Mapping[ObsKey, float], scales: Mapping[str, float], variance_damping: float) -> VarianceComponentEstimate:
        raise NotImplementedError


@dataclass(frozen=True)
class HelmertVceEstimator(VarianceComponentEstimator):
    """Helmert trace VCE using exact component effective redundancies."""

    components: tuple[VarianceComponentDefinition, ...]
    minimum_nonzero_factor: float = 1.0e-12
    minimum_effective_redundancy: float = 20.0
    minimum_variance_ratio_per_iteration: float = 0.25
    maximum_variance_ratio_per_iteration: float = 4.0
    method: str = "helmert"

    def estimate(self, *, equations, residuals, normals, parametrization, parameter_names, assignments, factors, scales, variance_damping):
        active = [eq for eq in equations if factors[eq.identity] > self.minimum_nonzero_factor]
        covariance = solve_normal_equations(normals).covariance
        if covariance is None:
            raise RuntimeError("Equivalent-weight normal equation covariance is unavailable.")
        if normals.obs_count != len(active):
            raise RuntimeError("Equivalent-weight normal equations and active observation set are inconsistent.")
        component_normals = {component.id: NormalEquations.zeros(parameter_names) for component in self.components}
        counts = {component.id: 0 for component in self.components}
        numerators = {component.id: 0.0 for component in self.components}
        for equation in active:
            component_id = assignments[equation.identity]
            weight = factors[equation.identity] / (scales[component_id] ** 2 * equation.sigma_m ** 2)
            component_normals[component_id].accumulate_sparse_row(parametrization.design_entries(equation), 0.0, weight=weight)
            counts[component_id] += 1
            numerators[component_id] += factors[equation.identity] * residuals[equation.identity] ** 2 / equation.sigma_m ** 2
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
                next_variance = current_variance * float(np.exp(variance_damping * np.log(limited_ratio)))
                status = "UPDATED"
            updates[component.id] = float(np.sqrt(next_variance))
            diagnostics[component.id] = {
                "active_count": float(active_count), "consumed_dof": consumed,
                "effective_redundancy": redundancy, "variance_before": float(current_variance),
                "raw_variance": float(raw_variance), "raw_variance_ratio": float(raw_ratio),
                "limited_variance_ratio": float(limited_ratio),
                "target_scale_log_change": float(abs(np.log(raw_ratio))),
                "variance_damping": float(variance_damping), "variance_after": float(next_variance),
                "variance_ratio": float(next_variance / current_variance), "update_status": status,
            }
        expected = float(len(active) - np.linalg.matrix_rank(normals.N))
        actual = sum(float(item["effective_redundancy"]) for item in diagnostics.values())
        if not np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(f"Helmert redundancy check failed: {actual:.12g} != {expected:.12g}.")
        return VarianceComponentEstimate(updates, diagnostics, normals)


class SimplifiedHelmertVceEstimator(VarianceComponentEstimator):
    method = "simplified_helmert"
    def estimate(self, **kwargs):
        raise NotImplementedError("Simplified Helmert VCE requires an explicitly selected and validated approximation model.")


class LeastSquaresVceEstimator(VarianceComponentEstimator):
    method = "least_squares"
    def estimate(self, **kwargs):
        raise NotImplementedError("Least-squares VCE requires covariance-component matrices and is not interchangeable with Helmert trace VCE.")


__all__ = ["HelmertVceEstimator", "LeastSquaresVceEstimator", "SimplifiedHelmertVceEstimator", "VarianceComponentEstimate", "VarianceComponentEstimator"]
