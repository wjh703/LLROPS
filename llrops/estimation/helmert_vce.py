"""Helmert variance-component estimation for LLR observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from llrops.estimation.variance_components import VarianceComponentDefinition
from llrops.fileio.normal_equations import NormalEquations


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

    def _finalize(
        self,
        *,
        covariance: np.ndarray,
        component_normal_matrices: Mapping[str, np.ndarray],
        counts: Mapping[str, int],
        numerators: Mapping[str, float],
        scales: Mapping[str, float],
        normal_matrix: np.ndarray,
        active_count: int,
    ) -> VarianceComponentEstimate:
        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, object]] = {}
        for component in self.components:
            component_id = component.id
            consumed = float(
                np.trace(covariance @ component_normal_matrices[component_id])
            )
            count = counts[component_id]
            redundancy = float(count - consumed)
            current_variance = float(scales[component_id]) ** 2
            if redundancy < self.minimum_effective_redundancy:
                raw_variance = current_variance
                raw_ratio = limited_ratio = 1.0
                next_variance = current_variance
                status = "INSUFFICIENT_REDUNDANCY"
            else:
                raw_variance = numerators[component_id] / redundancy
                if not np.isfinite(raw_variance) or raw_variance <= 0.0:
                    raise RuntimeError(
                        f"Invalid Helmert VCE estimate for component "
                        f"{component_id!r}: {raw_variance!r}."
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
            proposed_scale = float(np.sqrt(next_variance))
            updates[component_id] = proposed_scale
            diagnostics[component_id] = {
                "active_count": float(count),
                "consumed_dof": consumed,
                "effective_redundancy": redundancy,
                "current_variance": current_variance,
                "estimated_variance": float(raw_variance),
                "estimated_variance_ratio": float(raw_ratio),
                "bounded_variance_ratio": float(limited_ratio),
                "target_scale_log_change": float(abs(np.log(raw_ratio))),
                "proposed_variance": float(next_variance),
                "proposed_scale": proposed_scale,
                "update_status": status,
            }
        expected = float(active_count - np.linalg.matrix_rank(normal_matrix))
        actual = sum(
            float(item["effective_redundancy"])
            for item in diagnostics.values()
        )
        if not np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(
                f"Helmert redundancy check failed: {actual:.12g} != "
                f"{expected:.12g}."
            )
        return VarianceComponentEstimate(updates, diagnostics)

    def estimate(
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

        return self._finalize(
            covariance=covariance,
            component_normal_matrices=component_normal_matrices,
            counts=counts,
            numerators=numerators,
            scales=scales,
            normal_matrix=normals.N,
            active_count=int(np.count_nonzero(active)),
        )


__all__ = ["HelmertVceEstimator", "VarianceComponentEstimate"]
