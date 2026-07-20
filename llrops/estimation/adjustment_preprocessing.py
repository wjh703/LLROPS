"""Observation preprocessing for nonlinear LLR adjustment."""

from __future__ import annotations

from dataclasses import replace
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import ParametrizationList
from llrops.estimation.variance_components import VarianceComponentDefinition

ObsKey = Hashable


def _normalise(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _metadata_candidates(eq: ObservationEquation, *keys: str) -> set[str]:
    metadata = eq.metadata or {}
    values = [eq.station_key]
    values.extend(metadata.get(key) for key in keys)
    return {_normalise(value) for value in values if _normalise(value)}


def prefit_gross_threshold(
    equation: ObservationEquation,
    default: Optional[float],
    by_station: Optional[Mapping[str, Optional[float]]],
) -> Optional[float]:
    overrides = by_station or {}
    candidates = _metadata_candidates(
        equation,
        "station_catalog_key",
        "station_name",
        "station_full_name",
        "station_id",
        "station_code",
    )
    for key in candidates:
        if key in overrides:
            value = overrides[key]
            return None if value is None else float(value)
    return None if default is None else float(default)


def prefit_gross_rejections(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    threshold_m: Optional[float],
    threshold_by_station_m: Optional[Mapping[str, Optional[float]]],
) -> dict[ObsKey, float]:
    rejected: dict[ObsKey, float] = {}
    for equation in equations:
        threshold = prefit_gross_threshold(
            equation, threshold_m, threshold_by_station_m
        )
        if threshold is None:
            continue
        residual = float(parametrization.reduced_observation(equation))
        if abs(residual) > threshold:
            rejected[equation.identity] = residual
    return rejected


def floor_prefit_uncertainties(
    equations: Sequence[ObservationEquation],
    assignments: Mapping[ObsKey, str],
    *,
    minimum_sigma_m: float,
    minimum_group_median_fraction: float,
) -> tuple[
    list[ObservationEquation],
    dict[ObsKey, dict[str, object]],
    dict[str, dict[str, object]],
]:
    """Apply a fixed prefit sigma floor within each variance-component group."""
    grouped_sigmas: dict[str, list[float]] = {}
    for equation in equations:
        component_id = assignments[equation.identity]
        grouped_sigmas.setdefault(component_id, []).append(float(equation.sigma_m))

    group_diagnostics: dict[str, dict[str, object]] = {}
    for component_id, values in grouped_sigmas.items():
        median = float(np.median(np.asarray(values, dtype=float)))
        floor = max(
            float(minimum_sigma_m),
            float(minimum_group_median_fraction) * median,
        )
        group_diagnostics[component_id] = {
            "median_reported_sigma_m": median,
            "sigma_floor_m": floor,
        }

    adjusted: list[ObservationEquation] = []
    records: dict[ObsKey, dict[str, object]] = {}
    for equation in equations:
        component_id = assignments[equation.identity]
        reported = float(equation.sigma_m)
        floor = group_diagnostics[component_id]["sigma_floor_m"]
        effective = max(reported, floor)
        floored = effective > reported
        qc = {
            "component_id": component_id,
            "reported_sigma_m": reported,
            "effective_sigma_m": effective,
            "sigma_floor_m": floor,
            "status": "FLOORED" if floored else "UNCHANGED",
            "reason": "BELOW_PREFIT_UNCERTAINTY_FLOOR" if floored else None,
        }
        records[equation.identity] = qc
        metadata = dict(equation.metadata or {})
        metadata["uncertainty_quality_control"] = qc
        adjusted.append(replace(equation, sigma_m=effective, metadata=metadata))

    for component_id, diagnostics in group_diagnostics.items():
        component_records = [
            item for item in records.values() if item["component_id"] == component_id
        ]
        diagnostics["observation_count"] = len(component_records)
        diagnostics["floored_count"] = sum(
            item["status"] == "FLOORED" for item in component_records
        )
    return adjusted, records, group_diagnostics


def _bias_indices(names: Sequence[ParameterName]) -> np.ndarray:
    return np.asarray(
        [index for index, name in enumerate(names) if name.type == "rangeBias"],
        dtype=int,
    )


def robust_bias_initial_values(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    names: Sequence[ParameterName],
    *,
    weight_cap: float,
    maximum_iterations: int,
) -> np.ndarray:
    indices = _bias_indices(names)
    if not len(indices):
        return np.zeros(len(names), dtype=float)
    design = np.vstack([parametrization.design_row(eq)[indices] for eq in equations])
    observations = np.asarray(
        [parametrization.reduced_observation(eq) for eq in equations], dtype=float
    )
    formal_weights = np.asarray(
        [min(1.0 / (eq.sigma_m * eq.sigma_m), weight_cap) for eq in equations],
        dtype=float,
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return np.zeros(len(names), dtype=float)

    robust_weights = formal_weights.copy()
    beta = np.zeros(design.shape[1], dtype=float)
    for _ in range(maximum_iterations):
        normal = design.T @ (robust_weights[:, None] * design)
        rhs = design.T @ (robust_weights * observations)
        try:
            next_beta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(len(names), dtype=float)
        residuals = observations - design @ next_beta
        median = float(np.median(residuals))
        scale = 1.4826 * float(np.median(np.abs(residuals - median)))
        if not np.isfinite(scale) or scale <= 1.0e-12:
            beta = next_beta
            break
        magnitude = np.abs(residuals - median)
        huber = np.ones_like(magnitude)
        outside = magnitude > 1.345 * scale
        huber[outside] = 1.345 * scale / magnitude[outside]
        if np.max(np.abs(next_beta - beta)) <= 1.0e-10:
            beta = next_beta
            break
        beta = next_beta
        robust_weights = formal_weights * huber

    delta = np.zeros(len(names), dtype=float)
    delta[indices] = beta
    return delta


def initialize_mad_scales(
    equations: Sequence[ObservationEquation],
    parametrization: ParametrizationList,
    assignments: Mapping[ObsKey, str],
    components: Sequence[VarianceComponentDefinition],
    *,
    minimum_count: int,
    minimum_scale: float,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for component in components:
        values = np.asarray(
            [
                parametrization.reduced_observation(eq) / eq.sigma_m
                for eq in equations
                if assignments[eq.identity] == component.id
            ],
            dtype=float,
        )
        if len(values) < minimum_count:
            scales[component.id] = minimum_scale
            continue
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        scales[component.id] = (
            minimum_scale
            if not np.isfinite(scale) or scale <= 0.0
            else max(minimum_scale, scale)
        )
    return scales


__all__ = [
    "floor_prefit_uncertainties",
    "initialize_mad_scales",
    "prefit_gross_rejections",
    "prefit_gross_threshold",
    "robust_bias_initial_values",
]
