"""Grouped VCE, interval-bias, and IGGIII adjustment support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Hashable, Mapping, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName, names_to_strings
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import ParametrizationList
from llrops.estimation.normal_equation_engine import (
    build_normal_equations_streaming,
    normal_matrix_condition,
    postfit_residuals_streaming,
    solve_normal_equations,
)
from llrops.fileio.normal_equations import NormalEquations

ObsKey = Hashable


def _normalise(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _as_texts(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_normalise(value),)
    return tuple(_normalise(item) for item in value if _normalise(item))


def _metadata_candidates(eq: ObservationEquation, *keys: str) -> set[str]:
    metadata = eq.metadata or {}
    values = [eq.station_key]
    values.extend(metadata.get(key) for key in keys)
    return {_normalise(value) for value in values if _normalise(value)}


@dataclass(frozen=True)
class VceGroup:
    id: str
    station_system: str
    start: str
    end_exclusive: Optional[str]
    station_aliases: tuple[str, ...]
    system_aliases: tuple[str, ...]
    wavelength_min_nm: Optional[float] = None
    wavelength_max_nm: Optional[float] = None

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "VceGroup":
        group_id = str(value.get("id") or "").strip()
        station_system = str(value.get("station_system") or value.get("stationSystem") or "").strip()
        start = str(value.get("start") or "").strip()
        if not group_id or not station_system or not start:
            raise ValueError("Each VCE group requires id, station_system, and start.")
        aliases = value.get("station_aliases", value.get("stationAliases"))
        systems = value.get("system_aliases", value.get("systemAliases"))
        end = value.get("end_exclusive", value.get("endExclusive"))
        return cls(
            id=group_id,
            station_system=station_system,
            start=start[:10],
            end_exclusive=None if end in (None, "", "present") else str(end)[:10],
            station_aliases=_as_texts(aliases) or (_normalise(station_system),),
            system_aliases=_as_texts(systems),
            wavelength_min_nm=(
                None
                if value.get("wavelength_min_nm", value.get("wavelengthMinNm")) is None
                else float(value.get("wavelength_min_nm", value.get("wavelengthMinNm")))
            ),
            wavelength_max_nm=(
                None
                if value.get("wavelength_max_nm", value.get("wavelengthMaxNm")) is None
                else float(value.get("wavelength_max_nm", value.get("wavelengthMaxNm")))
            ),
        )

    def matches(self, equation: ObservationEquation) -> bool:
        date = equation.epoch.date_iso()
        if date < self.start or (self.end_exclusive is not None and date >= self.end_exclusive):
            return False
        stations = _metadata_candidates(
            equation,
            "station_catalog_key",
            "station_name",
            "station_full_name",
            "station_id",
        )
        if not stations.intersection(self.station_aliases):
            return False
        if self.system_aliases:
            systems = _metadata_candidates(
                equation,
                "system_config_id",
                "system_name",
                "station_system",
                "observation_mode",
            )
            if not systems.intersection(self.system_aliases):
                return False
        wavelength = (equation.metadata or {}).get("wavelength_nm")
        if self.wavelength_min_nm is not None or self.wavelength_max_nm is not None:
            if wavelength is None:
                return False
            wavelength = float(wavelength)
            if self.wavelength_min_nm is not None and wavelength < self.wavelength_min_nm:
                return False
            if self.wavelength_max_nm is not None and wavelength > self.wavelength_max_nm:
                return False
        return True


def assign_vce_groups(
    equations: Sequence[ObservationEquation],
    groups: Sequence[VceGroup],
) -> dict[ObsKey, str]:
    if not groups:
        raise ValueError("At least one VCE group is required.")
    assignments: dict[ObsKey, str] = {}
    for equation in equations:
        matches = [group.id for group in groups if group.matches(equation)]
        if len(matches) != 1:
            detail = "no matching group" if not matches else f"multiple matching groups {matches!r}"
            raise ValueError(
                f"Observation {equation.identity!r} at {equation.epoch.date_iso()} has {detail}."
            )
        assignments[equation.identity] = matches[0]
    return assignments


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
        threshold = prefit_gross_threshold(equation, threshold_m, threshold_by_station_m)
        if threshold is None:
            continue
        residual = float(parametrization.reduced_observation(equation))
        if abs(residual) > threshold:
            rejected[equation.identity] = residual
    return rejected


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
    result[middle] = (
        k0
        / magnitude[middle]
        * ((k1 - magnitude[middle]) / (k1 - k0)) ** 2
    )
    return result


def _bias_indices(names: Sequence[ParameterName]) -> np.ndarray:
    return np.asarray([index for index, name in enumerate(names) if name.type == "rangeBias"], dtype=int)


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
    observations = np.asarray([parametrization.reduced_observation(eq) for eq in equations], dtype=float)
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
    groups: Sequence[VceGroup],
    *,
    minimum_count: int,
    minimum_scale: float,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for group in groups:
        values = np.asarray(
            [
                parametrization.reduced_observation(eq) / eq.sigma_m
                for eq in equations
                if assignments[eq.identity] == group.id
            ],
            dtype=float,
        )
        if len(values) < minimum_count:
            scales[group.id] = minimum_scale
            continue
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        scales[group.id] = minimum_scale if not np.isfinite(scale) or scale <= 0.0 else max(minimum_scale, scale)
    return scales


@dataclass(frozen=True)
class GroupedVceOptions:
    groups: tuple[VceGroup, ...]
    prefit_gross_threshold_m: Optional[float] = 20.0
    prefit_gross_threshold_by_station_m: Optional[Mapping[str, Optional[float]]] = None
    function_max_iterations: int = 20
    function_damping: float = 1.0
    update_tolerance_m: float = 1.0e-3
    wrms_tolerance_m: float = 1.0e-4
    maximum_stochastic_iterations: int = 20
    k0: float = 1.5
    k1: float = 6.0
    minimum_one_minus_leverage: float = 1.0e-8
    minimum_nonzero_robust_factor: float = 1.0e-12
    minimum_mad_count: int = 10
    minimum_initial_scale: float = 1.0
    bias_weight_cap: float = 1.0e12
    bias_maximum_iterations: int = 30
    vce_damping: float = 0.5
    minimum_effective_redundancy: float = 20.0
    scale_log_tolerance: float = 1.0e-3
    robust_weight_tolerance: float = 1.0e-3
    minimum_variance_ratio_per_iteration: float = 0.25
    maximum_variance_ratio_per_iteration: float = 4.0


@dataclass
class GroupedVceIteration:
    iteration: int
    function_iterations: int
    maximum_scale_log_change: float
    maximum_robust_factor_change: float
    active_observation_count: int
    rejected_observation_count: int
    total_effective_redundancy: float
    expected_total_redundancy: float
    normal_matrix_condition: Optional[float]


@dataclass
class GroupedVceResult:
    converged: bool
    parameter_names: list[ParameterName]
    state: dict[str, object]
    gross_rejected: dict[ObsKey, float]
    scales: dict[str, float]
    robust_factors: dict[ObsKey, float]
    iterations: list[GroupedVceIteration]
    groups: list[dict[str, object]]
    observations: list[dict[str, object]]
    normals: Optional[NormalEquations]

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "parameter_names": names_to_strings(self.parameter_names),
            "state": self.state,
            "gross_rejected_observations": {str(key): value for key, value in self.gross_rejected.items()},
            "scales": self.scales,
            "iterations": [asdict(item) for item in self.iterations],
            "groups": self.groups,
            "observations": self.observations,
        }


@dataclass
class _InnerSolution:
    equations: list[ObservationEquation]
    residuals: dict[ObsKey, float]
    normals: NormalEquations
    function_iterations: int
    converged: bool


class GroupedVceAdjustment:
    def __init__(
        self,
        *,
        equation_source: Callable[[int], list[ObservationEquation]],
        parametrization: ParametrizationList,
        options: GroupedVceOptions,
        context=None,
    ) -> None:
        self.equation_source = equation_source
        self.parametrization = parametrization
        self.options = options
        self.context = context
        self._equation_iteration = 0
        self._gross_rejected: dict[ObsKey, float] = {}
        self._assignments: dict[ObsKey, str] = {}
        self._names: list[ParameterName] = []

    def _equations(self) -> list[ObservationEquation]:
        self._equation_iteration += 1
        equations = self.equation_source(self._equation_iteration)
        return [eq for eq in equations if eq.converged and eq.identity not in self._gross_rejected]

    def _weight(self, scales: Mapping[str, float], factors: Mapping[ObsKey, float], equation: ObservationEquation) -> float:
        group = self._assignments[equation.identity]
        return float(factors[equation.identity]) / (scales[group] * scales[group] * equation.sigma_m * equation.sigma_m)

    def _inner_solve(
        self,
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> _InnerSolution:
        previous_wrms: Optional[float] = None
        last_normals: Optional[NormalEquations] = None
        converged = False
        used_iterations = 0

        for iteration in range(1, self.options.function_max_iterations + 1):
            equations = self._equations()
            usable = [
                eq
                for eq in equations
                if self._weight(scales, factors, eq) > self.options.minimum_nonzero_robust_factor
            ]
            if len(usable) < len(self._names):
                raise RuntimeError("Too few non-zero-weight observations for the current parameter set.")
            normals = build_normal_equations_streaming(
                usable,
                self.parametrization,
                parameter_names=self._names,
                weight_for=lambda eq: self._weight(scales, factors, eq),
            )
            solution = solve_normal_equations(normals)
            delta = self.options.function_damping * np.asarray(solution.delta, dtype=float)
            residuals = list(postfit_residuals_streaming(usable, self.parametrization, delta))
            sum_weight = sum(self._weight(scales, factors, item.equation) for item in residuals)
            wrms = (
                None
                if sum_weight <= 0.0
                else float(np.sqrt(sum(self._weight(scales, factors, item.equation) * item.residual_m**2 for item in residuals) / sum_weight))
            )
            updates = self.parametrization.apply_update(delta)
            max_update = max(updates.values()) if updates else 0.0
            used_iterations = iteration
            last_normals = normals
            wrms_change = None if previous_wrms is None or wrms is None else abs(wrms - previous_wrms)
            previous_wrms = wrms
            if max_update <= self.options.update_tolerance_m and (
                wrms_change is None or wrms_change <= self.options.wrms_tolerance_m
            ):
                converged = True
                break

        final_equations = self._equations()
        if last_normals is None:
            raise RuntimeError("Nonlinear solve produced no normal equations.")
        return _InnerSolution(
            equations=final_equations,
            residuals={eq.identity: float(self.parametrization.reduced_observation(eq)) for eq in final_equations},
            normals=last_normals,
            function_iterations=used_iterations,
            converged=converged,
        )

    def _base_standardized(
        self,
        solution: _InnerSolution,
        scales: Mapping[str, float],
    ) -> tuple[dict[ObsKey, float], dict[ObsKey, float]]:
        base_weights = {
            eq.identity: 1.0 / (scales[self._assignments[eq.identity]] ** 2 * eq.sigma_m**2)
            for eq in solution.equations
        }
        normals = build_normal_equations_streaming(
            solution.equations,
            self.parametrization,
            parameter_names=self._names,
            weight_for=lambda eq: base_weights[eq.identity],
        )
        covariance = solve_normal_equations(normals).covariance
        if covariance is None:
            raise RuntimeError("Base normal equation covariance is unavailable.")
        standardized: dict[ObsKey, float] = {}
        residual_sigmas: dict[ObsKey, float] = {}
        for equation in solution.equations:
            row = self.parametrization.design_row(equation)
            leverage = base_weights[equation.identity] * float(row @ covariance @ row)
            one_minus = max(1.0 - leverage, self.options.minimum_one_minus_leverage)
            variance = scales[self._assignments[equation.identity]] ** 2 * equation.sigma_m**2
            sigma_v = float(np.sqrt(variance * one_minus))
            residual_sigmas[equation.identity] = sigma_v
            standardized[equation.identity] = solution.residuals[equation.identity] / sigma_v
        return standardized, residual_sigmas

    def _update_scales(
        self,
        solution: _InnerSolution,
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> tuple[dict[str, float], dict[str, dict[str, float]], NormalEquations]:
        active = [
            eq
            for eq in solution.equations
            if factors[eq.identity] > self.options.minimum_nonzero_robust_factor
        ]
        normals = build_normal_equations_streaming(
            active,
            self.parametrization,
            parameter_names=self._names,
            weight_for=lambda eq: self._weight(scales, factors, eq),
        )
        solution_normals = solve_normal_equations(normals)
        rank = int(np.linalg.matrix_rank(normals.N))
        total_active = len(active)
        updates: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for group in self.options.groups:
            group_equations = [
                eq for eq in active if self._assignments[eq.identity] == group.id
            ]
            group_normals = build_normal_equations_streaming(
                group_equations,
                self.parametrization,
                parameter_names=self._names,
                weight_for=lambda eq: self._weight(scales, factors, eq),
            )
            consumed = float(np.trace(np.linalg.solve(normals.N, group_normals.N)))
            redundancy = float(len(group_equations) - consumed)
            numerator = float(
                sum(
                    factors[eq.identity] * solution.residuals[eq.identity] ** 2 / eq.sigma_m**2
                    for eq in group_equations
                )
            )
            raw_variance = (
                scales[group.id] ** 2
                if redundancy < self.options.minimum_effective_redundancy
                else numerator / redundancy
            )
            current_variance = scales[group.id] ** 2
            ratio = np.clip(
                raw_variance / current_variance,
                self.options.minimum_variance_ratio_per_iteration,
                self.options.maximum_variance_ratio_per_iteration,
            )
            damped_variance = current_variance * float(ratio) ** self.options.vce_damping
            updates[group.id] = float(np.sqrt(damped_variance))
            diagnostics[group.id] = {
                "active_count": float(len(group_equations)),
                "consumed_dof": consumed,
                "effective_redundancy": redundancy,
                "raw_variance": float(raw_variance),
                "update_status": (
                    "INSUFFICIENT_REDUNDANCY"
                    if redundancy < self.options.minimum_effective_redundancy
                    else "UPDATED"
                ),
            }
        total_redundancy = sum(item["effective_redundancy"] for item in diagnostics.values())
        expected = float(total_active - rank)
        if not np.isclose(total_redundancy, expected, rtol=1.0e-10, atol=1.0e-8):
            raise RuntimeError(
                f"Grouped redundancy check failed: {total_redundancy:.12g} != {expected:.12g}."
            )
        return updates, diagnostics, normals

    def _observation_records(
        self,
        equations: Sequence[ObservationEquation],
        residuals: Mapping[ObsKey, float],
        residual_sigmas: Mapping[ObsKey, float],
        standardized: Mapping[ObsKey, float],
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for equation in equations:
            factor = float(factors[equation.identity])
            group_id = self._assignments[equation.identity]
            matched_bias = [
                str(name)
                for index, name in enumerate(self._names)
                if name.type == "rangeBias" and self.parametrization.design_row(equation)[index] != 0.0
            ]
            status = "REJECTED" if factor <= self.options.minimum_nonzero_robust_factor else (
                "FULL_WEIGHT" if factor == 1.0 else "DOWNWEIGHTED"
            )
            records.append(
                {
                    "observation_id": str(equation.identity),
                    "epoch": equation.epoch.isot(),
                    "station_id": equation.station_key,
                    "station_system": next(group.station_system for group in self.options.groups if group.id == group_id),
                    "vce_group_id": group_id,
                    "sigma_np": float(equation.sigma_m),
                    "base_scale": float(scales[group_id]),
                    "base_variance": float(scales[group_id] ** 2 * equation.sigma_m**2),
                    "base_weight": float(1.0 / (scales[group_id] ** 2 * equation.sigma_m**2)),
                    "postfit_residual": float(residuals[equation.identity]),
                    "residual_sigma": float(residual_sigmas[equation.identity]),
                    "standardized_residual": float(standardized[equation.identity]),
                    "igg3_factor": factor,
                    "equivalent_weight": float(factor / (scales[group_id] ** 2 * equation.sigma_m**2)),
                    "robust_status": status,
                    "matched_bias_ids": matched_bias,
                }
            )
        return records

    def run(self) -> GroupedVceResult:
        initial_equations = self._equations()
        self.parametrization.setup(initial_equations, self.context)
        self._names = self.parametrization.parameter_names()
        self._gross_rejected = prefit_gross_rejections(
            initial_equations,
            self.parametrization,
            threshold_m=self.options.prefit_gross_threshold_m,
            threshold_by_station_m=self.options.prefit_gross_threshold_by_station_m,
        )
        active_initial = [
            eq for eq in initial_equations if eq.identity not in self._gross_rejected
        ]
        self.parametrization.setup(active_initial, self.context)
        self._names = self.parametrization.parameter_names()
        self._assignments = assign_vce_groups(active_initial, self.options.groups)

        bias_delta = robust_bias_initial_values(
            active_initial,
            self.parametrization,
            self._names,
            weight_cap=self.options.bias_weight_cap,
            maximum_iterations=self.options.bias_maximum_iterations,
        )
        self.parametrization.apply_update(bias_delta)
        scales = initialize_mad_scales(
            active_initial,
            self.parametrization,
            self._assignments,
            self.options.groups,
            minimum_count=self.options.minimum_mad_count,
            minimum_scale=self.options.minimum_initial_scale,
        )
        initial_scales = dict(scales)
        factors = {eq.identity: 1.0 for eq in active_initial}
        iterations: list[GroupedVceIteration] = []
        diagnostics: dict[str, dict[str, float]] = {}
        converged = False
        latest_normals: Optional[NormalEquations] = None

        for outer in range(1, self.options.maximum_stochastic_iterations + 1):
            base_solution = self._inner_solve(scales, factors)
            standardized, _ = self._base_standardized(base_solution, scales)
            next_factors = {
                key: float(value)
                for key, value in zip(
                    standardized,
                    igg3_factors(np.asarray(list(standardized.values())), k0=self.options.k0, k1=self.options.k1),
                )
            }
            robust_solution = self._inner_solve(scales, next_factors)
            next_scales, diagnostics, latest_normals = self._update_scales(
                robust_solution,
                scales,
                next_factors,
            )
            scale_change = max(
                abs(np.log((next_scales[group.id] ** 2) / (scales[group.id] ** 2)))
                for group in self.options.groups
            )
            factor_change = max(
                abs(next_factors[key] - factors.get(key, 1.0)) for key in next_factors
            )
            iterations.append(
                GroupedVceIteration(
                    iteration=outer,
                    function_iterations=base_solution.function_iterations + robust_solution.function_iterations,
                    maximum_scale_log_change=float(scale_change),
                    maximum_robust_factor_change=float(factor_change),
                    active_observation_count=sum(
                        value > self.options.minimum_nonzero_robust_factor
                        for value in next_factors.values()
                    ),
                    rejected_observation_count=sum(
                        value <= self.options.minimum_nonzero_robust_factor
                        for value in next_factors.values()
                    ),
                    total_effective_redundancy=float(
                        sum(item["effective_redundancy"] for item in diagnostics.values())
                    ),
                    expected_total_redundancy=float(
                        sum(item["active_count"] for item in diagnostics.values())
                        - np.linalg.matrix_rank(latest_normals.N)
                    ),
                    normal_matrix_condition=normal_matrix_condition(latest_normals),
                )
            )
            scales = next_scales
            factors = next_factors
            if (
                scale_change <= self.options.scale_log_tolerance
                and factor_change <= self.options.robust_weight_tolerance
                and robust_solution.converged
            ):
                converged = True
                break

        final_solution = self._inner_solve(scales, factors)
        standardized, residual_sigmas = self._base_standardized(final_solution, scales)
        group_records: list[dict[str, object]] = []
        for group in self.options.groups:
            group_equations = [
                eq for eq in final_solution.equations if self._assignments[eq.identity] == group.id
            ]
            residuals = np.asarray([final_solution.residuals[eq.identity] for eq in group_equations], dtype=float)
            standards = np.asarray([standardized[eq.identity] for eq in group_equations], dtype=float)
            item = dict(diagnostics.get(group.id, {}))
            item.update(
                {
                    "group_id": group.id,
                    "configured_start": group.start,
                    "configured_end": group.end_exclusive or "present",
                    "observation_count": len(group_equations),
                    "initial_scale": float(initial_scales[group.id]),
                    "final_scale": float(scales[group.id]),
                    "variance_component": float(scales[group.id] ** 2),
                    "residual_rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
                    "standardized_rms": float(np.sqrt(np.mean(standards**2))) if len(standards) else None,
                    "median_standardized_residual": float(np.median(standards)) if len(standards) else None,
                }
            )
            group_records.append(item)

        return GroupedVceResult(
            converged=converged,
            parameter_names=list(self._names),
            state=self.parametrization.state(),
            gross_rejected=dict(self._gross_rejected),
            scales=dict(scales),
            robust_factors=dict(factors),
            iterations=iterations,
            groups=group_records,
            observations=self._observation_records(
                final_solution.equations,
                final_solution.residuals,
                residual_sigmas,
                standardized,
                scales,
                factors,
            ),
            normals=final_solution.normals if final_solution.normals is not None else latest_normals,
        )


__all__ = [
    "GroupedVceAdjustment",
    "GroupedVceOptions",
    "GroupedVceResult",
    "VceGroup",
    "assign_vce_groups",
    "igg3_factors",
    "initialize_mad_scales",
    "prefit_gross_rejections",
]

