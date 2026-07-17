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
    linearization_iteration: int
    stochastic_iteration: int
    maximum_scale_log_change: float
    maximum_robust_factor_change: float
    active_observation_count: int
    rejected_observation_count: int
    total_effective_redundancy: float
    expected_total_redundancy: float
    normal_matrix_condition: Optional[float]
    candidate_wrms_m: Optional[float]
    maximum_candidate_parameter_update_m: float
    candidate_update_by_block_m: dict[str, float]
    scales: dict[str, float]
    robust_factor_summary: dict[str, object]
    groups: dict[str, dict[str, object]]


@dataclass
class GroupedVceResult:
    converged: bool
    termination_reason: str
    settings: dict[str, object]
    equation_evaluations: list[dict[str, object]]
    parameter_names: list[ParameterName]
    state: dict[str, object]
    gross_rejected: dict[ObsKey, float]
    scales: dict[str, float]
    robust_factors: dict[ObsKey, float]
    iterations: list[GroupedVceIteration]
    linearizations: list[dict[str, object]]
    summary: dict[str, object]
    parameters: list[dict[str, object]]
    global_residuals: dict[str, object]
    groups: list[dict[str, object]]
    observations: list[dict[str, object]]
    normals: Optional[NormalEquations]

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "settings": self.settings,
            "equation_evaluations": self.equation_evaluations,
            "parameter_names": names_to_strings(self.parameter_names),
            "state": self.state,
            "gross_rejected_observations": {str(key): value for key, value in self.gross_rejected.items()},
            "scales": self.scales,
            "iterations": [asdict(item) for item in self.iterations],
            "linearizations": self.linearizations,
            "summary": self.summary,
            "parameters": self.parameters,
            "global_residuals": self.global_residuals,
            "groups": self.groups,
            "observations": self.observations,
        }


@dataclass
class _InnerSolution:
    equations: list[ObservationEquation]
    residuals: dict[ObsKey, float]
    normals: NormalEquations
    delta: np.ndarray
    wrms_m: Optional[float]


class GroupedVceAdjustment:
    def __init__(
        self,
        *,
        equation_source: Callable[[int], list[ObservationEquation]],
        parametrization: ParametrizationList,
        options: GroupedVceOptions,
        context=None,
        iteration_callback: Optional[Callable[[GroupedVceIteration], None]] = None,
    ) -> None:
        self.equation_source = equation_source
        self.parametrization = parametrization
        self.options = options
        self.context = context
        self.iteration_callback = iteration_callback
        self._equation_iteration = 0
        self._gross_rejected: dict[ObsKey, float] = {}
        self._assignments: dict[ObsKey, str] = {}
        self._retained_keys: Optional[set[ObsKey]] = None
        self._names: list[ParameterName] = []
        self._equation_evaluations: list[dict[str, object]] = []

    def _equations(self) -> list[ObservationEquation]:
        self._equation_iteration += 1
        equations = self.equation_source(self._equation_iteration)
        identities = [eq.identity for eq in equations]
        if len(set(identities)) != len(identities):
            duplicate = next(key for index, key in enumerate(identities) if key in identities[:index])
            raise ValueError(f"Observation identity {duplicate!r} is not unique.")

        active = [eq for eq in equations if eq.converged]
        retained = (
            active
            if self._retained_keys is None
            else [eq for eq in active if eq.identity in self._retained_keys]
        )
        self._equation_evaluations.append(
            {
                "linearization_iteration": self._equation_iteration,
                "source_observation_count": len(equations),
                "light_time_converged_count": len(active),
                "light_time_nonconverged_count": len(equations) - len(active),
                "fixed_domain_returned_count": len(retained),
                "converged_but_outside_fixed_domain_count": len(active) - len(retained),
            }
        )
        return retained

    def _weight(self, scales: Mapping[str, float], factors: Mapping[ObsKey, float], equation: ObservationEquation) -> float:
        group = self._assignments[equation.identity]
        return float(factors[equation.identity]) / (scales[group] * scales[group] * equation.sigma_m * equation.sigma_m)

    def _solve_linearized(
        self,
        equations: Sequence[ObservationEquation],
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> _InnerSolution:
        equations = list(equations)
        usable = [
            eq
            for eq in equations
            if factors[eq.identity] > self.options.minimum_nonzero_robust_factor
        ]
        if len(usable) < len(self._names):
            raise RuntimeError("Too few non-zero-factor observations for the current parameter set.")

        normals = build_normal_equations_streaming(
            usable,
            self.parametrization,
            parameter_names=self._names,
            weight_for=lambda eq: self._weight(scales, factors, eq),
        )
        solved = solve_normal_equations(normals)
        delta = np.asarray(solved.delta, dtype=float)
        residual_items = list(
            postfit_residuals_streaming(equations, self.parametrization, delta)
        )
        sum_weight = sum(
            self._weight(scales, factors, item.equation) for item in residual_items
        )
        wrms = (
            None
            if sum_weight <= 0.0
            else float(
                np.sqrt(
                    sum(
                        self._weight(scales, factors, item.equation)
                        * item.residual_m**2
                        for item in residual_items
                    )
                    / sum_weight
                )
            )
        )
        return _InnerSolution(
            equations=equations,
            residuals={
                item.equation.identity: float(item.residual_m)
                for item in residual_items
            },
            normals=normals,
            delta=delta,
            wrms_m=wrms,
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
        covariance = solve_normal_equations(normals).covariance
        if covariance is None:
            raise RuntimeError("Equivalent-weight normal equation covariance is unavailable.")
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
            consumed = float(np.trace(covariance @ group_normals.N))
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


    def _block_update_norms(self, delta: np.ndarray) -> dict[str, float]:
        return {
            f"{index}:{type(block).__name__}": float(
                block.max_update_norm(block_delta)
            )
            for index, (block, block_delta) in enumerate(
                zip(
                    self.parametrization.blocks,
                    self.parametrization.split(delta),
                )
            )
        }

    def _robust_factor_summary(
        self,
        equations: Sequence[ObservationEquation],
        factors: Mapping[ObsKey, float],
    ) -> dict[str, object]:
        values = np.asarray(
            [factors[eq.identity] for eq in equations],
            dtype=float,
        )
        if not len(values):
            return {
                "observation_count": 0,
                "full_weight_count": 0,
                "downweighted_count": 0,
                "rejected_count": 0,
            }
        threshold = self.options.minimum_nonzero_robust_factor
        active = values > threshold
        full = values == 1.0
        return {
            "observation_count": int(len(values)),
            "full_weight_count": int(np.count_nonzero(full)),
            "downweighted_count": int(np.count_nonzero(active & ~full)),
            "rejected_count": int(np.count_nonzero(~active)),
            "minimum": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }

    @staticmethod
    def _distribution_summary(values: np.ndarray) -> dict[str, object]:
        values = np.asarray(values, dtype=float)
        if not len(values):
            return {"count": 0}
        median = float(np.median(values))
        absolute = np.abs(values)
        return {
            "count": int(len(values)),
            "rms": float(np.sqrt(np.mean(values**2))),
            "median": median,
            "mad": 1.4826 * float(np.median(np.abs(values - median))),
            "absolute_p50": float(np.quantile(absolute, 0.50)),
            "absolute_p90": float(np.quantile(absolute, 0.90)),
            "absolute_p95": float(np.quantile(absolute, 0.95)),
            "absolute_p99": float(np.quantile(absolute, 0.99)),
            "absolute_maximum": float(np.max(absolute)),
        }

    def _residual_summary(
        self,
        solution: _InnerSolution,
        standardized: Mapping[ObsKey, float],
        scales: Mapping[str, float],
        factors: Mapping[ObsKey, float],
    ) -> dict[str, object]:
        equations = solution.equations
        residuals = np.asarray(
            [solution.residuals[eq.identity] for eq in equations],
            dtype=float,
        )
        standards = np.asarray(
            [standardized[eq.identity] for eq in equations],
            dtype=float,
        )
        weights = np.asarray(
            [self._weight(scales, factors, eq) for eq in equations],
            dtype=float,
        )
        weight_sum = float(np.sum(weights))
        return {
            "residual_m": self._distribution_summary(residuals),
            "standardized_residual": self._distribution_summary(standards),
            "equivalent_weighted_rms_m": (
                None
                if weight_sum <= 0.0
                else float(np.sqrt(np.sum(weights * residuals**2) / weight_sum))
            ),
            "robust_factors": self._robust_factor_summary(
                equations, factors
            ),
        }

    def _parameter_records(
        self,
        solution: _InnerSolution,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        solved = solve_normal_equations(solution.normals)
        covariance = solved.covariance
        if covariance is None:
            raise RuntimeError("Final parameter covariance is unavailable.")
        diagonal = np.maximum(np.diag(covariance), 0.0)
        cofactor_sigmas = np.sqrt(diagonal)
        denominator = np.outer(cofactor_sigmas, cofactor_sigmas)
        correlations = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
        records: list[dict[str, object]] = []
        for index, name in enumerate(self._names):
            candidates = np.abs(correlations[index]).copy()
            candidates[index] = -1.0
            correlated_index = (
                int(np.argmax(candidates)) if len(candidates) > 1 else None
            )
            records.append(
                {
                    "name": str(name),
                    "type": name.type,
                    "final_linearized_correction_m": float(solution.delta[index]),
                    "cofactor_sigma_m": float(cofactor_sigmas[index]),
                    "formal_sigma_m": (
                        None
                        if solved.sigma0_post is None
                        else float(solved.sigma0_post * cofactor_sigmas[index])
                    ),
                    "maximum_absolute_correlation": (
                        None
                        if correlated_index is None
                        else float(abs(correlations[index, correlated_index]))
                    ),
                    "maximum_correlated_parameter": (
                        None
                        if correlated_index is None
                        else str(self._names[correlated_index])
                    ),
                }
            )
        normal_summary = {
            "observation_count": int(solution.normals.obs_count),
            "parameter_count": len(self._names),
            "rank": int(np.linalg.matrix_rank(solution.normals.N)),
            "condition_number": normal_matrix_condition(solution.normals),
            "sigma0_post": solved.sigma0_post,
        }
        return records, normal_summary
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
        self._retained_keys = {eq.identity for eq in active_initial}
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
        current_equations = list(active_initial)
        iterations: list[GroupedVceIteration] = []
        linearizations: list[dict[str, object]] = []
        diagnostics: dict[str, dict[str, float]] = {}
        converged = False
        termination_reason = "PARAMETER_MODEL_NOT_CONVERGED"
        latest_normals: Optional[NormalEquations] = None
        final_solution: Optional[_InnerSolution] = None
        global_stochastic_iteration = 0

        for linearization in range(1, self.options.function_max_iterations + 1):
            stochastic_converged = False
            stochastic_iterations_used = 0

            for stochastic in range(1, self.options.maximum_stochastic_iterations + 1):
                base_solution = self._solve_linearized(
                    current_equations, scales, factors
                )
                standardized, _ = self._base_standardized(base_solution, scales)
                next_factor_values = igg3_factors(
                    np.asarray(list(standardized.values())),
                    k0=self.options.k0,
                    k1=self.options.k1,
                )
                next_factors = dict(factors)
                next_factors.update(
                    {
                        key: float(value)
                        for key, value in zip(standardized, next_factor_values)
                    }
                )
                robust_solution = self._solve_linearized(
                    current_equations, scales, next_factors
                )
                next_scales, diagnostics, latest_normals = self._update_scales(
                    robust_solution,
                    scales,
                    next_factors,
                )
                scale_change = max(
                    abs(
                        np.log(
                            (next_scales[group.id] ** 2)
                            / (scales[group.id] ** 2)
                        )
                    )
                    for group in self.options.groups
                )
                factor_change = max(
                    abs(next_factors[key] - factors[key])
                    for key in next_factors
                )
                current_factor_values = [
                    next_factors[eq.identity] for eq in current_equations
                ]
                candidate_update_by_block = self._block_update_norms(
                    robust_solution.delta
                )
                iteration_groups = {
                    group.id: {
                        **diagnostics[group.id],
                        "scale_before": float(scales[group.id]),
                        "scale_after": float(next_scales[group.id]),
                        "variance_before": float(scales[group.id] ** 2),
                        "variance_after": float(next_scales[group.id] ** 2),
                    }
                    for group in self.options.groups
                }
                global_stochastic_iteration += 1
                stochastic_iterations_used = stochastic
                iterations.append(
                    GroupedVceIteration(
                        iteration=global_stochastic_iteration,
                        linearization_iteration=linearization,
                        stochastic_iteration=stochastic,
                        maximum_scale_log_change=float(scale_change),
                        maximum_robust_factor_change=float(factor_change),
                        active_observation_count=sum(
                            value > self.options.minimum_nonzero_robust_factor
                            for value in current_factor_values
                        ),
                        rejected_observation_count=sum(
                            value <= self.options.minimum_nonzero_robust_factor
                            for value in current_factor_values
                        ),
                        total_effective_redundancy=float(
                            sum(
                                item["effective_redundancy"]
                                for item in diagnostics.values()
                            )
                        ),
                        expected_total_redundancy=float(
                            sum(
                                item["active_count"]
                                for item in diagnostics.values()
                            )
                            - np.linalg.matrix_rank(latest_normals.N)
                        ),
                        normal_matrix_condition=normal_matrix_condition(
                            latest_normals
                        ),
                        candidate_wrms_m=robust_solution.wrms_m,
                        maximum_candidate_parameter_update_m=max(
                            candidate_update_by_block.values(), default=0.0
                        ),
                        candidate_update_by_block_m=candidate_update_by_block,
                        scales={
                            key: float(value)
                            for key, value in next_scales.items()
                        },
                        robust_factor_summary=self._robust_factor_summary(
                            current_equations, next_factors
                        ),
                        groups=iteration_groups,
                    )
                )
                if self.iteration_callback is not None:
                    self.iteration_callback(iterations[-1])

                scales = next_scales
                factors = next_factors
                final_solution = robust_solution
                if (
                    scale_change <= self.options.scale_log_tolerance
                    and factor_change <= self.options.robust_weight_tolerance
                ):
                    stochastic_converged = True
                    break

            if final_solution is None:
                raise RuntimeError("Stochastic model produced no linearized solution.")

            final_solution = self._solve_linearized(
                current_equations, scales, factors
            )
            candidate_update_by_block = self._block_update_norms(
                final_solution.delta
            )
            maximum_update = max(candidate_update_by_block.values(), default=0.0)
            parameter_converged = (
                stochastic_converged and maximum_update <= self.options.update_tolerance_m
            )

            if stochastic_converged:
                applied_delta = (
                    final_solution.delta
                    if parameter_converged
                    else self.options.function_damping * final_solution.delta
                )
                applied_updates = self.parametrization.apply_update(applied_delta)
            else:
                applied_updates = {}

            linearizations.append(
                {
                    "iteration": linearization,
                    "stochastic_iterations": stochastic_iterations_used,
                    "stochastic_converged": stochastic_converged,
                    "maximum_parameter_update_m": float(maximum_update),
                    "parameter_converged": bool(parameter_converged),
                    "applied_update_by_block_m": applied_updates,
                    "wrms_m": final_solution.wrms_m,
                    "equation_count": len(current_equations),
                    "candidate_update_by_block_m": candidate_update_by_block,
                    "candidate_parameter_corrections_m": {
                        str(name): float(value)
                        for name, value in zip(
                            self._names, final_solution.delta
                        )
                    },
                    "scales": {
                        key: float(value) for key, value in scales.items()
                    },
                    "robust_factor_summary": self._robust_factor_summary(
                        current_equations, factors
                    ),
                    "normal_matrix_rank": int(
                        np.linalg.matrix_rank(final_solution.normals.N)
                    ),
                    "normal_matrix_condition": normal_matrix_condition(
                        final_solution.normals
                    ),
                    "state_after_update": self.parametrization.state(),
                }
            )

            if not stochastic_converged:
                termination_reason = "STOCHASTIC_MODEL_NOT_CONVERGED"
                break
            if parameter_converged:
                converged = True
                termination_reason = "CONVERGED"
                break
            if linearization < self.options.function_max_iterations:
                current_equations = self._equations()

        if final_solution is None:
            raise RuntimeError("Adjustment produced no final linearized solution.")

        standardized, residual_sigmas = self._base_standardized(
            final_solution, scales
        )
        _, diagnostics, latest_normals = self._update_scales(
            final_solution, scales, factors
        )
        parameter_records, normal_summary = self._parameter_records(
            final_solution
        )
        global_residuals = self._residual_summary(
            final_solution, standardized, scales, factors
        )
        settings = {
            "geometry_max_iterations": self.options.function_max_iterations,
            "geometry_damping": self.options.function_damping,
            "parameter_update_tolerance_m": self.options.update_tolerance_m,
            "stochastic_max_iterations_per_linearization": (
                self.options.maximum_stochastic_iterations
            ),
            "scale_log_tolerance": self.options.scale_log_tolerance,
            "robust_factor_tolerance": self.options.robust_weight_tolerance,
            "igg3_k0": self.options.k0,
            "igg3_k1": self.options.k1,
            "minimum_nonzero_robust_factor": (
                self.options.minimum_nonzero_robust_factor
            ),
            "vce_damping": self.options.vce_damping,
            "minimum_effective_redundancy": (
                self.options.minimum_effective_redundancy
            ),
        }
        first_evaluation = self._equation_evaluations[0]
        summary = {
            "converged": converged,
            "termination_reason": termination_reason,
            "source_observation_count": first_evaluation[
                "source_observation_count"
            ],
            "initial_light_time_converged_count": first_evaluation[
                "light_time_converged_count"
            ],
            "initial_light_time_nonconverged_count": first_evaluation[
                "light_time_nonconverged_count"
            ],
            "gross_rejected_count": len(self._gross_rejected),
            "retained_observation_count": len(self._retained_keys or ()),
            "final_equation_count": len(final_solution.equations),
            "equation_evaluation_count": len(self._equation_evaluations),
            "linearization_count": len(linearizations),
            "stochastic_iteration_count": len(iterations),
            **normal_summary,
        }
        group_records: list[dict[str, object]] = []
        for group in self.options.groups:
            group_equations = [
                eq
                for eq in final_solution.equations
                if self._assignments[eq.identity] == group.id
            ]
            residuals = np.asarray(
                [
                    final_solution.residuals[eq.identity]
                    for eq in group_equations
                ],
                dtype=float,
            )
            standards = np.asarray(
                [standardized[eq.identity] for eq in group_equations],
                dtype=float,
            )
            group_factors = self._robust_factor_summary(
                group_equations, factors
            )
            group_weights = np.asarray(
                [self._weight(scales, factors, eq) for eq in group_equations],
                dtype=float,
            )
            group_weight_sum = float(np.sum(group_weights))
            item = dict(diagnostics.get(group.id, {}))
            item.update(
                {
                    "group_id": group.id,
                    "configured_start": group.start,
                    "configured_end": group.end_exclusive or "present",
                    "actual_start_epoch": (
                        min(eq.epoch for eq in group_equations).isot()
                        if group_equations
                        else None
                    ),
                    "actual_end_epoch": (
                        max(eq.epoch for eq in group_equations).isot()
                        if group_equations
                        else None
                    ),
                    "observation_count": len(group_equations),
                    "retained_observation_count": sum(
                        assigned_group == group.id
                        for assigned_group in self._assignments.values()
                    ),
                    "initial_scale": float(initial_scales[group.id]),
                    "final_scale": float(scales[group.id]),
                    "variance_component": float(scales[group.id] ** 2),
                    "residual_rms": (
                        float(np.sqrt(np.mean(residuals**2)))
                        if len(residuals)
                        else None
                    ),
                    "standardized_rms": (
                        float(np.sqrt(np.mean(standards**2)))
                        if len(standards)
                        else None
                    ),
                    "median_standardized_residual": (
                        float(np.median(standards))
                        if len(standards)
                        else None
                    ),
                    "mad_standardized_residual": self._distribution_summary(standards).get("mad"),
                    "residual_wrms": None if group_weight_sum <= 0.0 else float(np.sqrt(np.sum(group_weights * residuals**2) / group_weight_sum)),
                    "residual_summary_m": self._distribution_summary(residuals),
                    "standardized_residual_summary": self._distribution_summary(standards),
                    "robust_factor_summary": group_factors,
                    "full_weight_count": group_factors["full_weight_count"],
                    "downweighted_count": group_factors["downweighted_count"],
                    "rejected_count": group_factors["rejected_count"],
                }
            )
            group_records.append(item)

        return GroupedVceResult(
            converged=converged,
            termination_reason=termination_reason,
            settings=settings,
            equation_evaluations=list(self._equation_evaluations),
            parameter_names=list(self._names),
            state=self.parametrization.state(),
            gross_rejected=dict(self._gross_rejected),
            scales=dict(scales),
            robust_factors=dict(factors),
            iterations=iterations,
            linearizations=linearizations,
            summary=summary,
            parameters=parameter_records,
            global_residuals=global_residuals,
            groups=group_records,
            observations=self._observation_records(
                final_solution.equations,
                final_solution.residuals,
                residual_sigmas,
                standardized,
                scales,
                factors,
            ),
            normals=(
                final_solution.normals
                if final_solution.normals is not None
                else latest_normals
            ),
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

