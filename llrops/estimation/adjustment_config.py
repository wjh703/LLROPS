"""Strict configuration schema for the nonlinear LLR adjustment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from llrops.base.station_identity import canonical_station_id
from llrops.estimation.adjustment_options import LlrAdjustmentOptions
from llrops.estimation.variance_components import VarianceComponentDefinition


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings.")
    return value


def _reject_unknown(
    value: Mapping[str, object],
    allowed: set[str],
    path: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}.")


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    return int(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string.")
    return value.strip()


def _string_sequence(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of strings.")
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must not contain duplicates.")
    return result


def _number_mapping(
    value: object,
    path: str,
    *,
    allow_none: bool = False,
) -> dict[str, float | None]:
    source = _mapping(value, path)
    result: dict[str, float | None] = {}
    for raw_key, raw_value in source.items():
        key = _string(raw_key, f"{path} key")
        if raw_value is None and allow_none:
            result[key] = None
        else:
            result[key] = _number(raw_value, f"{path}.{key}")
    return result


@dataclass(frozen=True)
class LlrAdjustmentStage:
    name: str
    parametrizations: tuple[str, ...] = ()
    maximum_linearizations: int | None = None
    parameter_update_factor: float | None = None
    update_tolerance_m: float | None = None
    required_consecutive_converged_linearizations: int | None = None

    def apply(self, options: LlrAdjustmentOptions) -> LlrAdjustmentOptions:
        overrides = {
            field: value
            for field, value in (
                ("maximum_linearizations", self.maximum_linearizations),
                ("parameter_update_factor", self.parameter_update_factor),
                ("update_tolerance_m", self.update_tolerance_m),
                (
                    "required_consecutive_converged_linearizations",
                    self.required_consecutive_converged_linearizations,
                ),
            )
            if value is not None
        }
        return replace(options, **overrides)

    def validate(self, options: LlrAdjustmentOptions) -> None:
        self.apply(options)


@dataclass(frozen=True)
class LlrAdjustmentPlan:
    options: LlrAdjustmentOptions
    stages: tuple[LlrAdjustmentStage, ...]
    warm_start_stochastic_model_across_stages: bool


_ADJUSTMENT_KEYS = {
    "maximumLinearizations",
    "parameterUpdateFactor",
    "prefitGrossThresholdByStationM",
    "prefitGrossThresholdM",
    "requiredConsecutiveConvergedLinearizations",
    "stages",
    "uncertaintyFloor",
    "updateToleranceByBlockM",
    "updateToleranceM",
    "warmStartStochasticModelAcrossStages",
}
_UNCERTAINTY_FLOOR_KEYS = {
    "minimumFractionOfGroupMedian",
    "minimumOneWaySigmaM",
}
_INITIALIZATION_KEYS = {
    "biasMaximumIterations",
    "biasWeightCap",
    "minimumInitialScale",
    "minimumMadCount",
}
_ROBUST_KEYS = {
    "activeFactorThreshold",
    "changeQuantile",
    "convergenceFactorFloor",
    "k0",
    "k1",
    "minimumOneMinusLeverage",
}
_VCE_KEYS = {
    "components",
    "maximumIterations",
    "maximumVarianceRatioPerIteration",
    "minimumEffectiveRedundancy",
    "minimumVarianceRatioPerIteration",
    "scaleLogTolerance",
    "targetActiveSetChangeTolerance",
    "targetFactorChangeTolerance",
}
_STAGE_KEYS = {
    "maximumLinearizations",
    "name",
    "parameterUpdateFactor",
    "parametrizations",
    "requiredConsecutiveConvergedLinearizations",
    "updateToleranceM",
}


def _parse_stages(value: object) -> tuple[LlrAdjustmentStage, ...]:
    if value is None:
        return (LlrAdjustmentStage(name="joint"),)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("adjustment.stages must be a sequence.")
    stages: list[LlrAdjustmentStage] = []
    for index, raw_stage in enumerate(value):
        path = f"adjustment.stages[{index}]"
        stage = _mapping(raw_stage, path)
        _reject_unknown(stage, _STAGE_KEYS, path)
        stages.append(
            LlrAdjustmentStage(
                name=_string(stage.get("name"), f"{path}.name"),
                parametrizations=(
                    ()
                    if "parametrizations" not in stage
                    else _string_sequence(
                        stage["parametrizations"], f"{path}.parametrizations"
                    )
                ),
                maximum_linearizations=(
                    None
                    if "maximumLinearizations" not in stage
                    else _integer(
                        stage["maximumLinearizations"],
                        f"{path}.maximumLinearizations",
                    )
                ),
                parameter_update_factor=(
                    None
                    if "parameterUpdateFactor" not in stage
                    else _number(
                        stage["parameterUpdateFactor"],
                        f"{path}.parameterUpdateFactor",
                    )
                ),
                update_tolerance_m=(
                    None
                    if "updateToleranceM" not in stage
                    else _number(
                        stage["updateToleranceM"], f"{path}.updateToleranceM"
                    )
                ),
                required_consecutive_converged_linearizations=(
                    None
                    if "requiredConsecutiveConvergedLinearizations" not in stage
                    else _integer(
                        stage["requiredConsecutiveConvergedLinearizations"],
                        f"{path}.requiredConsecutiveConvergedLinearizations",
                    )
                ),
            )
        )
    if not stages:
        raise ValueError("adjustment.stages must contain at least one stage.")
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError("adjustment.stages names must be unique.")
    return tuple(stages)


def parse_adjustment_plan(config: Mapping[str, object]) -> LlrAdjustmentPlan:
    """Parse one canonical schema and reject obsolete aliases."""

    if "robust_estimation" in config:
        raise ValueError(
            "robust_estimation was removed; use robustEstimation."
        )
    adjustment = _mapping(config.get("adjustment"), "adjustment")
    initialization = _mapping(config.get("initialization"), "initialization")
    robust = _mapping(config.get("robustEstimation"), "robustEstimation")
    vce = _mapping(config.get("vce"), "vce")
    uncertainty = _mapping(
        adjustment.get("uncertaintyFloor"),
        "adjustment.uncertaintyFloor",
    )
    _reject_unknown(adjustment, _ADJUSTMENT_KEYS, "adjustment")
    _reject_unknown(initialization, _INITIALIZATION_KEYS, "initialization")
    _reject_unknown(robust, _ROBUST_KEYS, "robustEstimation")
    _reject_unknown(vce, _VCE_KEYS, "vce")
    _reject_unknown(
        uncertainty,
        _UNCERTAINTY_FLOOR_KEYS,
        "adjustment.uncertaintyFloor",
    )

    raw_components = vce.get("components")
    if isinstance(raw_components, (str, bytes)) or not isinstance(
        raw_components, Sequence
    ):
        raise TypeError("vce.components must be a sequence.")
    components = tuple(
        VarianceComponentDefinition.from_config(
            _mapping(item, f"vce.components[{index}]")
        )
        for index, item in enumerate(raw_components)
    )
    defaults = LlrAdjustmentOptions(components=components)
    station_thresholds = _number_mapping(
        adjustment.get("prefitGrossThresholdByStationM"),
        "adjustment.prefitGrossThresholdByStationM",
        allow_none=True,
    )
    canonical_thresholds = {
        canonical_station_id(station): threshold
        for station, threshold in station_thresholds.items()
    }
    if len(canonical_thresholds) != len(station_thresholds):
        raise ValueError(
            "adjustment.prefitGrossThresholdByStationM contains duplicate "
            "canonical station identifiers."
        )
    block_tolerances = _number_mapping(
        adjustment.get("updateToleranceByBlockM"),
        "adjustment.updateToleranceByBlockM",
    )
    prefit_threshold = adjustment.get(
        "prefitGrossThresholdM", defaults.prefit_gross_threshold_m
    )

    options = LlrAdjustmentOptions(
        components=components,
        prefit_gross_threshold_m=(
            None
            if prefit_threshold is None
            else _number(prefit_threshold, "adjustment.prefitGrossThresholdM")
        ),
        prefit_gross_threshold_by_station_m=canonical_thresholds or None,
        maximum_linearizations=_integer(
            adjustment.get("maximumLinearizations", defaults.maximum_linearizations),
            "adjustment.maximumLinearizations",
        ),
        parameter_update_factor=_number(
            adjustment.get("parameterUpdateFactor", defaults.parameter_update_factor),
            "adjustment.parameterUpdateFactor",
        ),
        uncertainty_floor_minimum_m=_number(
            uncertainty.get("minimumOneWaySigmaM", defaults.uncertainty_floor_minimum_m),
            "adjustment.uncertaintyFloor.minimumOneWaySigmaM",
        ),
        uncertainty_floor_group_median_fraction=_number(
            uncertainty.get(
                "minimumFractionOfGroupMedian",
                defaults.uncertainty_floor_group_median_fraction,
            ),
            "adjustment.uncertaintyFloor.minimumFractionOfGroupMedian",
        ),
        update_tolerance_m=_number(
            adjustment.get("updateToleranceM", defaults.update_tolerance_m),
            "adjustment.updateToleranceM",
        ),
        update_tolerance_by_block_m={
            key: float(value) for key, value in block_tolerances.items()
        },
        required_consecutive_converged_linearizations=_integer(
            adjustment.get(
                "requiredConsecutiveConvergedLinearizations",
                defaults.required_consecutive_converged_linearizations,
            ),
            "adjustment.requiredConsecutiveConvergedLinearizations",
        ),
        maximum_stochastic_iterations=_integer(
            vce.get("maximumIterations", defaults.maximum_stochastic_iterations),
            "vce.maximumIterations",
        ),
        k0=_number(robust.get("k0", defaults.k0), "robustEstimation.k0"),
        k1=_number(robust.get("k1", defaults.k1), "robustEstimation.k1"),
        minimum_one_minus_leverage=_number(
            robust.get(
                "minimumOneMinusLeverage", defaults.minimum_one_minus_leverage
            ),
            "robustEstimation.minimumOneMinusLeverage",
        ),
        minimum_nonzero_robust_factor=_number(
            robust.get(
                "activeFactorThreshold", defaults.minimum_nonzero_robust_factor
            ),
            "robustEstimation.activeFactorThreshold",
        ),
        minimum_robust_factor_for_convergence=_number(
            robust.get(
                "convergenceFactorFloor",
                defaults.minimum_robust_factor_for_convergence,
            ),
            "robustEstimation.convergenceFactorFloor",
        ),
        robust_factor_change_quantile=_number(
            robust.get("changeQuantile", defaults.robust_factor_change_quantile),
            "robustEstimation.changeQuantile",
        ),
        minimum_mad_count=_integer(
            initialization.get("minimumMadCount", defaults.minimum_mad_count),
            "initialization.minimumMadCount",
        ),
        minimum_initial_scale=_number(
            initialization.get(
                "minimumInitialScale", defaults.minimum_initial_scale
            ),
            "initialization.minimumInitialScale",
        ),
        bias_weight_cap=_number(
            initialization.get("biasWeightCap", defaults.bias_weight_cap),
            "initialization.biasWeightCap",
        ),
        bias_maximum_iterations=_integer(
            initialization.get(
                "biasMaximumIterations", defaults.bias_maximum_iterations
            ),
            "initialization.biasMaximumIterations",
        ),
        minimum_effective_redundancy=_number(
            vce.get(
                "minimumEffectiveRedundancy", defaults.minimum_effective_redundancy
            ),
            "vce.minimumEffectiveRedundancy",
        ),
        scale_log_tolerance=_number(
            vce.get("scaleLogTolerance", defaults.scale_log_tolerance),
            "vce.scaleLogTolerance",
        ),
        robust_factor_change_tolerance=_number(
            vce.get(
                "targetFactorChangeTolerance",
                defaults.robust_factor_change_tolerance,
            ),
            "vce.targetFactorChangeTolerance",
        ),
        active_set_change_tolerance=_number(
            vce.get(
                "targetActiveSetChangeTolerance",
                defaults.active_set_change_tolerance,
            ),
            "vce.targetActiveSetChangeTolerance",
        ),
        minimum_variance_ratio_per_iteration=_number(
            vce.get(
                "minimumVarianceRatioPerIteration",
                defaults.minimum_variance_ratio_per_iteration,
            ),
            "vce.minimumVarianceRatioPerIteration",
        ),
        maximum_variance_ratio_per_iteration=_number(
            vce.get(
                "maximumVarianceRatioPerIteration",
                defaults.maximum_variance_ratio_per_iteration,
            ),
            "vce.maximumVarianceRatioPerIteration",
        ),
    )
    stages = _parse_stages(adjustment.get("stages"))
    for stage in stages:
        stage.validate(options)
    return LlrAdjustmentPlan(
        options=options,
        stages=stages,
        warm_start_stochastic_model_across_stages=_boolean(
            adjustment.get("warmStartStochasticModelAcrossStages", True),
            "adjustment.warmStartStochasticModelAcrossStages",
        ),
    )


__all__ = [
    "LlrAdjustmentPlan",
    "LlrAdjustmentStage",
    "parse_adjustment_plan",
]
