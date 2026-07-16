"""Typed observation results and table projection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from llrops.base.epoch import Epoch, TimeScale

from .containers import FrozenMapping


class ObservationOutputLevel(str, Enum):
    STANDARD = "standard"
    FULL = "full"

    @classmethod
    def parse(cls, value: object) -> "ObservationOutputLevel":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or cls.STANDARD.value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unknown observation output level {value!r}.") from exc


@dataclass(frozen=True, slots=True)
class OutputField:
    name: str
    description: str = ""


STANDARD_OUTPUT_SCHEMA = (
    OutputField("obs_time_utc", "transmit epoch formatted in UTC"),
    OutputField("normal_point_index"),
    OutputField("station_id"),
    OutputField("station_name"),
    OutputField("station_catalog_key"),
    OutputField("reflector_id"),
    OutputField("reflector_name"),
    OutputField("reflector_catalog_key"),
    OutputField("observed_rtt_s"),
    OutputField("computed_rtt_s"),
    OutputField("computed_rtt_raw_s"),
    OutputField("range_bias_model"),
    OutputField("range_bias_two_way_cm"),
    OutputField("range_bias_one_way_m"),
    OutputField("oc_rtt_s"),
    OutputField("oc_one_way_m"),
    OutputField("oc_rtt_raw_s"),
    OutputField("oc_one_way_raw_m"),
    OutputField("rho_up_m"),
    OutputField("rho_down_m"),
    OutputField("rel_up_m"),
    OutputField("rel_down_m"),
    OutputField("tropo_up_m"),
    OutputField("tropo_down_m"),
    OutputField("tropo_elevation_up_used_deg"),
    OutputField("tropo_elevation_down_used_deg"),
    OutputField("tropo_up_clamped"),
    OutputField("tropo_down_clamped"),
    OutputField("tropo_clamped"),
    OutputField("utc_rate_zeta"),
    OutputField("utc_rate_correction_one_way_m"),
    OutputField("longitude_libration_correction_model"),
    OutputField("longitude_libration_correction_mas"),
    OutputField("longitude_libration_correction_rad"),
    OutputField("uncertainty_model"),
    OutputField("range_uncertainty_one_way_m"),
    OutputField("fit_sigma_one_way_m"),
    OutputField("wrms_sigma_one_way_m"),
    OutputField("mini_range_uncertainty_one_way_m"),
    OutputField("uncertainty_two_way_ps"),
    OutputField("uncertainty_source"),
    OutputField("uncertainty_group"),
    OutputField("wrms_two_way_m"),
    OutputField("pressure_hpa"),
    OutputField("temperature_c"),
    OutputField("humidity_percent"),
    OutputField("wavelength_nm"),
    OutputField("elevation_up_deg"),
    OutputField("elevation_down_deg"),
    OutputField("iterations"),
    OutputField("converged"),
    OutputField("valid_geometry"),
    OutputField("below_horizon"),
    OutputField("status"),
)

STANDARD_OUTPUT_FIELDS = tuple(field.name for field in STANDARD_OUTPUT_SCHEMA)

REFLECTOR_DESIGN_OUTPUT_FIELDS = (
    "design_reflector_dx",
    "design_reflector_dy",
    "design_reflector_dz",
)


def required_output_fields(*, include_reflector_design: bool = False) -> tuple[str, ...]:
    fields = list(STANDARD_OUTPUT_FIELDS)
    if include_reflector_design:
        fields.extend(REFLECTOR_DESIGN_OUTPUT_FIELDS)
    return tuple(fields)


def missing_output_fields(values: Mapping[str, Any], *, include_reflector_design: bool = False) -> tuple[str, ...]:
    keys = set(values)
    return tuple(name for name in required_output_fields(include_reflector_design=include_reflector_design) if name not in keys)


def assert_output_schema(values: Mapping[str, Any], *, include_reflector_design: bool = False) -> None:
    missing = missing_output_fields(values, include_reflector_design=include_reflector_design)
    if missing:
        raise RuntimeError(f"Observation output schema is missing required fields: {list(missing)!r}")


@dataclass(frozen=True, slots=True, eq=False)
class LlrObservationResult:
    """One processed normal point, independent of its serialized table shape."""

    normal_point_index: int
    station_key: str
    reflector_key: str
    epoch: Epoch
    observed_minus_computed_m: float
    sigma_one_way_m: float
    converged: bool
    partials: Mapping[str, np.ndarray]
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        index = int(self.normal_point_index)
        sigma = float(self.sigma_one_way_m)
        residual = float(self.observed_minus_computed_m)
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch.")
        self.epoch.require_scale(TimeScale.UTC, name="epoch")
        if index < 0:
            raise ValueError("normal_point_index must be non-negative.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_one_way_m must be positive and finite.")
        if not np.isfinite(residual):
            raise ValueError("observed_minus_computed_m must be finite.")
        normalized_partials: dict[str, np.ndarray] = {}
        for name, value in dict(self.partials).items():
            array = np.array(value, dtype=float, copy=True).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Partial block {name!r} contains non-finite values.")
            array.setflags(write=False)
            normalized_partials[str(name)] = array
        object.__setattr__(self, "normal_point_index", index)
        object.__setattr__(self, "sigma_one_way_m", sigma)
        object.__setattr__(self, "observed_minus_computed_m", residual)
        object.__setattr__(self, "partials", FrozenMapping(normalized_partials))
        object.__setattr__(self, "values", FrozenMapping(self.values))

    def to_row(
        self,
        level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
    ) -> dict[str, Any]:
        level = ObservationOutputLevel.parse(level)
        row = dict(self.values)
        if level is ObservationOutputLevel.FULL:
            return row
        fields = required_output_fields(
            include_reflector_design="reflector_position_pa" in self.partials
        )
        return {name: row.get(name) for name in fields if name in row}

    def to_equation(self):
        from .equations import ObservationEquation

        return ObservationEquation(
            observed_minus_computed_m=self.observed_minus_computed_m,
            sigma_m=self.sigma_one_way_m,
            partials=self.partials,
            identity=self.normal_point_index,
            station_key=self.station_key,
            reflector_key=self.reflector_key,
            epoch=self.epoch,
            converged=self.converged,
            metadata=self.values,
        )


__all__ = [
    "LlrObservationResult",
    "ObservationOutputLevel",
    "OutputField",
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_SCHEMA",
    "assert_output_schema",
    "missing_output_fields",
    "required_output_fields",
]
