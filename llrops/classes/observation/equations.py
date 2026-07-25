"""Observation equations, diagnostics, and table projection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Hashable, Mapping, TYPE_CHECKING

import numpy as np

from llrops.base.constants import C
from llrops.base.epoch import Epoch, TimeScale

from .frozen_mapping import FrozenMapping

if TYPE_CHECKING:
    from .model import LlrPrediction
    from .reduction import ObservationReduction
    from .resolver import ResolvedObservation


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


STANDARD_OUTPUT_FIELDS = (
    "obs_time_utc",
    "normal_point_index",
    "station_id",
    "station_name",
    "reflector_id",
    "reflector_name",
    "observed_rtt_s",
    "computed_rtt_s",
    "oc_one_way_m",
    "fit_sigma_one_way_m",
    "elevation_up_deg",
    "converged",
    "status",
)

REFLECTOR_DESIGN_OUTPUT_FIELDS = (
    "design_reflector_dx",
    "design_reflector_dy",
    "design_reflector_dz",
)

_RESERVED_METADATA_FIELDS = frozenset(
    {
        "obs_time_utc",
        "normal_point_index",
        "oc_one_way_m",
        "oc_rtt_s",
        "fit_sigma_one_way_m",
        "range_uncertainty_one_way_m",
        "uncertainty_two_way_ps",
        "uncertainty_two_way_s",
        "converged",
        "station_catalog_key",
        "reflector_catalog_key",
        *REFLECTOR_DESIGN_OUTPUT_FIELDS,
    }
)


@dataclass(frozen=True, slots=True, eq=False)
class ObservationEquation:
    """One immutable linearized observation and its non-duplicated diagnostics."""

    observed_minus_computed_m: float
    sigma_m: float
    partials: Mapping[str, np.ndarray]
    identity: Hashable
    station_key: str
    reflector_key: str
    epoch: Epoch
    converged: bool = True
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        residual = float(self.observed_minus_computed_m)
        sigma = float(self.sigma_m)
        if not np.isfinite(residual):
            raise ValueError("observed_minus_computed_m must be finite.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_m must be positive and finite.")
        if not isinstance(self.epoch, Epoch):
            raise TypeError("epoch must be an Epoch.")
        self.epoch.require_scale(TimeScale.UTC, name="epoch")
        try:
            hash(self.identity)
        except TypeError as exc:
            raise TypeError("identity must be hashable.") from exc
        if not isinstance(self.station_key, str) or not self.station_key:
            raise TypeError("station_key must be a non-empty string.")
        if not isinstance(self.reflector_key, str) or not self.reflector_key:
            raise TypeError("reflector_key must be a non-empty string.")
        normalized: dict[str, np.ndarray] = {}
        for name, values in dict(self.partials).items():
            if not isinstance(name, str) or not name:
                raise TypeError("Partial-block names must be non-empty strings.")
            array = np.array(values, dtype=float, copy=True).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Partial block {name!r} contains non-finite values.")
            array.setflags(write=False)
            normalized[name] = array
        metadata = dict(self.metadata or {})
        if any(not isinstance(name, str) for name in metadata):
            raise TypeError("metadata field names must be strings.")
        duplicated = _RESERVED_METADATA_FIELDS.intersection(metadata)
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(
                f"metadata duplicates typed ObservationEquation fields: {names}."
            )
        object.__setattr__(self, "observed_minus_computed_m", residual)
        object.__setattr__(self, "sigma_m", sigma)
        object.__setattr__(self, "partials", FrozenMapping(normalized))
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "metadata", FrozenMapping(metadata))

    @property
    def normal_point_index(self) -> Hashable:
        return self.identity

    def _standard_row(self) -> dict[str, Any]:
        metadata = self.metadata or {}
        row = {
            "obs_time_utc": self.epoch.isot(scale=TimeScale.UTC),
            "normal_point_index": self.identity,
            "station_id": metadata.get("station_id", self.station_key),
            "station_name": metadata.get("station_name", self.station_key),
            "reflector_id": metadata.get("reflector_id", self.reflector_key),
            "reflector_name": metadata.get("reflector_name", self.reflector_key),
            "observed_rtt_s": metadata.get("observed_rtt_s"),
            "computed_rtt_s": metadata.get("computed_rtt_s"),
            "oc_one_way_m": self.observed_minus_computed_m,
            "fit_sigma_one_way_m": self.sigma_m,
            "elevation_up_deg": metadata.get("elevation_up_deg"),
            "converged": self.converged,
            "status": metadata.get("status"),
        }
        partial = self.partials.get("reflector_position_pa")
        if partial is not None:
            values = np.asarray(partial, dtype=float).reshape(3)
            row.update(
                {
                    name: float(value)
                    for name, value in zip(
                        REFLECTOR_DESIGN_OUTPUT_FIELDS,
                        values,
                    )
                }
            )
        return row

    def to_row(
        self,
        level: ObservationOutputLevel | str = ObservationOutputLevel.STANDARD,
    ) -> dict[str, Any]:
        level = ObservationOutputLevel.parse(level)
        row = self._standard_row()
        if level is ObservationOutputLevel.STANDARD:
            return row
        for name, value in (self.metadata or {}).items():
            row.setdefault(name, value)
        row["oc_rtt_s"] = 2.0 * self.observed_minus_computed_m / C
        row["range_uncertainty_one_way_m"] = self.sigma_m
        row["uncertainty_two_way_s"] = 2.0 * self.sigma_m / C
        row["uncertainty_two_way_ps"] = 2.0e12 * self.sigma_m / C
        row["station_catalog_key"] = self.station_key
        row["reflector_catalog_key"] = self.reflector_key
        return row


def build_observation_equation(
    observation: "ResolvedObservation",
    prediction: "LlrPrediction",
    reduction: "ObservationReduction",
) -> ObservationEquation:
    """Project the typed forward-model stages into their single final form."""
    record = observation.record
    solution = prediction.light_time
    bias = reduction.range_bias
    station_itrf_m = observation.station.itrf_xyz_at(observation.transmit_epoch)

    metadata = {
        "observed_rtt_s": record.observed_round_trip_time_s,
        "reflector_id": record.reflector_code,
        "station_id": record.station_code or record.station_name,
        "sigma_definition": reduction.sigma_definition,
        "pressure_hpa": record.pressure_hpa,
        "temperature_c": record.temperature_c,
        "humidity_percent": record.humidity_percent,
        "wavelength_nm": record.wavelength_nm,
        "computed_rtt_s": reduction.computed_rtt_s,
        "computed_rtt_raw_s": reduction.computed_rtt_raw_s,
        "computed_rtt_tdb_s": reduction.coordinate_round_trip_time_tdb_s,
        "range_bias_model": bias.model,
        "range_bias_two_way_cm": bias.two_way_cm,
        "range_bias_two_way_m": bias.two_way_m,
        "range_bias_two_way_s": bias.two_way_s,
        "range_bias_one_way_m": bias.one_way_m,
        "tt_minus_tdb_correction_s": reduction.tt_minus_tdb_interval_correction_s,
        "tt_minus_tdb_correction_one_way_m": (
            0.5 * C * reduction.tt_minus_tdb_interval_correction_s
        ),
        "utc_rate_zeta": solution.utc_rate_zeta,
        "utc_rate_correction_s": reduction.utc_rate_correction_s,
        "utc_rate_correction_one_way_m": 0.5 * C * reduction.utc_rate_correction_s,
        "longitude_libration_correction_model": reduction.longitude_libration_model,
        "longitude_libration_correction_mas": reduction.longitude_libration_correction_mas,
        "longitude_libration_correction_rad": reduction.longitude_libration_correction_rad,
        "transmit_jd1": solution.transmit_epoch.jd1,
        "transmit_jd2": solution.transmit_epoch.jd2,
        "transmit_scale": solution.transmit_epoch.scale.value,
        "bounce_jd1": solution.bounce_epoch.jd1,
        "bounce_jd2": solution.bounce_epoch.jd2,
        "bounce_scale": solution.bounce_epoch.scale.value,
        "receive_jd1": solution.receive_epoch.jd1,
        "receive_jd2": solution.receive_epoch.jd2,
        "receive_scale": solution.receive_epoch.scale.value,
        "oc_rtt_raw_s": reduction.observed_minus_computed_raw_rtt_s,
        "oc_one_way_raw_m": reduction.observed_minus_computed_raw_one_way_m,
        "rho_up_m": solution.uplink.geometric_range_m,
        "rho_down_m": solution.downlink.geometric_range_m,
        "rel_up_m": solution.uplink.gravitational_delay_m,
        "rel_down_m": solution.downlink.gravitational_delay_m,
        "tropo_up_m": solution.uplink.tropospheric_delay_m,
        "tropo_down_m": solution.downlink.tropospheric_delay_m,
        "tropo_elevation_up_used_deg": reduction.troposphere_elevation_up_used_deg,
        "tropo_elevation_down_used_deg": reduction.troposphere_elevation_down_used_deg,
        "tropo_up_clamped": reduction.troposphere_up_clamped,
        "tropo_down_clamped": reduction.troposphere_down_clamped,
        "tropo_clamped": reduction.troposphere_clamped,
        "elevation_up_deg": reduction.elevation_up_deg,
        "elevation_down_deg": reduction.elevation_down_deg,
        "station_displacement_transmit_dx_m": float(
            solution.station_displacement_transmit_itrf_m[0]
        ),
        "station_displacement_transmit_dy_m": float(
            solution.station_displacement_transmit_itrf_m[1]
        ),
        "station_displacement_transmit_dz_m": float(
            solution.station_displacement_transmit_itrf_m[2]
        ),
        "station_displacement_receive_dx_m": float(
            solution.station_displacement_receive_itrf_m[0]
        ),
        "station_displacement_receive_dy_m": float(
            solution.station_displacement_receive_itrf_m[1]
        ),
        "station_displacement_receive_dz_m": float(
            solution.station_displacement_receive_itrf_m[2]
        ),
        "reflector_displacement_bounce_dx_m": float(
            solution.reflector_displacement_bounce_pa_m[0]
        ),
        "reflector_displacement_bounce_dy_m": float(
            solution.reflector_displacement_bounce_pa_m[1]
        ),
        "reflector_displacement_bounce_dz_m": float(
            solution.reflector_displacement_bounce_pa_m[2]
        ),
        "iterations": solution.iterations,
        "station_name": observation.station.name,
        "station_itrf_x_m": float(station_itrf_m[0]),
        "station_itrf_y_m": float(station_itrf_m[1]),
        "station_itrf_z_m": float(station_itrf_m[2]),
        "reflector_name": observation.reflector.name,
        "valid_geometry": reduction.valid_geometry,
        "below_horizon": reduction.below_horizon,
        "status": reduction.status,
    }
    partials: dict[str, np.ndarray] = {
        "station_range_bias": np.array([1.0], dtype=float),
    }
    if prediction.reflector_position_partial_pa is not None:
        partials["reflector_position_pa"] = prediction.reflector_position_partial_pa

    return ObservationEquation(
        observed_minus_computed_m=reduction.observed_minus_computed_one_way_m,
        sigma_m=record.range_uncertainty_one_way_m,
        partials=partials,
        identity=int(record.index),
        station_key=observation.station_key,
        reflector_key=observation.reflector_key,
        epoch=observation.transmit_epoch,
        converged=solution.converged,
        metadata=metadata,
    )


__all__ = [
    "ObservationEquation",
    "ObservationOutputLevel",
    "REFLECTOR_DESIGN_OUTPUT_FIELDS",
    "STANDARD_OUTPUT_FIELDS",
    "build_observation_equation",
]
