"""Single LLR measurement evaluation from resolved input to equation or row."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llrops.base.constants import C, C2
from llrops.classes.frames import ReferenceFrameSystem
from llrops.classes.range_bias.models import RangeBiasModel
from llrops.classes.relativistic.constants import MOON_EXTERNAL_POTENTIAL_BODIES
from llrops.classes.displacement.terrestrial_geometry import itrf2geodetic

from .equations import (
    ObservationEquation,
    ObservationOutputLevel,
    REFLECTOR_DESIGN_OUTPUT_FIELDS,
)
from .light_time import LightTimeRequest, LightTimeSolution, LightTimeSolver, OpticalAtmosphere
from .resolver import ResolvedObservation


@dataclass(frozen=True, slots=True)
class ObservationEvaluation:
    equation: ObservationEquation
    row: dict[str, object] | None = None


class LlrMeasurement:
    def __init__(
        self,
        frames: ReferenceFrameSystem,
        light_time_solver: LightTimeSolver,
        range_bias: RangeBiasModel,
    ) -> None:
        self.frames = frames
        self.light_time_solver = light_time_solver
        self.range_bias = range_bias

    @property
    def ephemeris(self):
        return self.frames.ephemeris

    def _reflector_position_partial_pa(
        self,
        solution: LightTimeSolution,
    ) -> np.ndarray:
        uplink_vector = (
            solution.reflector_bcrs_bounce_m - solution.station_bcrs_transmit_m
        )
        downlink_vector = (
            solution.reflector_bcrs_bounce_m - solution.station_bcrs_receive_m
        )
        uplink_range = max(float(np.linalg.norm(uplink_vector)), 1.0e-30)
        downlink_range = max(float(np.linalg.norm(downlink_vector)), 1.0e-30)
        unit_sum = uplink_vector / uplink_range + downlink_vector / downlink_range

        pa2lcrs = self.ephemeris.pa2lcrs_matrix(solution.bounce_epoch)
        moon_velocity = self.ephemeris.body_state_bcrs(
            "MOON", solution.bounce_epoch
        ).velocity_mps
        external_potential = self.frames.external_potential(
            "MOON",
            solution.bounce_epoch,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - self.ephemeris.lb_minus_ll - external_potential / C2
        jacobian = (
            scale * pa2lcrs
            - 0.5 * np.outer(moon_velocity, moon_velocity @ pa2lcrs) / C2
        )
        return np.asarray(0.5 * unit_sum @ jacobian, dtype=float).reshape(3)

    def evaluate(
        self,
        observation: ResolvedObservation,
        *,
        min_elevation_deg: float,
        include_reflector_position_partial: bool = False,
        output_level: ObservationOutputLevel | str | None = None,
    ) -> ObservationEvaluation:
        record = observation.record
        station = observation.station
        station_itrf_m = station.itrf_xyz_at(observation.transmit_epoch)
        geodetic = itrf2geodetic(station_itrf_m)
        solution = self.light_time_solver.solve(
            LightTimeRequest(
                station_reference_itrf_m=station_itrf_m,
                station_position_at_utc=station.itrf_xyz_at,
                station_id=observation.station_key,
                reflector_reference_pa_m=observation.reflector.moon_fixed_xyz_m,
                transmit_epoch=observation.transmit_epoch,
                atmosphere=OpticalAtmosphere(
                    pressure_hpa=record.pressure_hpa,
                    temperature_k=record.temperature_k,
                    relative_humidity_percent=float(record.humidity_percent),
                    latitude_rad=geodetic.latitude_rad,
                    height_m=geodetic.height_m,
                    wavelength_um=record.wavelength_um,
                ),
            )
        )

        elevation_up_deg = float(np.rad2deg(solution.uplink.elevation_rad))
        elevation_down_deg = float(np.rad2deg(solution.downlink.elevation_rad))
        below_horizon = (
            elevation_up_deg < min_elevation_deg
            or elevation_down_deg < min_elevation_deg
        )
        bias = self.range_bias.correction(
            observation.station_candidates,
            observation.transmit_epoch,
        )
        computed_raw_s = float(solution.observable_round_trip_time_s)
        computed_s = computed_raw_s - bias.two_way_s
        oc_raw_s = float(record.observed_round_trip_time_s) - computed_raw_s
        oc_s = float(record.observed_round_trip_time_s) - computed_s
        oc_raw_m = 0.5 * C * oc_raw_s
        oc_m = 0.5 * C * oc_s
        partial = (
            self._reflector_position_partial_pa(solution)
            if include_reflector_position_partial
            else None
        )
        partials = {} if partial is None else {"reflector_position_pa": partial}
        equation = ObservationEquation(
            observed_minus_computed_m=oc_m,
            sigma_m=record.range_uncertainty_one_way_m,
            partials=partials,
            identity=int(record.index),
            station_key=observation.station_key,
            reflector_key=observation.reflector_key,
            epoch=observation.transmit_epoch,
            converged=solution.converged,
            wavelength_nm=record.wavelength_nm,
        )
        if output_level is None:
            return ObservationEvaluation(equation)

        level = ObservationOutputLevel.parse(output_level)
        status = "below_horizon" if below_horizon else "ok"
        row: dict[str, object] = {
            "obs_time_utc": equation.epoch.isot(),
            "normal_point_index": equation.identity,
            "station_id": record.station_code or record.station_name,
            "station_name": station.name,
            "reflector_id": record.reflector_code,
            "reflector_name": observation.reflector.name,
            "observed_rtt_s": record.observed_round_trip_time_s,
            "computed_rtt_s": computed_s,
            "oc_one_way_m": oc_m,
            "fit_sigma_one_way_m": equation.sigma_m,
            "elevation_up_deg": elevation_up_deg,
            "converged": solution.converged,
            "status": status,
        }
        if partial is not None:
            row.update(
                {
                    name: float(value)
                    for name, value in zip(REFLECTOR_DESIGN_OUTPUT_FIELDS, partial)
                }
            )
        if level is ObservationOutputLevel.FULL:
            self._add_full_diagnostics(
                row,
                observation,
                solution,
                station_itrf_m,
                bias,
                computed_raw_s,
                oc_raw_s,
                oc_raw_m,
                elevation_down_deg,
                below_horizon,
            )
        return ObservationEvaluation(equation, row)

    def _add_full_diagnostics(
        self,
        row: dict[str, object],
        observation: ResolvedObservation,
        solution: LightTimeSolution,
        station_itrf_m: np.ndarray,
        bias,
        computed_raw_s: float,
        oc_raw_s: float,
        oc_raw_m: float,
        elevation_down_deg: float,
        below_horizon: bool,
    ) -> None:
        record = observation.record
        coordinate_rtt_s = float(solution.coordinate_round_trip_time_tdb_s)
        tt_minus_tdb_s = float(solution.tt_minus_tdb_interval_correction_s)
        utc_rate_correction_s = computed_raw_s - (coordinate_rtt_s + tt_minus_tdb_s)
        libration_rad = float(
            self.ephemeris.longitude_libration_correction_rad(solution.bounce_epoch)
        )
        tropo_up_used = solution.uplink.troposphere_elevation_used_rad
        tropo_down_used = solution.downlink.troposphere_elevation_used_rad
        row.update(
            {
                "sigma_definition": "0.5 * c * uncertainty_two_way_s",
                "pressure_hpa": record.pressure_hpa,
                "temperature_c": record.temperature_c,
                "humidity_percent": record.humidity_percent,
                "wavelength_nm": record.wavelength_nm,
                "computed_rtt_raw_s": computed_raw_s,
                "computed_rtt_tdb_s": coordinate_rtt_s,
                "range_bias_model": bias.model,
                "range_bias_two_way_cm": bias.two_way_cm,
                "range_bias_two_way_m": bias.two_way_m,
                "range_bias_two_way_s": bias.two_way_s,
                "range_bias_one_way_m": bias.one_way_m,
                "tt_minus_tdb_correction_s": tt_minus_tdb_s,
                "tt_minus_tdb_correction_one_way_m": 0.5 * C * tt_minus_tdb_s,
                "utc_rate_zeta": solution.utc_rate_zeta,
                "utc_rate_correction_s": utc_rate_correction_s,
                "utc_rate_correction_one_way_m": 0.5 * C * utc_rate_correction_s,
                "longitude_libration_correction_model": str(
                    self.ephemeris.longitude_libration_model
                ),
                "longitude_libration_correction_mas": float(
                    np.rad2deg(libration_rad) * 3_600_000.0
                ),
                "longitude_libration_correction_rad": libration_rad,
                "transmit_jd1": solution.transmit_epoch.jd1,
                "transmit_jd2": solution.transmit_epoch.jd2,
                "transmit_scale": solution.transmit_epoch.scale.value,
                "bounce_jd1": solution.bounce_epoch.jd1,
                "bounce_jd2": solution.bounce_epoch.jd2,
                "bounce_scale": solution.bounce_epoch.scale.value,
                "receive_jd1": solution.receive_epoch.jd1,
                "receive_jd2": solution.receive_epoch.jd2,
                "receive_scale": solution.receive_epoch.scale.value,
                "oc_rtt_s": 2.0 * row["oc_one_way_m"] / C,
                "oc_rtt_raw_s": oc_raw_s,
                "oc_one_way_raw_m": oc_raw_m,
                "range_uncertainty_one_way_m": row["fit_sigma_one_way_m"],
                "uncertainty_two_way_s": 2.0 * row["fit_sigma_one_way_m"] / C,
                "uncertainty_two_way_ps": 2.0e12 * row["fit_sigma_one_way_m"] / C,
                "rho_up_m": solution.uplink.geometric_range_m,
                "rho_down_m": solution.downlink.geometric_range_m,
                "rel_up_m": solution.uplink.gravitational_delay_m,
                "rel_down_m": solution.downlink.gravitational_delay_m,
                "tropo_up_m": solution.uplink.tropospheric_delay_m,
                "tropo_down_m": solution.downlink.tropospheric_delay_m,
                "tropo_elevation_up_used_deg": None
                if tropo_up_used is None
                else float(np.rad2deg(tropo_up_used)),
                "tropo_elevation_down_used_deg": None
                if tropo_down_used is None
                else float(np.rad2deg(tropo_down_used)),
                "tropo_up_clamped": solution.uplink.troposphere_elevation_clamped,
                "tropo_down_clamped": solution.downlink.troposphere_elevation_clamped,
                "tropo_clamped": bool(
                    solution.uplink.troposphere_elevation_clamped
                    or solution.downlink.troposphere_elevation_clamped
                ),
                "elevation_down_deg": elevation_down_deg,
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
                "station_itrf_x_m": float(station_itrf_m[0]),
                "station_itrf_y_m": float(station_itrf_m[1]),
                "station_itrf_z_m": float(station_itrf_m[2]),
                "valid_geometry": not below_horizon,
                "below_horizon": below_horizon,
                "station_catalog_key": observation.station_key,
                "reflector_catalog_key": observation.reflector_key,
            }
        )


__all__ = ["LlrMeasurement", "ObservationEvaluation"]
