"""Assembly of typed observation stages into a transportable result."""
from __future__ import annotations

import numpy as np

from llrops.base.constants import C
from llrops.base.epoch import TimeScale

from .model import LlrPrediction
from .reduction import ObservationReduction
from .resolver import ResolvedObservation
from .results import LlrObservationResult, assert_output_schema


class LlrObservationResultBuilder:
    def build(
        self,
        observation: ResolvedObservation,
        prediction: LlrPrediction,
        reduction: ObservationReduction,
    ) -> LlrObservationResult:
        record = observation.record
        solution = prediction.light_time
        selected = reduction.selected_uncertainty
        mini = reduction.mini_uncertainty
        wrms = reduction.wrms_uncertainty
        bias = reduction.range_bias
        station_itrf_m = observation.station.itrf_xyz_at(observation.transmit_epoch)

        values = {
            "obs_time_utc": observation.transmit_epoch.isot(scale=TimeScale.UTC),
            "observed_rtt_s": record.observed_round_trip_time_s,
            "normal_point_index": int(record.index),
            "reflector_id": record.reflector_code,
            "station_id": record.station_code or record.station_name,
            "station_full_name": observation.station.name,
            "uncertainty_model": selected.kind.value,
            "uncertainty_raw": selected.uncertainty_raw,
            "uncertainty_two_way_s": selected.uncertainty_two_way_s,
            "uncertainty_two_way_ps": selected.uncertainty_two_way_ps,
            "range_uncertainty_one_way_m": selected.sigma_one_way_m,
            "fit_sigma_one_way_m": selected.sigma_one_way_m,
            "sigma_definition": reduction.sigma_definition,
            "uncertainty_source": selected.source,
            "uncertainty_group": selected.group,
            "wrms_two_way_m": None if wrms is None else wrms.wrms_two_way_m,
            "wrms_sigma_one_way_m": None if wrms is None else wrms.sigma_one_way_m,
            "wrms_uncertainty_raw": None if wrms is None else wrms.uncertainty_raw,
            "wrms_uncertainty_two_way_s": None if wrms is None else wrms.uncertainty_two_way_s,
            "wrms_uncertainty_two_way_ps": None if wrms is None else wrms.uncertainty_two_way_ps,
            "mini_uncertainty_raw": mini.uncertainty_raw,
            "mini_uncertainty_two_way_s": mini.uncertainty_two_way_s,
            "mini_uncertainty_two_way_ps": mini.uncertainty_two_way_ps,
            "mini_range_uncertainty_one_way_m": mini.sigma_one_way_m,
            "pressure_hpa": record.pressure_hpa,
            "temperature_c": record.temperature_c,
            "humidity_percent": record.humidity_percent,
            "wavelength_nm": record.wavelength_nm,
            "record_index": int(record.index),
            "computed_rtt_s": reduction.computed_rtt_s,
            "computed_rtt_raw_s": reduction.computed_rtt_raw_s,
            "computed_rtt_tdb_s": reduction.coordinate_round_trip_time_tdb_s,
            "range_bias_model": bias.model,
            "range_bias_two_way_cm": bias.two_way_cm,
            "range_bias_two_way_m": bias.two_way_m,
            "range_bias_two_way_s": bias.two_way_s,
            "range_bias_one_way_m": bias.one_way_m,
            "tt_minus_tdb_correction_s": reduction.tt_minus_tdb_interval_correction_s,
            "tt_minus_tdb_correction_one_way_m": 0.5 * C * reduction.tt_minus_tdb_interval_correction_s,
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
            "oc_rtt_s": reduction.observed_minus_computed_rtt_s,
            "oc_one_way_m": reduction.observed_minus_computed_one_way_m,
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
            "station_displacement_transmit_dx_m": float(solution.station_displacement_transmit_itrf_m[0]),
            "station_displacement_transmit_dy_m": float(solution.station_displacement_transmit_itrf_m[1]),
            "station_displacement_transmit_dz_m": float(solution.station_displacement_transmit_itrf_m[2]),
            "station_displacement_receive_dx_m": float(solution.station_displacement_receive_itrf_m[0]),
            "station_displacement_receive_dy_m": float(solution.station_displacement_receive_itrf_m[1]),
            "station_displacement_receive_dz_m": float(solution.station_displacement_receive_itrf_m[2]),
            "reflector_displacement_bounce_dx_m": float(solution.reflector_displacement_bounce_pa_m[0]),
            "reflector_displacement_bounce_dy_m": float(solution.reflector_displacement_bounce_pa_m[1]),
            "reflector_displacement_bounce_dz_m": float(solution.reflector_displacement_bounce_pa_m[2]),
            "iterations": solution.iterations,
            "converged": solution.converged,
            "station_name": observation.station.name,
            "station_catalog_key": observation.station_key,
            "station_itrf_x_m": float(station_itrf_m[0]),
            "station_itrf_y_m": float(station_itrf_m[1]),
            "station_itrf_z_m": float(station_itrf_m[2]),
            "reflector_name": observation.reflector.name,
            "reflector_catalog_key": observation.reflector_key,
            "valid_geometry": reduction.valid_geometry,
            "below_horizon": reduction.below_horizon,
            "status": reduction.status,
        }

        partials: dict[str, np.ndarray] = {
            "station_range_bias": np.array([1.0], dtype=float),
        }
        if prediction.reflector_position_partial_pa is not None:
            partial = prediction.reflector_position_partial_pa
            partials["reflector_position_pa"] = partial
            values.update(
                {
                    "design_reflector_dx": float(partial[0]),
                    "design_reflector_dy": float(partial[1]),
                    "design_reflector_dz": float(partial[2]),
                }
            )

        assert_output_schema(
            values,
            include_reflector_design=prediction.reflector_position_partial_pa is not None,
        )

        return LlrObservationResult(
            normal_point_index=int(record.index),
            station_key=observation.station_key,
            reflector_key=observation.reflector_key,
            epoch=observation.transmit_epoch,
            observed_minus_computed_m=reduction.observed_minus_computed_one_way_m,
            sigma_one_way_m=selected.sigma_one_way_m,
            converged=solution.converged,
            partials=partials,
            values=values,
        )


__all__ = ["LlrObservationResultBuilder"]
