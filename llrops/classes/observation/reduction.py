"""Reduction of a theoretical LLR prediction to a weighted O-C observable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llrops.base.constants import C
from llrops.classes.ephemerides import Ephemeris

from llrops.classes.range_bias.models import (
    RangeBiasCorrection,
    RangeBiasModel,
)

from .model import LlrPrediction
from .resolver import ResolvedObservation


@dataclass(frozen=True, slots=True)
class ObservationReduction:
    """Bias-corrected residual and scalar diagnostics."""

    range_bias: RangeBiasCorrection
    computed_rtt_raw_s: float
    computed_rtt_s: float
    observed_minus_computed_raw_rtt_s: float
    observed_minus_computed_rtt_s: float
    observed_minus_computed_raw_one_way_m: float
    observed_minus_computed_one_way_m: float
    coordinate_round_trip_time_tdb_s: float
    tt_minus_tdb_interval_correction_s: float
    utc_rate_correction_s: float
    longitude_libration_model: str
    longitude_libration_correction_rad: float
    longitude_libration_correction_mas: float
    elevation_up_deg: float
    elevation_down_deg: float
    troposphere_elevation_up_used_deg: float | None
    troposphere_elevation_down_used_deg: float | None
    troposphere_up_clamped: bool
    troposphere_down_clamped: bool
    below_horizon: bool

    @property
    def troposphere_clamped(self) -> bool:
        return bool(self.troposphere_up_clamped or self.troposphere_down_clamped)

    @property
    def valid_geometry(self) -> bool:
        return not self.below_horizon

    @property
    def status(self) -> str:
        return "below_horizon" if self.below_horizon else "ok"

    @property
    def sigma_definition(self) -> str:
        return (
            "normal-point record: sigma_one_way_m = "
            "0.5 * c * uncertainty_two_way_s; no sigma floor"
        )


class LlrObservationReducer:
    """Apply deterministic corrections to a resolved observation."""

    def __init__(
        self,
        *,
        ephemeris: Ephemeris,
        range_bias: RangeBiasModel,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        self.ephemeris = ephemeris
        self.range_bias = range_bias

    def reduce(
        self,
        observation: ResolvedObservation,
        prediction: LlrPrediction,
        *,
        min_elevation_deg: float,
    ) -> ObservationReduction:
        record = observation.record
        solution = prediction.light_time

        elevation_up_deg = float(np.rad2deg(solution.uplink.elevation_rad))
        elevation_down_deg = float(np.rad2deg(solution.downlink.elevation_rad))
        below_horizon = (
            elevation_up_deg < min_elevation_deg
            or elevation_down_deg < min_elevation_deg
        )
        tropo_up_used = (
            None
            if solution.uplink.troposphere_elevation_used_rad is None
            else float(np.rad2deg(solution.uplink.troposphere_elevation_used_rad))
        )
        tropo_down_used = (
            None
            if solution.downlink.troposphere_elevation_used_rad is None
            else float(np.rad2deg(solution.downlink.troposphere_elevation_used_rad))
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

        libration_rad = float(
            self.ephemeris.longitude_libration_correction_rad(
                solution.bounce_epoch
            )
        )
        libration_mas = float(np.rad2deg(libration_rad) * 3_600_000.0)
        coordinate_rtt_s = float(solution.coordinate_round_trip_time_tdb_s)
        tt_minus_tdb_s = float(solution.tt_minus_tdb_interval_correction_s)
        utc_rate_correction_s = computed_raw_s - (coordinate_rtt_s + tt_minus_tdb_s)

        return ObservationReduction(
            range_bias=bias,
            computed_rtt_raw_s=computed_raw_s,
            computed_rtt_s=computed_s,
            observed_minus_computed_raw_rtt_s=oc_raw_s,
            observed_minus_computed_rtt_s=oc_s,
            observed_minus_computed_raw_one_way_m=oc_raw_m,
            observed_minus_computed_one_way_m=oc_m,
            coordinate_round_trip_time_tdb_s=coordinate_rtt_s,
            tt_minus_tdb_interval_correction_s=tt_minus_tdb_s,
            utc_rate_correction_s=utc_rate_correction_s,
            longitude_libration_model=str(self.ephemeris.longitude_libration_model),
            longitude_libration_correction_rad=libration_rad,
            longitude_libration_correction_mas=libration_mas,
            elevation_up_deg=elevation_up_deg,
            elevation_down_deg=elevation_down_deg,
            troposphere_elevation_up_used_deg=tropo_up_used,
            troposphere_elevation_down_used_deg=tropo_down_used,
            troposphere_up_clamped=bool(solution.uplink.troposphere_elevation_clamped),
            troposphere_down_clamped=bool(solution.downlink.troposphere_elevation_clamped),
            below_horizon=below_horizon,
        )


__all__ = ["LlrObservationReducer", "ObservationReduction"]
