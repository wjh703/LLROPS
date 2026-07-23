"""Reduction of a theoretical LLR prediction to a weighted O-C observable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from llrops.base.constants import C
from llrops.classes.ephemerides import Ephemeris

from llrops.classes.range_bias.models import (
    RangeBiasCorrection,
    RangeBiasModel,
)
from llrops.classes.uncertainty.models import (
    MiniUncertainty,
    UncertaintyEstimate,
    UncertaintyKind,
    UncertaintyModel,
    WrmsTableUncertainty,
)
from llrops.classes.uncertainty.wrms_table import DEFAULT_WRMS_UNCERTAINTY_TABLE

from .model import LlrPrediction
from .resolver import ResolvedObservation


@dataclass(frozen=True, slots=True)
class ObservationReduction:
    """Bias-corrected residual, stochastic weight, and scalar diagnostics."""

    range_bias: RangeBiasCorrection
    selected_uncertainty: UncertaintyEstimate
    mini_uncertainty: UncertaintyEstimate
    wrms_uncertainty: UncertaintyEstimate | None
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
        if self.selected_uncertainty.kind is UncertaintyKind.WRMS_TABLE:
            return "WRMS table: sigma_one_way_m = 0.5 * wrms_two_way_m; no sigma floor"
        return (
            "MINI uncertainty: sigma_one_way_m = "
            "0.5 * c * mini_uncertainty_two_way_s; no sigma floor"
        )


class LlrObservationReducer:
    """Apply deterministic corrections and select an uncertainty model."""

    def __init__(
        self,
        *,
        ephemeris: Ephemeris,
        range_bias: RangeBiasModel,
        uncertainty_models: Mapping[UncertaintyKind, UncertaintyModel] | None = None,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        self.ephemeris = ephemeris
        self.range_bias = range_bias
        self.uncertainty_models = dict(
            uncertainty_models
            or {
                UncertaintyKind.WRMS_TABLE: WrmsTableUncertainty(DEFAULT_WRMS_UNCERTAINTY_TABLE),
                UncertaintyKind.MINI: MiniUncertainty(),
            }
        )
        missing = set(UncertaintyKind) - set(self.uncertainty_models)
        if missing:
            names = sorted(item.value for item in missing)
            raise ValueError(f"Missing uncertainty model(s): {names}")

    def validate_uncertainty(
        self,
        observations: Iterable[ResolvedObservation],
        kind: UncertaintyKind,
        *,
        source_name: str,
    ) -> None:
        model = self.uncertainty_models[kind]
        problems: list[str] = []
        for observation in observations:
            try:
                model.validate(
                    record=observation.record,
                    station_candidates=observation.station_candidates,
                    epoch_utc=observation.transmit_epoch,
                )
            except ValueError as exc:
                problems.append(
                    f"normal_point_index={observation.record.index}: {exc}"
                )
        if problems:
            detail = "\n  ".join(problems)
            raise ValueError(
                f"{source_name}: uncertainty validation failed for "
                f"{len(problems)} record(s):\n  {detail}"
            )

    def reduce(
        self,
        observation: ResolvedObservation,
        prediction: LlrPrediction,
        *,
        uncertainty: UncertaintyKind,
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
        selected = self.uncertainty_models[uncertainty].estimate(
            record=record,
            station_candidates=observation.station_candidates,
            epoch_utc=observation.transmit_epoch,
        )
        mini = self.uncertainty_models[UncertaintyKind.MINI].estimate(
            record=record,
            station_candidates=observation.station_candidates,
            epoch_utc=observation.transmit_epoch,
        )
        try:
            wrms = self.uncertainty_models[UncertaintyKind.WRMS_TABLE].estimate(
                record=record,
                station_candidates=observation.station_candidates,
                epoch_utc=observation.transmit_epoch,
            )
        except ValueError:
            wrms = None

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
            selected_uncertainty=selected,
            mini_uncertainty=mini,
            wrms_uncertainty=wrms,
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
