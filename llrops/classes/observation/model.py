"""Pure LLR forward model built on the light-time solver."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llrops.base.constants import C2
from llrops.classes.relativistic.constants import MOON_EXTERNAL_POTENTIAL_BODIES
from llrops.classes.frames import ReferenceFrameSystem

from .light_time import LightTimeRequest, LightTimeSolution, LightTimeSolver, OpticalAtmosphere
from .resolver import ResolvedObservation


@dataclass(frozen=True, slots=True, eq=False)
class LlrPrediction:
    """Theoretical two-way observable and optional PA-frame design partial."""

    light_time: LightTimeSolution
    reflector_position_partial_pa: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.reflector_position_partial_pa is None:
            return
        partial = np.array(self.reflector_position_partial_pa, dtype=float, copy=True)
        if partial.size != 3 or not np.all(np.isfinite(partial)):
            raise ValueError("reflector_position_partial_pa must contain three finite values.")
        partial = partial.reshape(3)
        partial.setflags(write=False)
        object.__setattr__(self, "reflector_position_partial_pa", partial)


class LlrObservationModel:
    """Compute the theoretical LLR observable for one resolved normal point."""

    def __init__(
        self,
        frames: ReferenceFrameSystem,
        light_time_solver: LightTimeSolver,
    ) -> None:
        if not isinstance(frames, ReferenceFrameSystem):
            raise TypeError("frames must be a ReferenceFrameSystem.")
        if not isinstance(light_time_solver, LightTimeSolver):
            raise TypeError("light_time_solver must be a LightTimeSolver.")
        if light_time_solver.frames is not frames:
            raise ValueError("frames and light_time_solver.frames must be the same object.")
        self.frames = frames
        self.light_time_solver = light_time_solver

    @property
    def ephemeris(self):
        return self.frames.ephemeris

    def close(self) -> None:
        self.frames.close()

    def predict(
        self,
        observation: ResolvedObservation,
        *,
        include_reflector_position_partial: bool = False,
    ) -> LlrPrediction:
        record = observation.record
        station = observation.station
        reflector = observation.reflector
        transmit_epoch = observation.transmit_epoch
        station_itrf_m = station.itrf_xyz_at(transmit_epoch)

        request = LightTimeRequest(
            station_reference_itrf_m=station_itrf_m,
            station_position_at_utc=station.itrf_xyz_at,
            reflector_reference_pa_m=reflector.moon_fixed_xyz_m,
            transmit_epoch=transmit_epoch,
            observed_round_trip_time_s=record.observed_round_trip_time_s,
            initial_round_trip_time_s=record.observed_round_trip_time_s,
            atmosphere=OpticalAtmosphere(
                pressure_hpa=record.pressure_hpa,
                temperature_k=record.temperature_k,
                relative_humidity_percent=float(record.humidity_percent),
                latitude_rad=station.latitude_rad_at(transmit_epoch),
                height_m=station.height_m_at(transmit_epoch),
                wavelength_um=record.wavelength_um,
            ),
        )
        solution = self.light_time_solver.solve(request)
        partial = None
        if include_reflector_position_partial:
            partial = self.reflector_position_partial_pa(observation, solution)
        return LlrPrediction(solution, partial)

    def reflector_position_partial_pa(
        self,
        observation: ResolvedObservation,
        solution: LightTimeSolution,
    ) -> np.ndarray:
        """One-way O−C partial with respect to reflector PA coordinates."""
        station_bcrs_transmit = solution.station_bcrs_transmit_m
        station_bcrs_receive = solution.station_bcrs_receive_m
        reflector_bcrs_bounce = solution.reflector_bcrs_bounce_m

        uplink_vector = reflector_bcrs_bounce - station_bcrs_transmit
        downlink_vector = reflector_bcrs_bounce - station_bcrs_receive
        uplink_range = max(float(np.linalg.norm(uplink_vector)), 1.0e-30)
        downlink_range = max(float(np.linalg.norm(downlink_vector)), 1.0e-30)
        unit_sum = uplink_vector / uplink_range + downlink_vector / downlink_range

        pa2lcrs = self.ephemeris.pa2lcrs_matrix(solution.bounce_epoch)
        moon_velocity = self.ephemeris.body_state_bcrs(
            "MOON",
            solution.bounce_epoch,
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


__all__ = ["LlrObservationModel", "LlrPrediction"]
