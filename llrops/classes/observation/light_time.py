"""Two-way lunar laser ranging light-time solution using unified epochs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from llrops.base.constants import C
from llrops.base.epoch import Epoch, TimeScale
from llrops.base.array_validation import readonly_vector3, vector3
from llrops.classes.delays import (
    GravitationalDelay,
    TroposphereDelay,
    TroposphereInput,
    ZeroGravitationalDelay,
)
from llrops.classes.displacement import (
    ReflectorDisplacement,
    ReflectorDisplacementInput,
    StationDisplacement,
    StationDisplacementInput,
    ZeroReflectorDisplacement,
    ZeroStationDisplacement,
)
from llrops.classes.frames import ReferenceFrameSystem
from llrops.classes.displacement.terrestrial_geometry import local_up_unit_itrf


@dataclass(frozen=True, slots=True)
class OpticalAtmosphere:
    pressure_hpa: float
    temperature_k: float
    relative_humidity_percent: float
    latitude_rad: float
    height_m: float
    wavelength_um: float

    def __post_init__(self) -> None:
        for name in (
            "pressure_hpa",
            "temperature_k",
            "relative_humidity_percent",
            "latitude_rad",
            "height_m",
            "wavelength_um",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be positive.")
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= self.relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um must be positive.")

    def troposphere_input(self, elevation_rad: float) -> TroposphereInput:
        return TroposphereInput(
            elevation_rad=float(elevation_rad),
            pressure_hpa=self.pressure_hpa,
            temperature_k=self.temperature_k,
            relative_humidity_percent=self.relative_humidity_percent,
            latitude_rad=self.latitude_rad,
            height_m=self.height_m,
            wavelength_um=self.wavelength_um,
        )


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeRequest:
    station_reference_itrf_m: np.ndarray
    reflector_reference_pa_m: np.ndarray
    transmit_epoch: Epoch
    atmosphere: OpticalAtmosphere
    observed_round_trip_time_s: float | None = None
    initial_round_trip_time_s: float | None = None
    station_position_at_utc: Callable[[Epoch], Sequence[float]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "station_reference_itrf_m",
            readonly_vector3(self.station_reference_itrf_m, name="station_reference_itrf_m"),
        )
        object.__setattr__(
            self,
            "reflector_reference_pa_m",
            readonly_vector3(self.reflector_reference_pa_m, name="reflector_reference_pa_m"),
        )
        if not isinstance(self.transmit_epoch, Epoch):
            raise TypeError("transmit_epoch must be an Epoch.")
        self.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
        for name in ("observed_round_trip_time_s", "initial_round_trip_time_s"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite when supplied.")
            object.__setattr__(self, name, value)

    def station_position(self, epoch_utc: Epoch) -> np.ndarray:
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        if self.station_position_at_utc is None:
            return np.array(self.station_reference_itrf_m, copy=True)
        return readonly_vector3(
            self.station_position_at_utc(epoch_utc),
            name="station_position_at_utc result",
        )


@dataclass(frozen=True, slots=True)
class LightTimeLeg:
    geometric_range_m: float
    gravitational_delay_m: float
    tropospheric_delay_m: float
    elevation_rad: float
    troposphere_elevation_used_rad: float | None = None
    troposphere_elevation_clamped: bool = False

    @property
    def path_length_m(self) -> float:
        return self.geometric_range_m + self.gravitational_delay_m + self.tropospheric_delay_m

    @property
    def travel_time_s(self) -> float:
        return self.path_length_m / C


@dataclass(frozen=True, slots=True, eq=False)
class LightTimeSolution:
    """Converged TDB event epochs and light-path diagnostics.

    UTC copies are deliberately not stored.  Callers convert an event through
    the shared ``TimeScaleConverter`` only where UTC is actually required.
    """

    transmit_epoch: Epoch
    bounce_epoch: Epoch
    receive_epoch: Epoch
    observable_round_trip_time_s: float
    coordinate_round_trip_time_tdb_s: float
    tt_minus_tdb_interval_correction_s: float
    utc_rate_zeta: float
    uplink: LightTimeLeg
    downlink: LightTimeLeg
    station_displacement_transmit_itrf_m: np.ndarray
    station_displacement_receive_itrf_m: np.ndarray
    reflector_displacement_bounce_pa_m: np.ndarray
    station_bcrs_transmit_m: np.ndarray
    station_bcrs_receive_m: np.ndarray
    reflector_bcrs_bounce_m: np.ndarray
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        for name in ("transmit_epoch", "bounce_epoch", "receive_epoch"):
            epoch = getattr(self, name)
            if not isinstance(epoch, Epoch):
                raise TypeError(f"{name} must be an Epoch.")
            epoch.require_scale(TimeScale.TDB, name=name)
        for name in (
            "station_displacement_transmit_itrf_m",
            "station_displacement_receive_itrf_m",
            "reflector_displacement_bounce_pa_m",
            "station_bcrs_transmit_m",
            "station_bcrs_receive_m",
            "reflector_bcrs_bounce_m",
        ):
            object.__setattr__(
                self,
                name,
                readonly_vector3(getattr(self, name), name=name),
            )


@dataclass(frozen=True, slots=True)
class _IterationState:
    transmit_epoch: Epoch
    bounce_epoch: Epoch
    receive_epoch: Epoch
    uplink: LightTimeLeg
    downlink: LightTimeLeg
    iterations: int


@dataclass(frozen=True, slots=True)
class _StationEventState:
    epoch_utc: Epoch
    reference_itrf_m: np.ndarray
    displacement_itrf_m: np.ndarray
    position_itrf_m: np.ndarray
    position_gcrs_m: np.ndarray


class LightTimeSolver:
    def __init__(
        self,
        frames: ReferenceFrameSystem,
        *,
        gravitational_delay: GravitationalDelay | None = None,
        troposphere_delay: TroposphereDelay,
        station_displacement: StationDisplacement | None = None,
        reflector_displacement: ReflectorDisplacement | None = None,
        max_iterations: int = 12,
        tolerance_s: float = 1e-13,
    ) -> None:
        if not isinstance(frames, ReferenceFrameSystem):
            raise TypeError("frames must be a ReferenceFrameSystem.")
        if troposphere_delay is None:
            raise ValueError("troposphere_delay is required.")
        if int(max_iterations) <= 0:
            raise ValueError("max_iterations must be positive.")
        if float(tolerance_s) <= 0.0:
            raise ValueError("tolerance_s must be positive.")
        self.frames = frames
        self.time_converter = frames.time_converter
        self.gravitational_delay = gravitational_delay or ZeroGravitationalDelay()
        self.troposphere_delay = troposphere_delay
        self.station_displacement = station_displacement or ZeroStationDisplacement()
        self.reflector_displacement = reflector_displacement or ZeroReflectorDisplacement()
        self.max_iterations = int(max_iterations)
        self.tolerance_s = float(tolerance_s)

    def _station_displacement_itrf_m(
        self,
        station_itrf_m: Sequence[float],
        epoch_utc: Epoch,
    ) -> np.ndarray:
        return np.asarray(
            self.station_displacement.displacement_itrf_m(
                StationDisplacementInput(station_itrf_m=station_itrf_m, epoch_utc=epoch_utc)
            ),
            dtype=float,
        ).reshape(3)

    def _reflector_state_lcrs_m(
        self,
        reflector_pa_m: Sequence[float],
        epoch_tdb: Epoch,
    ) -> tuple[np.ndarray, np.ndarray]:
        reflector_lcrs = self.frames.pa2lcrs(reflector_pa_m, epoch_tdb)
        displacement_lcrs = np.asarray(
            self.reflector_displacement.displacement_lcrs_m(
                ReflectorDisplacementInput(
                    reflector_lcrs_m=reflector_lcrs,
                    epoch_tdb=epoch_tdb,
                )
            ),
            dtype=float,
        ).reshape(3)
        return reflector_lcrs + displacement_lcrs, displacement_lcrs

    def _station_state_at_utc(
        self,
        request: LightTimeRequest,
        epoch_utc: Epoch,
    ) -> _StationEventState:
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        reference = request.station_position(epoch_utc)
        displacement = self._station_displacement_itrf_m(reference, epoch_utc)
        position = reference + displacement
        gcrs = self.frames.itrf2gcrs(position, epoch_utc)
        return _StationEventState(
            epoch_utc=epoch_utc,
            reference_itrf_m=reference,
            displacement_itrf_m=displacement,
            position_itrf_m=position,
            position_gcrs_m=gcrs,
        )

    def _station_state_from_tdb(
        self,
        request: LightTimeRequest,
        epoch_tdb: Epoch,
    ) -> _StationEventState:
        """Resolve the station UTC epoch including the topocentric TDB-TT term."""
        epoch_tdb.require_scale(TimeScale.TDB, name="epoch_tdb")
        epoch_utc = self.time_converter.convert(epoch_tdb, TimeScale.UTC)
        state = self._station_state_at_utc(request, epoch_utc)
        for _ in range(3):
            updated_utc = self.time_converter.convert(
                epoch_tdb,
                TimeScale.UTC,
                station_gcrs_m=state.position_gcrs_m,
            )
            if abs(epoch_utc.seconds_until(updated_utc)) <= self.tolerance_s:
                if updated_utc == epoch_utc:
                    return state
                return self._station_state_at_utc(request, updated_utc)
            epoch_utc = updated_utc
            state = self._station_state_at_utc(request, epoch_utc)
        return state

    def _vacuum_elevation_rad(
        self,
        station_itrf_m: Sequence[float],
        target_bcrs_m: Sequence[float],
        station_epoch_utc: Epoch,
        target_epoch_tdb: Epoch,
    ) -> float:
        """Vacuum geometric elevation from explicit frame rotations only."""
        station_itrf = vector3(station_itrf_m, name="station_itrf_m")
        target_gcrs_m = self.frames.bcrs2gcrs(target_bcrs_m, target_epoch_tdb)
        target_itrf_m = self.frames.gcrs2itrf(target_gcrs_m, station_epoch_utc)
        los_itrf = target_itrf_m - station_itrf
        distance = float(np.linalg.norm(los_itrf))
        if distance <= 0.0:
            raise RuntimeError("Cannot compute elevation for a zero-length topocentric vector.")
        up = local_up_unit_itrf(station_itrf)
        sine_elevation = float(np.dot(los_itrf / distance, up))
        return float(np.arcsin(np.clip(sine_elevation, -1.0, 1.0)))

    def _troposphere_evaluation_elevation(self, elevation_rad: float) -> tuple[float, bool]:
        min_deg = getattr(self.troposphere_delay, "min_elevation_deg", None)
        if min_deg is None:
            return float(elevation_rad), False
        min_rad = float(np.deg2rad(float(min_deg)))
        if float(elevation_rad) < min_rad:
            return min_rad, True
        return float(elevation_rad), False

    @staticmethod
    def _pre_1972_utc_rate_zeta(epoch_utc: Epoch) -> float:
        epoch_utc.require_scale(TimeScale.UTC, name="epoch_utc")
        start = Epoch.from_isot("1968-02-01T00:00:00", scale=TimeScale.UTC)
        end = Epoch.from_isot("1972-01-01T00:00:00", scale=TimeScale.UTC)
        return 3.0e-8 if start <= epoch_utc < end else 0.0

    def solve(self, request: LightTimeRequest) -> LightTimeSolution:
        if not isinstance(request, LightTimeRequest):
            raise TypeError("request must be a LightTimeRequest.")

        transmit_utc = request.transmit_epoch
        transmit_station = self._station_state_at_utc(request, transmit_utc)
        transmit_tdb = self.time_converter.convert(
            transmit_utc,
            TimeScale.TDB,
            station_gcrs_m=transmit_station.position_gcrs_m,
        )

        station_bcrs_transmit = self.frames.gcrs2bcrs(
            transmit_station.position_gcrs_m,
            transmit_tdb,
        )

        initial_rtt_s = (
            request.initial_round_trip_time_s
            or request.observed_round_trip_time_s
            or 2.4
        )
        bounce_tdb = transmit_tdb.shifted(0.5 * initial_rtt_s)
        receive_tdb = transmit_tdb.shifted(initial_rtt_s)
        previous_rtt_s = float(initial_rtt_s)
        final_state: _IterationState | None = None
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            receive_station = self._station_state_from_tdb(request, receive_tdb)
            receive_utc = receive_station.epoch_utc
            reflector_lcrs_bounce, _ = self._reflector_state_lcrs_m(
                request.reflector_reference_pa_m,
                bounce_tdb,
            )

            station_bcrs_receive = self.frames.gcrs2bcrs(
                receive_station.position_gcrs_m,
                receive_tdb,
            )
            reflector_bcrs_bounce = self.frames.lcrs2bcrs(
                reflector_lcrs_bounce,
                bounce_tdb,
            )

            geometric_up_m = float(np.linalg.norm(reflector_bcrs_bounce - station_bcrs_transmit))
            geometric_down_m = float(np.linalg.norm(station_bcrs_receive - reflector_bcrs_bounce))
            gravitational_up_m = float(
                self.gravitational_delay.path_delay_m(
                    station_bcrs_transmit,
                    reflector_bcrs_bounce,
                    bounce_tdb,
                )
            )
            gravitational_down_m = float(
                self.gravitational_delay.path_delay_m(
                    reflector_bcrs_bounce,
                    station_bcrs_receive,
                    bounce_tdb,
                )
            )
            elevation_up_rad = self._vacuum_elevation_rad(
                transmit_station.position_itrf_m,
                reflector_bcrs_bounce,
                transmit_utc,
                bounce_tdb,
            )
            elevation_down_rad = self._vacuum_elevation_rad(
                receive_station.position_itrf_m,
                reflector_bcrs_bounce,
                receive_utc,
                bounce_tdb,
            )
            tropo_elevation_up_rad, tropo_up_clamped = self._troposphere_evaluation_elevation(
                elevation_up_rad
            )
            tropo_elevation_down_rad, tropo_down_clamped = self._troposphere_evaluation_elevation(
                elevation_down_rad
            )
            troposphere_up_m = float(
                self.troposphere_delay.slant_delay_m(
                    request.atmosphere.troposphere_input(tropo_elevation_up_rad)
                )
            )
            troposphere_down_m = float(
                self.troposphere_delay.slant_delay_m(
                    request.atmosphere.troposphere_input(tropo_elevation_down_rad)
                )
            )

            uplink = LightTimeLeg(
                geometric_range_m=geometric_up_m,
                gravitational_delay_m=gravitational_up_m,
                tropospheric_delay_m=troposphere_up_m,
                elevation_rad=elevation_up_rad,
                troposphere_elevation_used_rad=tropo_elevation_up_rad,
                troposphere_elevation_clamped=tropo_up_clamped,
            )
            downlink = LightTimeLeg(
                geometric_range_m=geometric_down_m,
                gravitational_delay_m=gravitational_down_m,
                tropospheric_delay_m=troposphere_down_m,
                elevation_rad=elevation_down_rad,
                troposphere_elevation_used_rad=tropo_elevation_down_rad,
                troposphere_elevation_clamped=tropo_down_clamped,
            )
            new_bounce_tdb = transmit_tdb.shifted(uplink.travel_time_s)
            new_receive_tdb = new_bounce_tdb.shifted(downlink.travel_time_s)
            new_rtt_s = transmit_tdb.seconds_until(new_receive_tdb)

            final_state = _IterationState(
                transmit_epoch=transmit_tdb,
                bounce_epoch=new_bounce_tdb,
                receive_epoch=new_receive_tdb,
                uplink=uplink,
                downlink=downlink,
                iterations=iteration,
            )
            bounce_tdb = new_bounce_tdb
            receive_tdb = new_receive_tdb
            if abs(new_rtt_s - previous_rtt_s) < self.tolerance_s:
                converged = True
                break
            previous_rtt_s = new_rtt_s

        if final_state is None:
            raise RuntimeError("Light-time solver failed before the first iteration.")

        receive_station = self._station_state_from_tdb(request, final_state.receive_epoch)
        reflector_lcrs_bounce, reflector_displacement_lcrs_bounce = (
            self._reflector_state_lcrs_m(
                request.reflector_reference_pa_m,
                final_state.bounce_epoch,
            )
        )
        reflector_displacement_pa_bounce = self.frames.lcrs2pa(
            reflector_displacement_lcrs_bounce,
            final_state.bounce_epoch,
        )
        station_bcrs_transmit_final = station_bcrs_transmit
        station_bcrs_receive_final = self.frames.gcrs2bcrs(
            receive_station.position_gcrs_m,
            final_state.receive_epoch,
        )
        reflector_bcrs_bounce_final = self.frames.lcrs2bcrs(
            reflector_lcrs_bounce,
            final_state.bounce_epoch,
        )

        transmit_tt = self.time_converter.tdb2tt(
            final_state.transmit_epoch,
            station_gcrs_m=transmit_station.position_gcrs_m,
        )
        receive_tt = self.time_converter.tdb2tt(
            final_state.receive_epoch,
            station_gcrs_m=receive_station.position_gcrs_m,
        )

        coordinate_rtt_s = final_state.transmit_epoch.seconds_until(final_state.receive_epoch)
        tt_rtt_s = transmit_tt.seconds_until(receive_tt)
        tt_minus_tdb_s = tt_rtt_s - coordinate_rtt_s
        zeta = self._pre_1972_utc_rate_zeta(transmit_utc)
        observable_rtt_s = tt_rtt_s / (1.0 + zeta)

        return LightTimeSolution(
            transmit_epoch=final_state.transmit_epoch,
            bounce_epoch=final_state.bounce_epoch,
            receive_epoch=final_state.receive_epoch,
            observable_round_trip_time_s=float(observable_rtt_s),
            coordinate_round_trip_time_tdb_s=float(coordinate_rtt_s),
            tt_minus_tdb_interval_correction_s=float(tt_minus_tdb_s),
            utc_rate_zeta=float(zeta),
            uplink=final_state.uplink,
            downlink=final_state.downlink,
            station_displacement_transmit_itrf_m=transmit_station.displacement_itrf_m,
            station_displacement_receive_itrf_m=receive_station.displacement_itrf_m,
            reflector_displacement_bounce_pa_m=reflector_displacement_pa_bounce,
            station_bcrs_transmit_m=station_bcrs_transmit_final,
            station_bcrs_receive_m=station_bcrs_receive_final,
            reflector_bcrs_bounce_m=reflector_bcrs_bounce_final,
            iterations=final_state.iterations,
            converged=converged,
        )


__all__ = [
    "LightTimeLeg",
    "LightTimeRequest",
    "LightTimeSolution",
    "LightTimeSolver",
    "OpticalAtmosphere",
]
