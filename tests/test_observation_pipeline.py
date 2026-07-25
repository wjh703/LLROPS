from pathlib import Path

import numpy as np
import pytest

from llrops.base.constants import C
from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.delays import ZeroTroposphereDelay
from llrops.classes.ephemerides import BodyState, Ephemeris
from llrops.classes.frames import EarthOrientation, PolarMotion, ReferenceFrameSystem
from llrops.classes.observation import (
    LlrObservationModel,
    LlrObservationProcessor,
    LlrObservationReducer,
    LightTimeSolver,
    ObservationModelState,
    ObservationProcessingOptions,
    ObservationResolver,
)
from llrops.classes.parametrization.reflector_position import (
    ReflectorPositionParametrization,
)
from llrops.classes.range_bias.models import ZeroRangeBiasModel
from llrops.fileio.catalogs import ReflectorRecord, StationRecord
from llrops.fileio.normal_points import NptDataset, NptRecord


class _Ephemeris(Ephemeris):
    _POSITIONS = {
        "SSB": np.zeros(3),
        "EARTH": np.zeros(3),
        "MOON": np.array([384_400_000.0, 0.0, 0.0]),
        "SUN": np.array([149_597_870_700.0, 0.0, 0.0]),
        "MERCURY BARYCENTER": np.array([5.0e10, 2.0e10, 0.0]),
        "VENUS BARYCENTER": np.array([1.0e11, 3.0e10, 0.0]),
        "MARS BARYCENTER": np.array([2.0e11, 4.0e10, 0.0]),
        "JUPITER BARYCENTER": np.array([7.0e11, 5.0e10, 0.0]),
        "SATURN BARYCENTER": np.array([1.4e12, 6.0e10, 0.0]),
        "URANUS BARYCENTER": np.array([2.8e12, 7.0e10, 0.0]),
        "NEPTUNE BARYCENTER": np.array([4.5e12, 8.0e10, 0.0]),
    }

    @property
    def source_file(self) -> Path:
        return Path("test.eph")

    def body_state_bcrs(self, body: str, epoch: Epoch) -> BodyState:
        epoch.require_scale(TimeScale.TDB)
        return BodyState(self._POSITIONS[body.upper()], np.zeros(3))

    def pa2lcrs_matrix(self, epoch: Epoch) -> np.ndarray:
        epoch.require_scale(TimeScale.TDB)
        return np.eye(3)

    def tdb_minus_tt_sec(self, epoch: Epoch) -> float:
        epoch.require_scale(TimeScale.TDB)
        return 0.0


class _EarthOrientation(EarthOrientation):
    @property
    def source_file(self) -> Path:
        return Path("test.eop")

    def polar_motion(self, epoch_utc: Epoch) -> PolarMotion:
        return PolarMotion(0.0, 0.0)

    def ut1_minus_utc_sec(self, epoch_utc: Epoch) -> float:
        return 0.0


def _record(index: int = 4) -> NptRecord:
    return NptRecord(
        station_name="APOL",
        reflector_name="Apollo 15",
        transmit_epoch=Epoch.from_isot(
            "2020-01-01T00:00:00",
            scale=TimeScale.UTC,
        ),
        round_trip_time_s=2.55,
        uncertainty_two_way_s=100.0e-12,
        pressure_hpa=900.0,
        temperature_k=285.0,
        humidity_percent=25.0,
        wavelength_nm=532.0,
        index=index,
        station_code="70610",
        reflector_code="A15",
    )


def _pipeline():
    frames = ReferenceFrameSystem(_Ephemeris(), _EarthOrientation())
    solver = LightTimeSolver(
        frames,
        troposphere_delay=ZeroTroposphereDelay(),
    )
    state = ObservationModelState.from_catalogs(
        {
            "APOLLO": StationRecord(
                name="Apache Point",
                aliases=("APOL", "70610"),
                itrf_xyz_m=(6_378_137.0, 0.0, 0.0),
            )
        },
        {
            "APOLLO15": ReflectorRecord(
                name="Apollo 15",
                aliases=("A15",),
                moon_fixed_xyz_m=(1_737_400.0, 0.0, 0.0),
            )
        },
    )
    resolver = ObservationResolver(state)
    model = LlrObservationModel(frames, solver)
    reducer = LlrObservationReducer(
        ephemeris=frames.ephemeris,
        range_bias=ZeroRangeBiasModel(),
    )
    return LlrObservationProcessor(
        resolver=resolver,
        model=model,
        reducer=reducer,
    )


def test_full_observation_pipeline_builds_one_equation():
    processor = _pipeline()
    equation = processor.process(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(
            min_elevation_deg=-90.0,
            include_reflector_position_partial=True,
        ),
    )[0]

    assert equation.identity == 4
    assert equation.station_key == "APOLLO"
    assert equation.reflector_key == "APOLLO15"
    assert equation.converged
    assert equation.sigma_m == pytest.approx(0.5 * C * 100.0e-12)
    assert equation.partials["station_range_bias"] == pytest.approx([1.0])
    assert equation.partials["reflector_position_pa"].shape == (3,)
    assert equation.metadata["status"] == "ok"
    assert equation.to_row("standard")["oc_one_way_m"] == pytest.approx(
        equation.observed_minus_computed_m
    )


def test_reducer_marks_geometry_below_requested_elevation():
    processor = _pipeline()
    observation = processor.resolver.resolve(_record())
    prediction = processor.model.predict(observation)
    reduction = processor.reducer.reduce(
        observation,
        prediction,
        min_elevation_deg=91.0,
    )

    assert reduction.below_horizon
    assert not reduction.valid_geometry
    assert reduction.status == "below_horizon"
    assert reduction.computed_rtt_s == pytest.approx(
        prediction.light_time.observable_round_trip_time_s
    )


def test_resolver_reports_all_unresolved_records():
    processor = _pipeline()
    missing_station = _record(1)
    missing_station.station_name = "UNKNOWN"
    missing_station.station_code = None
    missing_reflector = _record(2)
    missing_reflector.reflector_name = "UNKNOWN"
    missing_reflector.reflector_code = None

    with pytest.raises(ValueError, match="2 record") as error:
        processor.resolver.validate([missing_station, missing_reflector])
    assert "record_index=0" in str(error.value)
    assert "record_index=1" in str(error.value)


def test_reflector_parametrization_updates_explicit_model_state():
    processor = _pipeline()
    equation = processor.process(
        NptDataset([_record()]),
        options=ObservationProcessingOptions(
            include_reflector_position_partial=True,
        ),
    )[0]
    block = ReflectorPositionParametrization(reflectors=["APOLLO15"])
    original = processor.model_state.reflector_catalog["APOLLO15"]

    block.setup([equation], processor.model_state)
    block.apply_update(np.array([1.0, 2.0, 3.0]))

    updated = processor.model_state.reflector_catalog["APOLLO15"]
    assert updated is not original
    assert updated.moon_fixed_xyz_m == pytest.approx([1_737_401.0, 2.0, 3.0])
    assert processor.resolver.resolve(_record()).reflector is updated
