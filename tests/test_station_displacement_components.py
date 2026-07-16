from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from llrops.classes.builders import ensure_registered
from llrops.classes.displacement import (
    CompositeStationDisplacement,
    Iers2010OceanPoleTide,
    Iers2010PoleTide,
    LunarSolidTide,
    OceanPoleTideGrid,
    ReflectorDisplacementInput,
    StationDisplacementInput,
    ZeroReflectorDisplacement,
    ZeroStationDisplacement,
)
from llrops.config.context import RunContext
from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.ephemerides import BodyState, Ephemeris
from llrops.classes.frames import EarthOrientation, PolarMotion


class _ConstantDisplacement:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)

    def displacement_itrf_m(self, data: StationDisplacementInput):
        assert isinstance(data, StationDisplacementInput)
        return self.xyz


def _station_input() -> StationDisplacementInput:
    return StationDisplacementInput(
        station_itrf_m=[6_378_137.0, 0.0, 0.0],
        epoch_utc=Epoch.from_isot("2000-01-01T00:00:00", scale=TimeScale.UTC),
    )


def test_displacement_inputs_are_frozen_slotted_and_read_only():
    station = _station_input()
    reflector = ReflectorDisplacementInput(
        reflector_lcrs_m=[1_737_400.0, 0.0, 0.0],
        epoch_tdb=Epoch.from_jd(2451544.5, scale=TimeScale.TDB),
    )

    assert not hasattr(station, "__dict__")
    assert not hasattr(reflector, "__dict__")
    assert not station.station_itrf_m.flags.writeable
    assert not reflector.reflector_lcrs_m.flags.writeable
    with pytest.raises(FrozenInstanceError):
        station.epoch_utc = station.epoch_utc
    with pytest.raises(ValueError):
        station.station_itrf_m[0] = 0.0
    with pytest.raises(ValueError):
        reflector.reflector_lcrs_m[0] = 0.0


def test_zero_displacement_models_return_three_component_vectors():
    assert np.allclose(
        ZeroStationDisplacement().displacement_itrf_m(_station_input()),
        np.zeros(3),
    )
    reflector_data = ReflectorDisplacementInput(
        reflector_lcrs_m=[1.0, 0.0, 0.0],
        epoch_tdb=Epoch.from_jd(2451544.5, scale=TimeScale.TDB),
    )
    assert np.allclose(
        ZeroReflectorDisplacement().displacement_lcrs_m(reflector_data),
        np.zeros(3),
    )


def test_composite_station_displacement_sums_components():
    model = CompositeStationDisplacement(
        components=(
            _ConstantDisplacement([1.0, 2.0, 3.0]),
            ZeroStationDisplacement(),
            _ConstantDisplacement([-0.5, 0.5, 1.0]),
        )
    )
    assert np.allclose(
        model.displacement_itrf_m(_station_input()),
        [0.5, 2.5, 4.0],
    )


def test_composite_station_displacement_rejects_invalid_components():
    with pytest.raises(TypeError, match="cannot contain None"):
        CompositeStationDisplacement(components=(None,))


def test_registered_station_sum_and_context_cache():
    ensure_registered()
    context = RunContext()
    model = context.create_class(
        "stationDisplacement",
        {"type": "sum", "components": ["none", {"type": "none"}]},
        cache=True,
    )
    assert np.allclose(model.displacement_itrf_m(_station_input()), 0.0)

    first = context.create_class("stationDisplacement", "none", cache=True)
    second = context.create_class("stationDisplacement", "none", cache=True)
    assert first is second


class _FakeEarthOrientation(EarthOrientation):
    @property
    def source_file(self):
        return None

    def polar_motion(self, epoch_utc):
        return PolarMotion(0.1, 0.2)

    def ut1_minus_utc_sec(self, epoch_utc):
        return 0.0


class _FakeEphemeris(Ephemeris):
    @property
    def source_file(self):
        from pathlib import Path
        return Path("fake.eph")

    def body_state_bcrs(self, body, epoch: Epoch):
        positions = {
            "MOON": np.zeros(3),
            "EARTH": np.array([384_400_000.0, 0.0, 0.0]),
            "SUN": np.array([149_597_870_700.0, 0.0, 0.0]),
            "MERCURY BARYCENTER": np.array([5.0e10, 1.0e10, 0.0]),
            "VENUS BARYCENTER": np.array([1.0e11, 2.0e10, 0.0]),
            "MARS BARYCENTER": np.array([2.0e11, 3.0e10, 0.0]),
            "JUPITER BARYCENTER": np.array([7.0e11, 4.0e10, 0.0]),
            "SATURN BARYCENTER": np.array([1.4e12, 5.0e10, 0.0]),
            "URANUS BARYCENTER": np.array([2.8e12, 6.0e10, 0.0]),
            "NEPTUNE BARYCENTER": np.array([4.5e12, 7.0e10, 0.0]),
        }
        position = positions[body]
        return BodyState(position, np.zeros(3))

    def pa2lcrs_matrix(self, epoch: Epoch):
        return np.eye(3)


def test_pole_tide_exposes_typed_evaluation_result():
    model = Iers2010PoleTide(earth_orientation=_FakeEarthOrientation())
    result = model.evaluate(_station_input())
    assert result.displacement_itrf_m.shape == (3,)
    assert result.displacement_enu_m.shape == (3,)
    assert np.all(np.isfinite(result.displacement_itrf_m))
    assert np.allclose(
        model.displacement_itrf_m(_station_input()),
        result.displacement_itrf_m,
    )


def test_ocean_pole_tide_grid_and_model(tmp_path):
    coefficient_file = tmp_path / "ocean_pole_tide.txt"
    coefficient_file.write_text(
        "0 -90 1 0 2 0 3 0\n"
        "180 -90 1 0 2 0 3 0\n"
        "0 90 1 0 2 0 3 0\n"
        "180 90 1 0 2 0 3 0\n"
    )
    grid = OceanPoleTideGrid(coefficient_file)
    model = Iers2010OceanPoleTide(grid=grid, earth_orientation=_FakeEarthOrientation())
    result = model.evaluate(_station_input())
    assert grid.info.latitude_nodes == 2
    assert grid.info.longitude_nodes == 2
    assert np.all(np.isfinite(result.displacement_itrf_m))


def test_lunar_solid_tide_requires_no_runtime_backend_injection():
    model = LunarSolidTide(ephemeris=_FakeEphemeris())
    data = ReflectorDisplacementInput(
        reflector_lcrs_m=[1_737_400.0, 0.0, 0.0],
        epoch_tdb=Epoch.from_jd(2451544.5, scale=TimeScale.TDB),
    )
    displacement = model.displacement_lcrs_m(data)
    assert displacement.shape == (3,)
    assert np.all(np.isfinite(displacement))
