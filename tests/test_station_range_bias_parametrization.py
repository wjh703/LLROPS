import numpy as np

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.station_range_bias import StationRangeBiasParametrization


def _eq(station_key, epoch, station_id=None):
    return ObservationEquation(
        observed_minus_computed_m=0.0,
        sigma_m=1.0,
        partials={"station_range_bias": np.array([1.0])},
        identity=1,
        station_key=station_key,
        reflector_key="apollo15",
        epoch=epoch,
        metadata={
            "station_catalog_key": station_key,
            "station_id": station_id,
            "station_name": station_key,
            "station_full_name": station_key,
        },
    )


def test_station_mode_one_parameter_per_station():
    eqs = [_eq("APOLLO", Epoch(2454466.5, 0.0, TimeScale.UTC)), _eq("GRASSE", Epoch(2454832.5, 0.0, TimeScale.UTC))]
    block = StationRangeBiasParametrization(per="station")
    block.setup(eqs, None)

    assert block.keys == ["APOLLO", "GRASSE"]
    assert [str(name) for name in block.parameter_names()] == [
        "APOLLO:rangeBias::",
        "GRASSE:rangeBias::",
    ]
    assert np.allclose(block.design_columns(eqs[0]), [1.0, 0.0])


def test_station_interval_mode_keeps_overlapping_explicit_intervals():
    # APOLLO on 2008-01-01 is inside both the broad 2006-2010 interval and
    # the shorter 2007-12-15 to 2008-06-30 interval.
    eq = _eq("APOLLO", Epoch(2454466.5, 0.0, TimeScale.UTC), station_id="7045")
    block = StationRangeBiasParametrization(
        per="station+interval",
        intervals={
            "APOLLO": [
                "2006-04-07/2010-11-01",
                "2007-12-15/2008-06-30",
            ]
        },
    )
    block.setup([eq], None)

    assert block.keys == [
        "APOLLO_2006-04-07_2010-11-01",
        "APOLLO_2007-12-15_2008-06-30",
    ]
    assert np.allclose(block.design_columns(eq), [1.0, 1.0])

    block.apply_update(np.array([0.2, -0.05]))
    assert block.reduce_observation(eq) == 0.15000000000000002
    assert [str(name) for name in block.parameter_names()] == [
        "APOLLO:rangeBias:interval:2006-04-07/2010-11-01",
        "APOLLO:rangeBias:interval:2007-12-15/2008-06-30",
    ]


def test_station_mode_requested_alias_matches_canonical_observation():
    eq = _eq("APOL", Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC), station_id="70610")
    block = StationRangeBiasParametrization(stations=["APOLLO"], per="station")
    block.setup([eq], None)

    assert block.keys == ["APOLLO"]
    assert [str(name) for name in block.parameter_names()] == ["APOLLO:rangeBias::"]
    assert np.allclose(block.design_columns(eq), [1.0])
