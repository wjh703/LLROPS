import pickle

import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.observation.results import LlrObservationResult, ObservationOutputLevel

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def test_observation_equation_normalizes_and_freezes_partials():
    eq = ObservationEquation(
        observed_minus_computed_m=0.25,
        sigma_m=0.01,
        partials={"geometry": [1.0, 2.0, 3.0]},
        identity=7,
        station_key="STA",
        reflector_key="REF",
        epoch=_UTC_EPOCH,
        metadata={"station_name": "Station"},
    )

    assert eq.observed_minus_computed_m == 0.25
    assert eq.epoch is _UTC_EPOCH
    assert np.allclose(eq.partials["geometry"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        eq.partials["geometry"][0] = 9.0


def test_typed_result_projects_rows_and_builds_equation():
    result = LlrObservationResult(
        normal_point_index=3,
        station_key="STA",
        reflector_key="REF",
        epoch=_UTC_EPOCH,
        observed_minus_computed_m=0.12,
        sigma_one_way_m=0.02,
        converged=True,
        partials={
            "station_range_bias": [1.0],
            "reflector_position_pa": [0.1, 0.2, 0.3],
        },
        values={
            "obs_time_utc": "2020-01-01T00:00:00",
            "normal_point_index": 3,
            "station_catalog_key": "STA",
            "reflector_catalog_key": "REF",
            "oc_one_way_m": 0.12,
            "fit_sigma_one_way_m": 0.02,
            "design_reflector_dx": 0.1,
            "design_reflector_dy": 0.2,
            "design_reflector_dz": 0.3,
            "private_diagnostic": 42,
        },
    )

    standard = result.to_row(ObservationOutputLevel.STANDARD)
    full = result.to_row(ObservationOutputLevel.FULL)
    equation = result.to_equation()

    assert "private_diagnostic" not in standard
    assert full["private_diagnostic"] == 42
    assert standard["design_reflector_dx"] == 0.1
    assert equation.identity == 3
    assert equation.epoch == _UTC_EPOCH
    assert equation.metadata["station_catalog_key"] == "STA"
    assert np.allclose(equation.partials["reflector_position_pa"], [0.1, 0.2, 0.3])
    with pytest.raises(TypeError):
        result.partials["new"] = np.array([1.0])
    with pytest.raises(TypeError):
        result.values["new"] = 1


def test_output_level_accepts_enum_and_string():
    assert ObservationOutputLevel.parse(ObservationOutputLevel.FULL) is ObservationOutputLevel.FULL
    assert ObservationOutputLevel.parse("standard") is ObservationOutputLevel.STANDARD
    with pytest.raises(ValueError):
        ObservationOutputLevel.parse("verbose")


def test_result_is_pickle_safe_for_mpi_transport():
    result = LlrObservationResult(
        normal_point_index=1,
        station_key="STA",
        reflector_key="REF",
        epoch=_UTC_EPOCH,
        observed_minus_computed_m=0.0,
        sigma_one_way_m=0.01,
        converged=True,
        partials={"station_range_bias": [1.0]},
        values={"normal_point_index": 1},
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.normal_point_index == 1
    assert restored.epoch == _UTC_EPOCH
    assert restored.values["normal_point_index"] == 1
    assert np.allclose(restored.partials["station_range_bias"], [1.0])


def test_standard_output_schema_contains_only_per_record_oc_fields():
    from llrops.classes.observation.results import STANDARD_OUTPUT_FIELDS

    assert STANDARD_OUTPUT_FIELDS == (
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
