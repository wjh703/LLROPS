import pickle

import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.observation.equations import (
    ObservationEquation,
    ObservationOutputLevel,
    STANDARD_OUTPUT_FIELDS,
)

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def _equation() -> ObservationEquation:
    return ObservationEquation(
        observed_minus_computed_m=0.12,
        sigma_m=0.02,
        partials={
            "station_range_bias": [1.0],
            "reflector_position_pa": [0.1, 0.2, 0.3],
        },
        identity=3,
        station_key="STA",
        reflector_key="REF",
        epoch=_UTC_EPOCH,
        converged=True,
        metadata={
            "station_id": "001",
            "station_name": "Station",
            "reflector_id": "R1",
            "reflector_name": "Reflector",
            "observed_rtt_s": 2.5,
            "computed_rtt_s": 2.4,
            "elevation_up_deg": 30.0,
            "status": "ok",
            "private_diagnostic": 42,
        },
    )


def test_equation_projects_standard_and_full_rows_without_duplicate_state():
    equation = _equation()

    standard = equation.to_row(ObservationOutputLevel.STANDARD)
    full = equation.to_row(ObservationOutputLevel.FULL)

    assert "private_diagnostic" not in standard
    assert full["private_diagnostic"] == 42
    assert standard["normal_point_index"] == equation.identity
    assert standard["oc_one_way_m"] == equation.observed_minus_computed_m
    assert standard["fit_sigma_one_way_m"] == equation.sigma_m
    assert standard["design_reflector_dx"] == 0.1
    assert full["oc_rtt_s"] == pytest.approx(
        2.0 * equation.observed_minus_computed_m / 299_792_458.0
    )
    assert full["range_uncertainty_one_way_m"] == equation.sigma_m
    assert full["station_catalog_key"] == "STA"
    assert full["reflector_catalog_key"] == "REF"


def test_output_level_accepts_enum_and_string():
    assert ObservationOutputLevel.parse(ObservationOutputLevel.FULL) is ObservationOutputLevel.FULL
    assert ObservationOutputLevel.parse("standard") is ObservationOutputLevel.STANDARD


def test_equation_is_pickle_safe_for_mpi_transport():
    restored = pickle.loads(pickle.dumps(_equation()))
    assert restored.identity == 3
    assert restored.epoch == _UTC_EPOCH
    assert restored.metadata["private_diagnostic"] == 42
    assert np.allclose(restored.partials["reflector_position_pa"], [0.1, 0.2, 0.3])


def test_equation_rejects_metadata_that_duplicates_typed_state():
    with pytest.raises(ValueError, match="oc_one_way_m"):
        ObservationEquation(
            observed_minus_computed_m=0.12,
            sigma_m=0.02,
            partials={},
            identity=3,
            station_key="STA",
            reflector_key="REF",
            epoch=_UTC_EPOCH,
            metadata={"oc_one_way_m": 99.0},
        )


def test_standard_output_schema_contains_only_per_record_oc_fields():
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
