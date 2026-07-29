import pickle

import numpy as np

from lunarops.base.epoch import Epoch, TimeScale
from lunarops.classes.observation.equations import (
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
        wavelength_nm=532.0,
    )


def test_output_level_accepts_enum_and_string():
    assert (
        ObservationOutputLevel.parse(ObservationOutputLevel.FULL)
        is ObservationOutputLevel.FULL
    )
    assert (
        ObservationOutputLevel.parse("standard")
        is ObservationOutputLevel.STANDARD
    )


def test_equation_is_pickle_safe_for_mpi_transport():
    restored = pickle.loads(pickle.dumps(_equation()))
    assert restored.identity == 3
    assert restored.epoch == _UTC_EPOCH
    assert restored.wavelength_nm == 532.0
    assert np.allclose(
        restored.partials["reflector_position_pa"], [0.1, 0.2, 0.3]
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
