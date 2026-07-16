import json

import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import Parametrization
from llrops.classes.parametrization.station_range_bias import StationRangeBiasParametrization


def _eq(station="APOLLO"):
    return ObservationEquation(
        observed_minus_computed_m=0.0,
        sigma_m=1.0,
        partials={"station_range_bias": np.array([1.0])},
        identity=station,
        station_key=station,
        reflector_key="REF",
        epoch=Epoch.from_isot("2008-01-01T00:00:00", scale=TimeScale.UTC),
        metadata={"station_name": station},
    )


def assert_parametrization_contract(block: Parametrization, equations, context=None):
    block.setup(equations, context)
    names1 = block.parameter_names()
    names2 = block.parameter_names()
    assert names1 == names2
    for eq in equations:
        assert len(block.design_columns(eq)) == len(names1)
    json.dumps(block.state(), default=str)
    with pytest.raises(ValueError):
        block.apply_update(np.zeros(len(names1) + 1))


def test_station_range_bias_parametrization_contract():
    assert_parametrization_contract(StationRangeBiasParametrization(per="station"), [_eq("APOLLO")])
