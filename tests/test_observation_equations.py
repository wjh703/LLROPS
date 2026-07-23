import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.observation.equations import ObservationEquation

_UTC_EPOCH = Epoch(2458849.5, 0.0, TimeScale.UTC)


def test_observation_equation_normalizes_and_freezes_partials():
    equation = ObservationEquation(
        observed_minus_computed_m=0.25,
        sigma_m=0.01,
        partials={"geometry": [1.0, 2.0, 3.0]},
        identity=7,
        station_key="STA",
        reflector_key="REF",
        epoch=_UTC_EPOCH,
        metadata={"station_name": "Station"},
    )

    assert equation.observed_minus_computed_m == 0.25
    assert equation.epoch is _UTC_EPOCH
    assert np.allclose(equation.partials["geometry"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        equation.partials["geometry"][0] = 9.0
