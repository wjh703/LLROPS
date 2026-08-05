from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

from lunarops.classes.delays import TroposphereInput, ZeroTroposphereDelay
from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere


def _input(**changes):
    values = {
        "elevation_rad": np.deg2rad(30.0),
        "pressure_hpa": 1013.25,
        "temperature_k": 293.15,
        "relative_humidity_percent": 50.0,
        "latitude_rad": np.deg2rad(45.0),
        "height_m": 100.0,
        "wavelength_um": 0.532,
    }
    values.update(changes)
    return TroposphereInput(**values)


def test_troposphere_input_is_frozen_and_slotted():
    data = _input()

    assert not hasattr(data, "__dict__")
    with pytest.raises(FrozenInstanceError):
        cast(Any, data).pressure_hpa = 900.0


def test_troposphere_models_consume_input_object():
    data = _input()

    assert ZeroTroposphereDelay().slant_delay_m(data) == 0.0
    model = Iers2010MendesPavlisTroposphere()
    assert model.elevation_floor_rad == pytest.approx(np.deg2rad(3.0))
    assert ZeroTroposphereDelay().elevation_floor_rad is None
    delay = model.slant_delay_m(data)
    assert delay == pytest.approx(4.882704085022898, rel=0.0, abs=1.0e-14)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"elevation_rad": np.inf}, "elevation_rad must be finite"),
        ({"elevation_rad": np.deg2rad(91.0)}, "elevation_rad must be in"),
        ({"pressure_hpa": 0.0}, "pressure_hpa must be positive"),
        ({"temperature_k": 0.0}, "temperature_k must be positive"),
        ({"relative_humidity_percent": 101.0}, "relative_humidity_percent must be in"),
        ({"latitude_rad": np.deg2rad(-91.0)}, "latitude_rad must be in"),
        ({"height_m": np.nan}, "height_m must be finite"),
        ({"wavelength_um": 0.0}, "wavelength_um must be positive"),
    ],
)
def test_troposphere_input_rejects_invalid_values(changes, message):
    with pytest.raises(ValueError, match=message):
        _input(**changes)


def test_water_vapor_pressure_rejects_invalid_inputs():
    model = Iers2010MendesPavlisTroposphere()
    with pytest.raises(ValueError, match="conversion domain"):
        model._water_vapor_pressure_hpa(1.0e308, 50.0)


def test_water_vapor_pressure_conversion_is_explicit():
    actual = Iers2010MendesPavlisTroposphere._water_vapor_pressure_hpa(293.15, 50.0)

    assert actual == pytest.approx(11.686412364256276, rel=0.0, abs=1.0e-14)
