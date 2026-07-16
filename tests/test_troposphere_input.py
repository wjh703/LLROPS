from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from llrops.classes.delays import (
    Iers2010MendesPavlisTroposphere,
    TroposphereInput,
    ZeroTroposphereDelay,
)


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
        data.pressure_hpa = 900.0


def test_troposphere_models_consume_input_object():
    data = _input()

    assert ZeroTroposphereDelay().slant_delay_m(data) == 0.0
    delay = Iers2010MendesPavlisTroposphere().slant_delay_m(data)
    assert np.isfinite(delay)
    assert delay > 0.0
