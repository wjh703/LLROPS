"""Optional longitude-libration corrections applied to lunar orientation."""
from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import erfa
import numpy as np

from llrops.base.epoch import Epoch, TimeScale

from .base import require_tdb_epoch

_MAS2RAD = np.deg2rad(1.0 / 3_600_000.0)
_JULIAN_CENTURY_DAYS = 36525.0


class LongitudeLibrationModel(StrEnum):
    NONE = "none"
    INPOP21A = "inpop21a"


def normalize_longitude_libration_model(value) -> LongitudeLibrationModel:
    if isinstance(value, LongitudeLibrationModel):
        return value
    if value is None:
        return LongitudeLibrationModel.NONE
    text = str(value).strip().lower()
    if text in {"", "no", "off", "false", "0"}:
        return LongitudeLibrationModel.NONE
    try:
        return LongitudeLibrationModel(text)
    except ValueError:
        allowed = ", ".join(model.value for model in LongitudeLibrationModel)
        raise ValueError(
            f"Unsupported longitude-libration model {value!r}; expected one of: {allowed}."
        ) from None


@runtime_checkable
class LongitudeLibrationCorrection(Protocol):
    def correction_rad(self, epoch: Epoch, *, j2000_tdb: Epoch) -> float:
        ...


class ZeroLongitudeLibration:
    def correction_rad(self, epoch: Epoch, *, j2000_tdb: Epoch) -> float:
        require_tdb_epoch(epoch)
        require_tdb_epoch(j2000_tdb, name="j2000_tdb")
        return 0.0


class Inpop21aLongitudeLibration:
    """INPOP21a Table 6 correction to lunar longitude libration."""

    def correction_rad(self, epoch: Epoch, *, j2000_tdb: Epoch) -> float:
        epoch = require_tdb_epoch(epoch)
        j2000_tdb = require_tdb_epoch(j2000_tdb, name="j2000_tdb")
        centuries = (
            (epoch.jd1 - j2000_tdb.jd1) + (epoch.jd2 - j2000_tdb.jd2)
        ) / _JULIAN_CENTURY_DAYS
        lunar_anomaly = float(erfa.fal03(centuries))
        solar_anomaly = float(erfa.falp03(centuries))
        argument_latitude = float(erfa.faf03(centuries))
        elongation = float(erfa.fad03(centuries))
        correction_mas = (
            4.5 * np.cos(solar_anomaly)
            + 1.8 * np.cos(2.0 * lunar_anomaly - 2.0 * elongation)
            + 10.5 * np.cos(2.0 * argument_latitude - 2.0 * lunar_anomaly)
        )
        return float(correction_mas * _MAS2RAD)


def make_longitude_libration_correction(value) -> LongitudeLibrationCorrection:
    model = normalize_longitude_libration_model(value)
    if model is LongitudeLibrationModel.NONE:
        return ZeroLongitudeLibration()
    if model is LongitudeLibrationModel.INPOP21A:
        return Inpop21aLongitudeLibration()
    raise AssertionError(f"Unhandled longitude-libration model: {model!r}")


__all__ = [
    "Inpop21aLongitudeLibration",
    "LongitudeLibrationCorrection",
    "LongitudeLibrationModel",
    "ZeroLongitudeLibration",
    "make_longitude_libration_correction",
    "normalize_longitude_libration_model",
]
