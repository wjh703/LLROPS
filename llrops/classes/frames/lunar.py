"""Lunar principal-axis and LCRS rotations."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from llrops.base.epoch import Epoch
from llrops.classes.ephemerides import Ephemeris, require_tdb_epoch

from llrops.base.validation import vector3


class LunarFrameTransform:
    def __init__(self, ephemeris: Ephemeris) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        self.ephemeris = ephemeris

    def pa2lcrs(self, position_pa_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return self.ephemeris.pa2lcrs_matrix(epoch) @ vector3(
            position_pa_m,
            name="position_pa_m",
        )

    def lcrs2pa(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return self.ephemeris.pa2lcrs_matrix(epoch).T @ vector3(
            position_lcrs_m,
            name="position_lcrs_m",
        )


__all__ = ["LunarFrameTransform"]
