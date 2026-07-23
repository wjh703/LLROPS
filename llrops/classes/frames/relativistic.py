"""Relativistic BCRS/GCRS/LCRS spatial transformations."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from llrops.base.constants import C2
from llrops.classes.relativistic.constants import (
    EARTH_EXTERNAL_POTENTIAL_BODIES,
    GM_BY_BODY,
    L_B_MINUS_L_G,
    MOON_EXTERNAL_POTENTIAL_BODIES,
)
from llrops.base.epoch import Epoch
from llrops.classes.ephemerides import Ephemeris, require_tdb_epoch

from llrops.base.array_validation import vector3


class RelativisticFrameTransform:
    def __init__(self, ephemeris: Ephemeris) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        self.ephemeris = ephemeris

    def external_potential(
        self,
        center: str,
        epoch_tdb: Epoch,
        bodies: Iterable[str],
    ) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        center_position = self.ephemeris.body_position_bcrs(center, epoch)
        total = 0.0
        for body in bodies:
            key = str(body).strip().upper()
            try:
                gm = GM_BY_BODY[key]
            except KeyError:
                raise KeyError(f"No gravitational parameter configured for body {body!r}.") from None
            displacement = self.ephemeris.body_position_bcrs(body, epoch) - center_position
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0:
                raise RuntimeError(f"Ephemeris returned coincident positions for {center!r} and {body!r}.")
            total += gm / distance
        return float(total)

    def gcrs2bcrs(self, position_gcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        earth = self.ephemeris.body_state_bcrs("EARTH", epoch)
        position = vector3(position_gcrs_m, name="position_gcrs_m")
        potential = self.external_potential(
            "EARTH",
            epoch,
            EARTH_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - L_B_MINUS_L_G - potential / C2
        tdb_position = (
            scale * position
            - 0.5 * (np.dot(earth.velocity_mps, position) / C2) * earth.velocity_mps
        )
        return earth.position_m + tdb_position

    def bcrs2gcrs(self, position_bcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        earth = self.ephemeris.body_state_bcrs("EARTH", epoch)
        relative = vector3(position_bcrs_m, name="position_bcrs_m") - earth.position_m
        potential = self.external_potential(
            "EARTH",
            epoch,
            EARTH_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 + L_B_MINUS_L_G + potential / C2
        return (
            scale * relative
            + 0.5 * (np.dot(earth.velocity_mps, relative) / C2) * earth.velocity_mps
        )

    def lcrs2bcrs(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        moon = self.ephemeris.body_state_bcrs("MOON", epoch)
        position = vector3(position_lcrs_m, name="position_lcrs_m")
        potential = self.external_potential(
            "MOON",
            epoch,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 - self.ephemeris.lb_minus_ll - potential / C2
        tdb_position = (
            scale * position
            - 0.5 * (np.dot(moon.velocity_mps, position) / C2) * moon.velocity_mps
        )
        return moon.position_m + tdb_position

    def bcrs2lcrs(self, position_bcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        moon = self.ephemeris.body_state_bcrs("MOON", epoch)
        relative = vector3(position_bcrs_m, name="position_bcrs_m") - moon.position_m
        potential = self.external_potential(
            "MOON",
            epoch,
            MOON_EXTERNAL_POTENTIAL_BODIES,
        )
        scale = 1.0 + self.ephemeris.lb_minus_ll + potential / C2
        return (
            scale * relative
            + 0.5 * (np.dot(moon.velocity_mps, relative) / C2) * moon.velocity_mps
        )

    def lcrs2gcrs(self, position_lcrs_m: Sequence[float], epoch_tdb: Epoch) -> np.ndarray:
        return self.bcrs2gcrs(self.lcrs2bcrs(position_lcrs_m, epoch_tdb), epoch_tdb)


__all__ = ["RelativisticFrameTransform"]
