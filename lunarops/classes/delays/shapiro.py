from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from lunarops.base.constants import C2
from lunarops.base.array_validation import vector3
from lunarops.classes.relativistic.constants import GM_BY_BODY
from lunarops.base.epoch import Epoch
from lunarops.classes.ephemerides import Ephemeris, require_tdb_epoch
from lunarops.classes.delays.base import GravitationalDelay


# IERS Conventions 2010 S11.2 (Eq. 11.17) one-way path delay used for LLR.
# Pavlov, Williams & Suvorkin (2016) S4 explicitly include Sun, Earth, Moon,
# Jupiter and Saturn as point masses contributing to the gravitational delay.
_DEFAULT_LLR_SHAPIRO_BODIES = (
    "SUN",
    "EARTH",
    "MOON",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
)


class Iers2010ShapiroDelay(GravitationalDelay):
    """IERS 2010 Eq. (11.17) one-way gravitational path delay."""

    def __init__(
        self,
        ephemeris: Ephemeris,
        bodies: Iterable[str] = _DEFAULT_LLR_SHAPIRO_BODIES,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if isinstance(bodies, str):
            bodies = (bodies,)
        normalized_bodies = tuple(
            dict.fromkeys(str(body).strip().upper() for body in bodies)
        )
        if not normalized_bodies:
            raise ValueError("Iers2010ShapiroDelay requires at least one gravitating body.")
        unknown = [body for body in normalized_bodies if body not in GM_BY_BODY]
        if unknown:
            raise KeyError(f"No gravitational parameter configured for: {unknown!r}")
        self.ephemeris = ephemeris
        self.bodies = normalized_bodies

    def _body_position_bcrs(self, body: str, epoch: Epoch) -> np.ndarray:
        return vector3(
            self.ephemeris.body_position_bcrs(body, epoch),
            name=f"{body} BCRS position",
        )

    def path_delay_m(
        self,
        transmitter_bcrs_m: Sequence[float],
        receiver_bcrs_m: Sequence[float],
        epoch_tdb: Epoch,
    ) -> float:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        x1 = vector3(transmitter_bcrs_m, name="transmitter_bcrs_m")
        x2 = vector3(receiver_bcrs_m, name="receiver_bcrs_m")
        rho = float(np.linalg.norm(x2 - x1))

        total = 0.0
        for body in self.bodies:
            xb = self._body_position_bcrs(body, epoch)
            r1 = float(np.linalg.norm(x1 - xb))
            r2 = float(np.linalg.norm(x2 - xb))
            denom = r1 + r2 - rho
            numer = r1 + r2 + rho
            if denom <= 0.0 or numer <= 0.0:
                continue
            total += 2.0 * GM_BY_BODY[body] / C2 * np.log(numer / denom)
        return float(total)
