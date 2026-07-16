"""LCRS solid-tide displacement of lunar retroreflectors."""
from __future__ import annotations

import numpy as np

from llrops.classes.relativistic.constants import GM_EARTH, GM_MOON, GM_SUN
from llrops.classes.displacement.constants import LUNAR_H2, LUNAR_L2, R_MOON
from llrops.classes.ephemerides import Ephemeris, require_tdb_epoch
from llrops.classes.frames.relativistic import RelativisticFrameTransform

from .base import ReflectorDisplacementInput


class LunarSolidTide:
    """Degree-2 lunar solid tide following Pavlov et al. (2016), Eq. (24)."""

    def __init__(
        self,
        ephemeris: Ephemeris,
        h2: float = LUNAR_H2,
        l2: float = LUNAR_L2,
        moon_radius_m: float = R_MOON,
        moon_gm_m3_s2: float = GM_MOON,
        earth_gm_m3_s2: float = GM_EARTH,
        sun_gm_m3_s2: float = GM_SUN,
    ) -> None:
        if not isinstance(ephemeris, Ephemeris):
            raise TypeError("ephemeris must implement Ephemeris.")
        if moon_radius_m <= 0.0 or moon_gm_m3_s2 <= 0.0:
            raise ValueError("moon_radius_m and moon_gm_m3_s2 must be positive.")
        self.ephemeris = ephemeris
        self.h2 = h2
        self.l2 = l2
        self.moon_radius_m = moon_radius_m
        self.moon_gm_m3_s2 = moon_gm_m3_s2
        self.earth_gm_m3_s2 = earth_gm_m3_s2
        self.sun_gm_m3_s2 = sun_gm_m3_s2

    def displacement_lcrs_m(self, data: ReflectorDisplacementInput) -> np.ndarray:
        epoch = require_tdb_epoch(data.epoch_tdb, name="data.epoch_tdb")
        reflector = data.reflector_lcrs_m
        reflector_norm = float(np.linalg.norm(reflector))
        if reflector_norm <= 0.0:
            raise ValueError("reflector_lcrs_m must have a positive norm.")
        reflector_direction = reflector / reflector_norm

        transform = RelativisticFrameTransform(self.ephemeris)
        earth_lcrs = transform.bcrs2lcrs(
            self.ephemeris.body_position_bcrs("EARTH", epoch),
            epoch,
        )
        sun_lcrs = transform.bcrs2lcrs(
            self.ephemeris.body_position_bcrs("SUN", epoch),
            epoch,
        )

        def body_term(body_lcrs_m: np.ndarray, body_gm_m3_s2: float) -> np.ndarray:
            distance_m = float(np.linalg.norm(body_lcrs_m))
            if distance_m <= 0.0:
                raise RuntimeError("Ephemeris returned a zero Moon-to-body vector.")
            body_direction = body_lcrs_m / distance_m
            cosine = float(np.dot(body_direction, reflector_direction))
            radial = (
                0.5
                * self.h2
                * (3.0 * cosine * cosine - 1.0)
                * reflector_direction
            )
            tangential = (
                3.0
                * self.l2
                * cosine
                * (body_direction - cosine * reflector_direction)
            )
            scale = (
                body_gm_m3_s2
                * self.moon_radius_m**4
                / (self.moon_gm_m3_s2 * distance_m**3)
            )
            return scale * (radial + tangential)

        return body_term(earth_lcrs, self.earth_gm_m3_s2) + body_term(
            sun_lcrs,
            self.sun_gm_m3_s2,
        )
