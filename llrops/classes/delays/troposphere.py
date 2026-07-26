from __future__ import annotations

import numpy as np

from llrops import _iers2010
from llrops.classes.delays.base import TroposphereDelay, TroposphereInput


class Iers2010MendesPavlisTroposphere(TroposphereDelay):
    """Optical troposphere model from IERS Conventions 2010 S9.1.

    The model consumes a :class:`TroposphereInput`, keeping the atmospheric,
    station, wavelength, and line-of-sight inputs together as one immutable
    value object.
    """

    def __init__(self, min_elevation_deg: float = 3.0) -> None:
        self.min_elevation_deg = min_elevation_deg

    @staticmethod
    def _water_vapor_pressure_hpa(
        temperature_k: float,
        relative_humidity_percent: float,
    ) -> float:
        t_c = temperature_k - 273.15
        e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
        return float((relative_humidity_percent / 100.0) * e_s)

    @staticmethod
    def fculzd_hpa(
        latitude_deg: float,
        ellip_ht_m: float,
        pressure_hpa: float,
        water_vapor_pressure_hpa: float,
        lambda_um: float,
    ) -> tuple[float, float, float]:
        return _iers2010.fculzd_hpa(
            latitude_deg,
            ellip_ht_m,
            pressure_hpa,
            water_vapor_pressure_hpa,
            lambda_um,
        )

    @staticmethod
    def fcul_a(
        latitude_deg: float,
        height_m: float,
        temperature_k: float,
        elevation_deg: float,
    ) -> float:
        return _iers2010.fcul_a(
            latitude_deg,
            height_m,
            temperature_k,
            elevation_deg,
        )

    def slant_delay_m(self, data: TroposphereInput) -> float:
        elevation_deg = max(
            float(np.rad2deg(data.elevation_rad)),
            self.min_elevation_deg,
        )
        latitude_deg = float(np.rad2deg(data.latitude_rad))
        wvp_hpa = self._water_vapor_pressure_hpa(
            data.temperature_k,
            data.relative_humidity_percent,
        )
        ztd, _, _ = self.fculzd_hpa(
            latitude_deg=latitude_deg,
            ellip_ht_m=data.height_m,
            pressure_hpa=data.pressure_hpa,
            water_vapor_pressure_hpa=wvp_hpa,
            lambda_um=data.wavelength_um,
        )
        mapping = self.fcul_a(
            latitude_deg=latitude_deg,
            height_m=data.height_m,
            temperature_k=data.temperature_k,
            elevation_deg=elevation_deg,
        )
        return float(ztd * mapping)
