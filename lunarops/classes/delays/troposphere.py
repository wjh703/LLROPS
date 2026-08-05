from __future__ import annotations

import numpy as np

import lunarops._iers2010 as _iers2010
from lunarops.classes.delays.base import TroposphereDelay, TroposphereInput


def _finite_float(value: float, *, name: str) -> float:
    scalar = np.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


class Iers2010MendesPavlisTroposphere(TroposphereDelay):
    """Optical troposphere model from IERS Conventions 2010 S9.1.

    The model consumes a :class:`TroposphereInput`, keeping the atmospheric,
    station, wavelength, and line-of-sight inputs together as one immutable
    value object.
    """

    def __init__(self, elevation_floor_deg: float = 3.0) -> None:
        minimum = _finite_float(elevation_floor_deg, name="elevation_floor_deg")
        if not 0.0 <= minimum <= 90.0:
            raise ValueError("elevation_floor_deg must be in [0, 90].")
        self.elevation_floor_deg = minimum

    @property
    def elevation_floor_rad(self) -> float:
        return float(np.deg2rad(self.elevation_floor_deg))

    @staticmethod
    def _water_vapor_pressure_hpa(
        temperature_k: float,
        relative_humidity_percent: float,
    ) -> float:
        temperature_k = _finite_float(temperature_k, name="temperature_k")
        relative_humidity_percent = _finite_float(
            relative_humidity_percent,
            name="relative_humidity_percent",
        )
        if temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= relative_humidity_percent <= 100.0:
            raise ValueError("relative_humidity_percent must be in [0, 100].")
        t_c = temperature_k - 273.15
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
        result = float((relative_humidity_percent / 100.0) * e_s)
        if not np.isfinite(result):
            raise ValueError("temperature_k is outside the water-vapour conversion domain.")
        return result

    @staticmethod
    def fculzd_hpa(
        latitude_deg: float,
        ellip_ht_m: float,
        pressure_hpa: float,
        water_vapor_pressure_hpa: float,
        lambda_um: float,
    ) -> tuple[float, float, float]:
        latitude_deg = _finite_float(latitude_deg, name="latitude_deg")
        ellip_ht_m = _finite_float(ellip_ht_m, name="ellip_ht_m")
        pressure_hpa = _finite_float(pressure_hpa, name="pressure_hpa")
        water_vapor_pressure_hpa = _finite_float(
            water_vapor_pressure_hpa,
            name="water_vapor_pressure_hpa",
        )
        lambda_um = _finite_float(lambda_um, name="lambda_um")
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be in [-90, 90].")
        if pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be positive.")
        if water_vapor_pressure_hpa < 0.0:
            raise ValueError("water_vapor_pressure_hpa must be non-negative.")
        if lambda_um <= 0.0:
            raise ValueError("lambda_um must be positive.")
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
        latitude_deg = _finite_float(latitude_deg, name="latitude_deg")
        height_m = _finite_float(height_m, name="height_m")
        temperature_k = _finite_float(temperature_k, name="temperature_k")
        elevation_deg = _finite_float(elevation_deg, name="elevation_deg")
        if not -90.0 <= latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be in [-90, 90].")
        if temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= elevation_deg <= 90.0:
            raise ValueError("elevation_deg must be in [0, 90].")
        return _iers2010.fcul_a(
            latitude_deg,
            height_m,
            temperature_k,
            elevation_deg,
        )

    def slant_delay_m(self, data: TroposphereInput) -> float:
        if not isinstance(data, TroposphereInput):
            raise TypeError("data must be a TroposphereInput.")
        elevation_deg = max(
            float(np.rad2deg(data.elevation_rad)),
            self.elevation_floor_deg,
        )
        latitude_deg = float(np.rad2deg(data.latitude_rad))
        wvp_hpa = self._water_vapor_pressure_hpa(
            data.temperature_k,
            data.relative_humidity_percent,
        )
        ellipsoidal_height_m = data.height_m
        mean_sea_level_height_m = data.height_m
        ztd, _, _ = self.fculzd_hpa(
            latitude_deg=latitude_deg,
            ellip_ht_m=ellipsoidal_height_m,
            pressure_hpa=data.pressure_hpa,
            water_vapor_pressure_hpa=wvp_hpa,
            lambda_um=data.wavelength_um,
        )
        mapping = self.fcul_a(
            latitude_deg=latitude_deg,
            height_m=mean_sea_level_height_m,
            temperature_k=data.temperature_k,
            elevation_deg=elevation_deg,
        )
        return float(ztd * mapping)
