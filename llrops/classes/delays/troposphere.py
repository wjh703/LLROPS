from __future__ import annotations

import numpy as np

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
    ):
        xc = 375.0
        k0 = 238.0185
        k1 = 19990.975
        k2 = 57.362
        k3 = 579.55174
        w0 = 295.235
        w1 = 2.6422
        w2 = -0.032380
        w3 = 0.004028

        sigma = 1.0 / lambda_um
        f = 1.0 - 0.00266 * np.cos(np.deg2rad(2.0 * latitude_deg)) - 0.28e-6 * ellip_ht_m
        corr = 1.0 + 0.534e-6 * (xc - 450.0)

        fh = 0.01 * corr * (
            k1 * (k0 + sigma**2) / ((k0 - sigma**2) ** 2)
            + k3 * (k2 + sigma**2) / ((k2 - sigma**2) ** 2)
        )
        zhd = 2.416579e-3 * fh * pressure_hpa / f

        fnh = 0.003101 * (
            w0 + 3.0 * w1 * sigma**2 + 5.0 * w2 * sigma**4 + 7.0 * w3 * sigma**6
        )
        zwd = 1.0e-4 * (5.316 * fnh - 3.759 * fh) * water_vapor_pressure_hpa / f
        ztd = zhd + zwd
        return float(ztd), float(zhd), float(zwd)

    @staticmethod
    def fcul_a(
        latitude_deg: float,
        height_m: float,
        temperature_k: float,
        elevation_deg: float,
    ) -> float:
        epsilon = np.deg2rad(elevation_deg)
        sine = np.sin(epsilon)
        t_c = temperature_k - 273.15
        cosphi = np.cos(np.deg2rad(latitude_deg))

        a10 = 0.121008e-2
        a11 = 0.17295e-5
        a12 = 0.3191e-4
        a13 = -0.18478e-7

        a20 = 0.304965e-2
        a21 = 0.2346e-5
        a22 = -0.1035e-3
        a23 = -0.1856e-7

        a30 = 0.68777e-1
        a31 = 0.1972e-4
        a32 = -0.3458e-2
        a33 = 0.1060e-6

        A1 = a10 + a11 * t_c + a12 * cosphi + a13 * height_m
        A2 = a20 + a21 * t_c + a22 * cosphi + a23 * height_m
        A3 = a30 + a31 * t_c + a32 * cosphi + a33 * height_m

        map_zen = 1.0 + A1 / (1.0 + A2 / (1.0 + A3))
        return float(map_zen / (sine + A1 / (sine + A2 / (sine + A3))))

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
