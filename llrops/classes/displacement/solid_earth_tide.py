"""IERS 2010 solid-Earth tide station displacement."""
from __future__ import annotations

import numpy as np

from .base import StationDisplacementInput
from .terrestrial_geometry import enu2itrf, itrf2geodetic

try:
    import pysolid
except ImportError:  # pragma: no cover
    pysolid = None


class Iers2010SolidEarthTide:
    """Solid-Earth tide displacement evaluated through ``pysolid``."""

    def __init__(self, sampling_interval_s: float = 60.0) -> None:
        if sampling_interval_s <= 0.0:
            raise ValueError("sampling_interval_s must be positive.")
        self.sampling_interval_s = sampling_interval_s

    def displacement_itrf_m(self, data: StationDisplacementInput) -> np.ndarray:
        if pysolid is None:
            raise ImportError(
                "pysolid is required for Iers2010SolidEarthTide. "
                "Install it with `pip install pysolid`."
            )

        site = itrf2geodetic(data.station_itrf_m)
        start = data.epoch_utc.to_datetime()
        stop = data.epoch_utc.shifted(self.sampling_interval_s).to_datetime()

        try:
            _, east, north, up = pysolid.calc_solid_earth_tides_point(
                site.latitude_deg,
                site.longitude_deg,
                start,
                stop,
                step_sec=float(self.sampling_interval_s),
                display=False,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Solid-Earth tide evaluation failed for "
                f"{data.epoch_utc.isot(scale='utc')} at lat={site.latitude_deg:.9f} deg, "
                f"lon={site.longitude_deg:.9f} deg."
            ) from exc

        enu_m = np.array([float(east[0]), float(north[0]), float(up[0])])
        if not np.all(np.isfinite(enu_m)):
            raise RuntimeError("Solid-Earth tide model returned non-finite values.")
        return enu2itrf(
            enu_m,
            latitude_rad=site.latitude_rad,
            longitude_rad=site.longitude_rad,
        )
