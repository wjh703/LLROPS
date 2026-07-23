"""CALCEPH implementation of the LLR ephemeris interface."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from llrops.classes.relativistic.constants import l_b_minus_l_l_for_ephemeris

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.time_scale_converter import TimeScaleConverter

from .base import BodyState, Ephemeris, require_tdb_epoch, readonly_matrix3x3
from .longitude_libration import (
    LongitudeLibrationCorrection,
    make_longitude_libration_correction,
    normalize_longitude_libration_model,
)

_BODY_ID_BY_NAME = {
    "SSB": 0,
    "SOLAR SYSTEM BARYCENTER": 0,
    "MERCURY BARYCENTER": 1,
    "VENUS BARYCENTER": 2,
    "EARTH MOON BARYCENTER": 3,
    "EARTH BARYCENTER": 3,
    "MARS BARYCENTER": 4,
    "JUPITER BARYCENTER": 5,
    "SATURN BARYCENTER": 6,
    "URANUS BARYCENTER": 7,
    "NEPTUNE BARYCENTER": 8,
    "PLUTO BARYCENTER": 9,
    "SUN": 10,
    "MOON": 301,
    "EARTH": 399,
}
_J2000_TT_JD1 = 2451545.0
_J2000_TT_JD2 = 0.0


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=float,
    )


class CalcephEphemeris(Ephemeris):
    """INPOP/DE binary ephemeris read through :mod:`calcephpy`."""

    _LIBRATION_TARGET = 15
    _TT_MINUS_TDB_TARGET = 16

    def __init__(
        self,
        file: str | Path,
        *,
        longitude_libration=None,
    ) -> None:
        path = Path(file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CALCEPH ephemeris file not found: {path}")
        try:
            from calcephpy import CalcephBin, Constants
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise ImportError(
                "The CALCEPH ephemeris requires a working calcephpy installation. "
                f"Original import error: {exc}"
            ) from exc

        self._source_file = path
        self._lb_minus_ll = float(l_b_minus_l_l_for_ephemeris(path))
        self._longitude_libration_model = normalize_longitude_libration_model(
            longitude_libration
        )
        self._longitude_libration: LongitudeLibrationCorrection = (
            make_longitude_libration_correction(self._longitude_libration_model)
        )
        self._j2000_tdb: Epoch | None = None
        self._handle = CalcephBin.open(str(path))
        self._state_units = Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID
        self._angle_units = Constants.UNIT_RAD + Constants.UNIT_SEC

    @property
    def source_file(self) -> Path:
        return self._source_file

    @property
    def lb_minus_ll(self) -> float:
        return self._lb_minus_ll

    @property
    def longitude_libration_model(self) -> str:
        return self._longitude_libration_model.value

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("CALCEPH ephemeris is closed.")
        return self._handle

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:  # pragma: no cover - backend cleanup detail
                raise RuntimeError("CALCEPH ephemeris close() failed.") from exc

    def body_state_bcrs(self, body: str, epoch: Epoch) -> BodyState:
        epoch = require_tdb_epoch(epoch)
        key = str(body).strip().upper()
        try:
            target = _BODY_ID_BY_NAME[key]
        except KeyError:
            raise KeyError(f"Unknown CALCEPH body name: {body!r}") from None
        values = self._require_open().compute_unit(
            epoch.jd1,
            epoch.jd2,
            target,
            0,
            self._state_units,
        )
        state = np.asarray(values, dtype=float)
        if state.size < 6:
            raise RuntimeError(
                f"CALCEPH returned {state.size} state values for {key}; expected at least six."
            )
        return BodyState(
            position_m=state[:3] * 1000.0,
            velocity_mps=state[3:6] * 1000.0,
        )

    def libration_angles_rad(self, epoch: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch)
        values = self._require_open().compute_unit(
            epoch.jd1,
            epoch.jd2,
            self._LIBRATION_TARGET,
            0,
            self._angle_units,
        )
        angles = np.asarray(values, dtype=float)
        if angles.size < 3:
            raise RuntimeError("CALCEPH libration target returned fewer than three angles.")
        result = np.array(angles[:3], dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def _j2000_tdb_epoch(self) -> Epoch:
        if self._j2000_tdb is None:
            converter = TimeScaleConverter(self)
            self._j2000_tdb = converter.tt2tdb(
                Epoch(_J2000_TT_JD1, _J2000_TT_JD2, TimeScale.TT)
            )
        return self._j2000_tdb

    def longitude_libration_correction_rad(self, epoch: Epoch) -> float:
        epoch = require_tdb_epoch(epoch)
        if self.longitude_libration_model == "none":
            return 0.0
        return self._longitude_libration.correction_rad(
            epoch,
            j2000_tdb=self._j2000_tdb_epoch(),
        )

    def pa2lcrs_matrix(self, epoch: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch)
        phi, theta, psi = self.libration_angles_rad(epoch)
        psi += self.longitude_libration_correction_rad(epoch)
        lcrs2pa = _rotation_z(psi) @ _rotation_x(theta) @ _rotation_z(phi)
        return readonly_matrix3x3(lcrs2pa.T, name="pa2lcrs_matrix")

    @staticmethod
    def _looks_like_missing_target_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "target" in text
            and any(token in text for token in ("missing", "not found", "available", "unknown", "invalid"))
        ) or ("body" in text and "not" in text and "found" in text)

    def tdb_minus_tt_sec(self, epoch: Epoch) -> float | None:
        epoch = require_tdb_epoch(epoch)
        try:
            values = self._require_open().compute_unit(
                epoch.jd1,
                epoch.jd2,
                self._TT_MINUS_TDB_TARGET,
                0,
                self._angle_units,
            )
        except Exception as exc:
            if self._looks_like_missing_target_error(exc):
                return None
            raise RuntimeError(
                "CALCEPH failed while reading target 16 (TT−TDB) at "
                f"jd=({epoch.jd1}, {epoch.jd2})."
            ) from exc
        values = np.asarray(values, dtype=float)
        if values.size < 1 or not np.isfinite(values[0]):
            raise RuntimeError("CALCEPH target 16 returned an invalid TT−TDB value.")
        # CALCEPH target 16 stores TT−TDB; the public interface exposes TDB−TT.
        return -float(values[0])

    def require_tdb_minus_tt(self) -> None:
        if self.tdb_minus_tt_sec(Epoch(2451545.0, 0.0, TimeScale.TDB)) is None:
            raise RuntimeError(
                "The loaded CALCEPH ephemeris does not provide target 16 (TT−TDB)."
            )


def load_calceph_ephemeris(
    file: str | Path,
    *,
    longitude_libration=None,
) -> CalcephEphemeris:
    ephemeris = CalcephEphemeris(
        file,
        longitude_libration=longitude_libration,
    )
    try:
        ephemeris.require_tdb_minus_tt()
    except Exception:
        ephemeris.close()
        raise
    return ephemeris


__all__ = ["CalcephEphemeris", "load_calceph_ephemeris"]
