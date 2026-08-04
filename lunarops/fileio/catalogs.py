"""Typed station/reflector catalogs and catalog-key resolution."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from lunarops.base.array_validation import catalog_vector3
from lunarops.base.constants import SECONDS_PER_DAY
from lunarops.base.epoch import Epoch, TimeScale
from lunarops.classes.displacement.terrestrial_geometry import (
    GeodeticPosition,
    itrf2geodetic,
)
from lunarops.fileio.archive import (
    atomic_text_writer,
    data_lines,
    decode_token,
    encode_token,
    format_float,
    open_text_reader,
    parse_float,
    parse_header,
)


# ---------------------------------------------------------------------------
# Catalog records
# ---------------------------------------------------------------------------
@dataclass(eq=False, repr=False)
class StationRecord:
    name: str
    itrf_xyz_m: Sequence[float]
    aliases: Sequence[str] = field(default_factory=tuple)
    itrf_velocity_m_per_year: Sequence[float] = (0.0, 0.0, 0.0)
    position_epoch_utc: str = "2010-01-01T00:00:00"

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Station catalog names must not be empty.")
        self.itrf_xyz_m = catalog_vector3(self.itrf_xyz_m, name="station.itrf_xyz_m")
        self.itrf_velocity_m_per_year = catalog_vector3(
            self.itrf_velocity_m_per_year,
            name="station.itrf_velocity_m_per_year",
        )
        self.aliases = tuple(str(alias).strip() for alias in self.aliases)
        if any(not alias for alias in self.aliases) or len(set(self.aliases)) != len(
            self.aliases
        ):
            raise ValueError("Station aliases must be non-empty and unique.")
        self.position_epoch_utc = str(self.position_epoch_utc).strip()
        Epoch.from_isot(self.position_epoch_utc, scale=TimeScale.UTC)

    def _position_epoch(self) -> Epoch:
        cached = getattr(self, "_position_epoch_cache", None)
        if cached is None:
            cached = Epoch.from_isot(self.position_epoch_utc, scale=TimeScale.UTC)
            self._position_epoch_cache = cached
        return cached

    @staticmethod
    def _utc_epoch(value: Epoch) -> Epoch:
        if not isinstance(value, Epoch):
            raise TypeError("Station catalog queries require an Epoch.")
        return value.require_scale(TimeScale.UTC, name="obstime_utc")

    def itrf_xyz_at(self, obstime_utc: Epoch) -> np.ndarray:
        """Linear station motion model: XYZ(t) = XYZ0 + V * (t - epoch)."""
        epoch = self._position_epoch()
        time = self._utc_epoch(obstime_utc)
        years = epoch.seconds_until(time) / (365.25 * SECONDS_PER_DAY)
        return np.asarray(self.itrf_xyz_m, dtype=float) + years * np.asarray(
            self.itrf_velocity_m_per_year, dtype=float
        )

    def geodetic_at(self, obstime_utc: Epoch) -> GeodeticPosition:
        return itrf2geodetic(self.itrf_xyz_at(obstime_utc))

    @property
    def geodetic(self) -> GeodeticPosition:
        return itrf2geodetic(self.itrf_xyz_m)

    @property
    def latitude_rad(self) -> float:
        return self.geodetic.latitude_rad

    @property
    def height_m(self) -> float:
        return self.geodetic.ellipsoidal_height_m

    def latitude_rad_at(self, obstime_utc: Epoch) -> float:
        return self.geodetic_at(obstime_utc).latitude_rad

    def height_m_at(self, obstime_utc: Epoch) -> float:
        return self.geodetic_at(obstime_utc).ellipsoidal_height_m


@dataclass(eq=False, repr=False)
class ReflectorRecord:
    name: str
    moon_fixed_xyz_m: Sequence[float]
    aliases: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Reflector catalog names must not be empty.")
        self.moon_fixed_xyz_m = catalog_vector3(
            self.moon_fixed_xyz_m,
            name="reflector.moon_fixed_xyz_m",
        )
        self.aliases = tuple(str(alias).strip() for alias in self.aliases)
        if any(not alias for alias in self.aliases) or len(set(self.aliases)) != len(
            self.aliases
        ):
            raise ValueError("Reflector aliases must be non-empty and unique.")


# ---------------------------------------------------------------------------
# Catalog key resolution
# ---------------------------------------------------------------------------
def _canonical_catalog_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def resolve_catalog_key(value: object, catalog: Dict[str, object], label: str) -> str:
    """Resolve exact keys, case-insensitive keys, aliases, and compact tokens."""
    raw = str(value or "").strip()
    if not raw:
        raise KeyError(f"{label} name is empty and cannot be resolved.")

    if raw in catalog:
        return raw

    raw_upper = raw.upper()
    for key in catalog:
        if key.upper() == raw_upper:
            return key

    target = _canonical_catalog_token(raw)
    for key, record in catalog.items():
        tokens = {
            _canonical_catalog_token(key),
            _canonical_catalog_token(getattr(record, "name", "")),
        }
        tokens.update(
            _canonical_catalog_token(alias) for alias in getattr(record, "aliases", ())
        )
        if target in tokens:
            return key

    raise KeyError(f"{label} '{raw}' not found in catalog.")


def first_resolvable_key(
    candidates: Sequence[object], catalog: Dict[str, object], label: str
) -> str:
    last_error = None
    for candidate in candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        try:
            return resolve_catalog_key(candidate, catalog, label)
        except KeyError as exc:
            last_error = exc
    raise last_error or KeyError(f"{label} could not be resolved.")


# ---------------------------------------------------------------------------
# Native catalog persistence and config-driven loading
# ---------------------------------------------------------------------------
def write_station_catalog(
    catalog: Dict[str, StationRecord],
    path: str | Path,
) -> Path:
    if any(not str(key).strip() for key in catalog):
        raise ValueError("Station catalog keys must not be empty.")
    if not all(isinstance(record, StationRecord) for record in catalog.values()):
        raise TypeError("Station catalogs must contain StationRecord values.")
    target = Path(path).expanduser()
    with atomic_text_writer(target, "stationCatalog") as stream:
        stream.write("frame ITRF\n")
        stream.write(f"recordCount {len(catalog)}\n")
        stream.write(
            "# key name x_m y_m z_m vx_m_per_year vy_m_per_year "
            "vz_m_per_year position_epoch_utc alias_count aliases...\n"
        )
        stream.write("data\n")
        for key, record in sorted(catalog.items()):
            position = np.asarray(record.itrf_xyz_m, dtype=float)
            velocity = np.asarray(record.itrf_velocity_m_per_year, dtype=float)
            fields = [
                encode_token(key),
                encode_token(record.name),
                *(format_float(value) for value in position),
                *(format_float(value) for value in velocity),
                encode_token(record.position_epoch_utc),
                str(len(record.aliases)),
                *(encode_token(alias) for alias in record.aliases),
            ]
            stream.write(" ".join(fields) + "\n")
    return target


def read_station_catalog(path: str | Path) -> Dict[str, StationRecord]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "stationCatalog")
        lines = iter(data_lines(stream))
        try:
            frame = next(lines).split()
            count_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(f"Truncated station catalog header in {source}.") from exc
        if (
            frame != ["frame", "ITRF"]
            or len(count_line) != 2
            or count_line[0] != "recordCount"
            or marker != "data"
        ):
            raise ValueError(f"Malformed station catalog header in {source}.")
        count = int(count_line[1])
        if count < 0:
            raise ValueError("Station catalog record count must be non-negative.")
        catalog: Dict[str, StationRecord] = {}
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                raise ValueError(f"Malformed station catalog row in {source}: {line!r}")
            alias_count = int(fields[9])
            if alias_count < 0:
                raise ValueError("Station alias count must be non-negative.")
            if len(fields) != 10 + alias_count:
                raise ValueError(f"Station alias count mismatch in {source}: {line!r}")
            key = decode_token(fields[0])
            if not key:
                raise ValueError("Station catalog keys must not be empty.")
            if key in catalog:
                raise ValueError(f"Duplicate station catalog key {key!r}.")
            catalog[key] = StationRecord(
                name=decode_token(fields[1]),
                itrf_xyz_m=[
                    parse_float(value, field="station position")
                    for value in fields[2:5]
                ],
                itrf_velocity_m_per_year=[
                    parse_float(value, field="station velocity")
                    for value in fields[5:8]
                ],
                position_epoch_utc=decode_token(fields[8]),
                aliases=tuple(decode_token(value) for value in fields[10:]),
            )
    if len(catalog) != count:
        raise ValueError(
            f"Station catalog declares {count} records, found {len(catalog)}."
        )
    return catalog


def write_reflector_catalog(
    catalog: Dict[str, ReflectorRecord],
    path: str | Path,
) -> Path:
    if any(not str(key).strip() for key in catalog):
        raise ValueError("Reflector catalog keys must not be empty.")
    if not all(isinstance(record, ReflectorRecord) for record in catalog.values()):
        raise TypeError("Reflector catalogs must contain ReflectorRecord values.")
    target = Path(path).expanduser()
    with atomic_text_writer(target, "reflectorCatalog") as stream:
        stream.write("frame MOON_PA\n")
        stream.write(f"recordCount {len(catalog)}\n")
        stream.write("# key name x_m y_m z_m alias_count aliases...\n")
        stream.write("data\n")
        for key, record in sorted(catalog.items()):
            position = np.asarray(record.moon_fixed_xyz_m, dtype=float)
            fields = [
                encode_token(key),
                encode_token(record.name),
                *(format_float(value) for value in position),
                str(len(record.aliases)),
                *(encode_token(alias) for alias in record.aliases),
            ]
            stream.write(" ".join(fields) + "\n")
    return target


def read_reflector_catalog(path: str | Path) -> Dict[str, ReflectorRecord]:
    source = Path(path).expanduser()
    with open_text_reader(source) as stream:
        parse_header(stream, "reflectorCatalog")
        lines = iter(data_lines(stream))
        try:
            frame = next(lines).split()
            count_line = next(lines).split()
            marker = next(lines)
        except StopIteration as exc:
            raise ValueError(
                f"Truncated reflector catalog header in {source}."
            ) from exc
        if (
            frame != ["frame", "MOON_PA"]
            or len(count_line) != 2
            or count_line[0] != "recordCount"
            or marker != "data"
        ):
            raise ValueError(f"Malformed reflector catalog header in {source}.")
        count = int(count_line[1])
        if count < 0:
            raise ValueError("Reflector catalog record count must be non-negative.")
        catalog: Dict[str, ReflectorRecord] = {}
        for line in lines:
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(
                    f"Malformed reflector catalog row in {source}: {line!r}"
                )
            alias_count = int(fields[5])
            if alias_count < 0:
                raise ValueError("Reflector alias count must be non-negative.")
            if len(fields) != 6 + alias_count:
                raise ValueError(
                    f"Reflector alias count mismatch in {source}: {line!r}"
                )
            key = decode_token(fields[0])
            if not key:
                raise ValueError("Reflector catalog keys must not be empty.")
            if key in catalog:
                raise ValueError(f"Duplicate reflector catalog key {key!r}.")
            catalog[key] = ReflectorRecord(
                name=decode_token(fields[1]),
                moon_fixed_xyz_m=[
                    parse_float(value, field="reflector position")
                    for value in fields[2:5]
                ],
                aliases=tuple(decode_token(value) for value in fields[6:]),
            )
    if len(catalog) != count:
        raise ValueError(
            f"Reflector catalog declares {count} records, found {len(catalog)}."
        )
    return catalog


def load_station_catalog(source) -> Dict[str, StationRecord]:
    """Build a station catalog.

    ``source`` may be
      * ``"builtin"`` -> :data:`lunarops.fileio.builtin_catalogs.STATIONS`
      * a path to a native ``stationCatalog`` text file
      * an already-built ``Dict[str, StationRecord]`` (passed through).
    """
    if isinstance(source, dict) and all(
        isinstance(v, StationRecord) for v in source.values()
    ):
        return source
    if source in (None, "builtin"):
        from lunarops.fileio.builtin_catalogs import STATIONS

        # Builtin catalogs are module-level constants.  Return an independent
        # graph so estimator/model-state updates cannot pollute later programs
        # or a fresh RunContext in the same Python process.
        return copy.deepcopy(STATIONS)
    return read_station_catalog(source)


def load_reflector_catalog(source) -> Dict[str, ReflectorRecord]:
    """Build a reflector catalog; see :func:`load_station_catalog`."""
    if isinstance(source, dict) and all(
        isinstance(v, ReflectorRecord) for v in source.values()
    ):
        return source
    if source in (None, "builtin"):
        from lunarops.fileio.builtin_catalogs import REFLECTORS

        # See load_station_catalog: reflector coordinates are mutable model
        # state during fitting, so builtin globals must never be handed out.
        return copy.deepcopy(REFLECTORS)
    return read_reflector_catalog(source)


__all__ = [
    "ReflectorRecord",
    "StationRecord",
    "first_resolvable_key",
    "load_reflector_catalog",
    "load_station_catalog",
    "read_reflector_catalog",
    "read_station_catalog",
    "resolve_catalog_key",
    "write_reflector_catalog",
    "write_station_catalog",
]
