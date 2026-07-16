"""Station / reflector catalogs and catalog-key resolution.

Moved unchanged from ``llr_processor_refactored.pipeline`` (v24).  This module
owns the *data* side of catalogs; loading catalogs from configuration files is
implemented in :func:`load_station_catalog` / :func:`load_reflector_catalog`.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from llrops.base.constants import SECONDS_PER_DAY
from llrops.base.epoch import Epoch, TimeScale
from llrops.base.validation import catalog_vector3

from llrops.classes.displacement.geometry import GeodeticPosition, itrf2geodetic



# ---------------------------------------------------------------------------
# Catalog records
# ---------------------------------------------------------------------------
@dataclass
class StationRecord:
    name: str
    itrf_xyz_m: Sequence[float]
    aliases: Sequence[str] = field(default_factory=tuple)
    itrf_velocity_m_per_year: Sequence[float] = (0.0, 0.0, 0.0)
    position_epoch_utc: str = "2010-01-01T00:00:00"

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.itrf_xyz_m = catalog_vector3(self.itrf_xyz_m, name="station.itrf_xyz_m")
        self.itrf_velocity_m_per_year = catalog_vector3(
            self.itrf_velocity_m_per_year,
            name="station.itrf_velocity_m_per_year",
        )
        self.aliases = tuple(str(alias) for alias in self.aliases)
        self.position_epoch_utc = str(self.position_epoch_utc)

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
        return self.geodetic.height_m

    def latitude_rad_at(self, obstime_utc: Epoch) -> float:
        return self.geodetic_at(obstime_utc).latitude_rad

    def height_m_at(self, obstime_utc: Epoch) -> float:
        return self.geodetic_at(obstime_utc).height_m


@dataclass
class ReflectorRecord:
    name: str
    moon_fixed_xyz_m: Sequence[float]
    aliases: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.moon_fixed_xyz_m = catalog_vector3(
            self.moon_fixed_xyz_m,
            name="reflector.moon_fixed_xyz_m",
        )
        self.aliases = tuple(str(alias) for alias in self.aliases)


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
        tokens = {_canonical_catalog_token(key), _canonical_catalog_token(getattr(record, "name", ""))}
        tokens.update(_canonical_catalog_token(alias) for alias in getattr(record, "aliases", ()))
        if target in tokens:
            return key

    raise KeyError(f"{label} '{raw}' not found in catalog.")


def first_resolvable_key(candidates: Sequence[object], catalog: Dict[str, object], label: str) -> str:
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
# Config-driven catalog loading (new in llrops)
# ---------------------------------------------------------------------------
def _load_structured(path) -> object:
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        import yaml  # optional dependency, only needed for YAML catalogs
        return yaml.safe_load(text)
    return json.loads(text)


def load_station_catalog(source) -> Dict[str, StationRecord]:
    """Build a station catalog.

    ``source`` may be
      * ``"builtin"`` -> :data:`llrops.fileio.sample_catalogs.STATIONS`
      * a path to a JSON/YAML file: ``{key: {name, itrf_xyz_m, aliases,
        itrf_velocity_m_per_year, position_epoch_utc}}``
      * an already-built ``Dict[str, StationRecord]`` (passed through).
    """
    if isinstance(source, dict) and all(isinstance(v, StationRecord) for v in source.values()):
        return source
    if source in (None, "builtin"):
        from llrops.fileio.sample_catalogs import STATIONS

        # Builtin catalogs are module-level constants.  Return an independent
        # graph so estimator/model-state updates cannot pollute later programs
        # or a fresh RunContext in the same Python process.
        return copy.deepcopy(STATIONS)
    data = _load_structured(source)
    catalog: Dict[str, StationRecord] = {}
    for key, entry in data.items():
        catalog[key] = StationRecord(
            name=entry.get("name", key),
            itrf_xyz_m=entry["itrf_xyz_m"],
            aliases=tuple(entry.get("aliases", ())),
            itrf_velocity_m_per_year=tuple(entry.get("itrf_velocity_m_per_year", (0.0, 0.0, 0.0))),
            position_epoch_utc=entry.get("position_epoch_utc", "2010-01-01T00:00:00"),
        )
    return catalog


def load_reflector_catalog(source) -> Dict[str, ReflectorRecord]:
    """Build a reflector catalog; see :func:`load_station_catalog`."""
    if isinstance(source, dict) and all(isinstance(v, ReflectorRecord) for v in source.values()):
        return source
    if source in (None, "builtin"):
        from llrops.fileio.sample_catalogs import REFLECTORS

        # See load_station_catalog: reflector coordinates are mutable model
        # state during fitting, so builtin globals must never be handed out.
        return copy.deepcopy(REFLECTORS)
    data = _load_structured(source)
    catalog: Dict[str, ReflectorRecord] = {}
    for key, entry in data.items():
        catalog[key] = ReflectorRecord(
            name=entry.get("name", key),
            moon_fixed_xyz_m=entry["moon_fixed_xyz_m"],
            aliases=tuple(entry.get("aliases", ())),
        )
    return catalog
