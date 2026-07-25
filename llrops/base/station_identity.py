"""Canonical station identity data shared by catalogs and estimators."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class _StationIdentity:
    ilrs_code: str
    display_name: str
    aliases: tuple[str, ...]


_STATIONS: dict[str, _StationIdentity] = {
    "APOLLO": _StationIdentity(
        "70610",
        "Apache Point Observatory",
        ("APOL", "APACHE", "APACHEPOINT", "7045"),
    ),
    "GRASSE": _StationIdentity(
        "01910",
        "Grasse",
        ("GRSM", "CERGA", "COTEDAZUR", "OCA", "7845"),
    ),
    "HALEAKALA": _StationIdentity("56610", "Haleakala", ("HALE", "HALL")),
    "MATERA": _StationIdentity("07941", "Matera", ("MATM", "MATE")),
    "MCDONALD": _StationIdentity("71110", "McDonald 2.70", ("MDOL",)),
    "MLRS1": _StationIdentity("71111", "McDonald MLRS1", ()),
    "MLRS2": _StationIdentity("71112", "McDonald MLRS2", ()),
    "WETTZELL": _StationIdentity(
        "08834",
        "Wettzell",
        ("WETZELL", "WETL", "WLRS"),
    ),
}


def station_token(value: object) -> str:
    """Return the punctuation-insensitive station token used for lookup."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


_ALIAS_TO_CANONICAL = {
    station_token(alias): canonical
    for canonical, identity in _STATIONS.items()
    for alias in (
        canonical,
        identity.ilrs_code,
        identity.display_name,
        *identity.aliases,
    )
}


def canonical_station_id(value: object) -> str:
    """Return one stable identifier for a built-in or custom station."""
    token = station_token(value)
    if not token:
        raise ValueError("Station identifier must not be empty.")
    return _ALIAS_TO_CANONICAL.get(token, token)


def station_aliases(value: object) -> tuple[str, ...]:
    station = canonical_station_id(value)
    identity = _STATIONS.get(station)
    if identity is None:
        return ()
    return (identity.ilrs_code, *identity.aliases)


def station_ilrs_code(value: object) -> str:
    station = canonical_station_id(value)
    identity = _STATIONS.get(station)
    if identity is None:
        raise ValueError(f"No ILRS code is registered for station {station!r}.")
    return identity.ilrs_code


def station_display_name(value: object) -> str:
    station = canonical_station_id(value)
    identity = _STATIONS.get(station)
    return station if identity is None else identity.display_name


__all__ = [
    "canonical_station_id",
    "station_aliases",
    "station_display_name",
    "station_ilrs_code",
    "station_token",
]
