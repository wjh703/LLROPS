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


def normalize_station_key(value: object) -> str:
    """Return a punctuation-insensitive key for a station name."""
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


_ALIAS_TO_CANONICAL = {
    normalize_station_key(alias): canonical
    for canonical, identity in _STATIONS.items()
    for alias in (
        canonical,
        identity.ilrs_code,
        identity.display_name,
        *identity.aliases,
    )
}


def canonical_station_id(value: object) -> str:
    """Return a stable identifier for a built-in or custom station.

    Registered aliases resolve to their canonical ID. Unknown names are
    normalized so callers that support custom stations can still use them.
    Use :func:`registered_station_id` when unknown names must be rejected.
    """
    token = normalize_station_key(value)
    if not token:
        raise ValueError("Station identifier must not be empty.")
    return _ALIAS_TO_CANONICAL.get(token, token)


def registered_station_id(value: object) -> str:
    """Resolve a name to a registered station, rejecting unknown names."""
    token = normalize_station_key(value)
    if not token:
        raise ValueError("Station identifier must not be empty.")
    try:
        return _ALIAS_TO_CANONICAL[token]
    except KeyError as exc:
        raise ValueError(f"Unknown registered station {value!r}.") from exc


def station_aliases(value: object) -> tuple[str, ...]:
    station = registered_station_id(value)
    identity = _STATIONS[station]
    return (identity.ilrs_code, *identity.aliases)


def station_names(value: object) -> tuple[str, ...]:
    """Return every registered spelling accepted for a station."""
    station = registered_station_id(value)
    identity = _STATIONS[station]
    names = (station, identity.ilrs_code, identity.display_name, *identity.aliases)
    return tuple(dict.fromkeys(names))


def station_ilrs_code(value: object) -> str:
    station = registered_station_id(value)
    return _STATIONS[station].ilrs_code


def station_display_name(value: object) -> str:
    station = registered_station_id(value)
    return _STATIONS[station].display_name


__all__ = [
    "canonical_station_id",
    "normalize_station_key",
    "registered_station_id",
    "station_aliases",
    "station_display_name",
    "station_ilrs_code",
    "station_names",
]
