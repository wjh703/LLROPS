from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence

from llrops.base.epoch import Epoch


DEFAULT_STATION_ALIASES: dict[str, str] = {
    "APOLLO": "APOLLO",
    "APOL": "APOLLO",
    "APACHEPOINT": "APOLLO",
    "APACHEPOINTOBSERVATORY": "APOLLO",
    "7045": "APOLLO",
    "70610": "APOLLO",
    "GRASSE": "GRASSE",
    "GRSM": "GRASSE",
    "COTEDAZUR": "GRASSE",
    "OCA": "GRASSE",
    "7845": "GRASSE",
    "01910": "GRASSE",
    "HALEAKALA": "HALEAKALA",
    "HALE": "HALEAKALA",
    "HALL": "HALEAKALA",
    "56610": "HALEAKALA",
    "MATERA": "MATERA",
    "MATM": "MATERA",
    "MATE": "MATERA",
    "07941": "MATERA",
    "MCDONALD": "MCDONALD",
    "MDOL": "MCDONALD",
    "71110": "MCDONALD",
    "MLRS1": "MLRS1",
    "71111": "MLRS1",
    "MLRS2": "MLRS2",
    "71112": "MLRS2",
    "WETTZELL": "WETTZELL",
    "WETZELL": "WETTZELL",
    "WETL": "WETTZELL",
    "08834": "WETTZELL",
}


def station_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def normalize_station(value: object) -> Optional[str]:
    """Map a catalog key, station name, or ILRS code to a canonical station.

    Station aliases are part of the built-in catalog normalization rules, not a
    per-table YAML schema.
    """
    token = station_token(value)
    if not token:
        return None
    normalized = {station_token(alias): str(station).strip().upper() for alias, station in DEFAULT_STATION_ALIASES.items()}
    if token in normalized:
        return normalized[token]
    for alias, station in normalized.items():
        if alias and alias in token:
            return station
    return None


def _candidate_list(station_values: Sequence[object] | object) -> list[object]:
    if isinstance(station_values, (str, bytes)) or not isinstance(station_values, Iterable):
        return [station_values]
    return list(station_values)


def _parse_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip().replace("/", "-")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return datetime.strptime(str(value).strip(), "%Y/%m/%d").date()


@dataclass(frozen=True, slots=True)
class RangeBiasEntry:
    """Fixed station range-bias segment.

    ``bias_two_way_cm`` is the correction in centimetres of two-way light-travel
    distance.  Interval ends are exclusive.
    """

    station: str
    start: date
    end: date
    bias_two_way_cm: float
    source: str | None = None

    def __post_init__(self) -> None:
        station = normalize_station(self.station) or str(self.station).strip().upper()
        start = _parse_date(self.start)
        end = _parse_date(self.end)
        value = float(self.bias_two_way_cm)
        if not station:
            raise ValueError("RangeBiasEntry.station must not be empty.")
        if end <= start:
            raise ValueError(f"RangeBiasEntry end must be after start for {station}.")
        if not math.isfinite(value):
            raise ValueError(f"RangeBiasEntry.bias_two_way_cm must be finite for {station}.")
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "bias_two_way_cm", value)
        object.__setattr__(self, "source", None if self.source is None else str(self.source).strip() or None)

    @classmethod
    def from_config_item(cls, item: object, *, default_source: str | None = None) -> "RangeBiasEntry":
        """Parse one declarative bias row.

        Supported row forms are deliberately small and explicit::

            APOLLO 2020-01-01/2021-01-01 1.25
            [APOLLO, 2020-01-01/2021-01-01, 1.25]
            [APOLLO, 2020-01-01, 2021-01-01, 1.25]
            {station: APOLLO, interval: 2020-01-01/2021-01-01, biasCm: 1.25}
        """
        source = default_source
        if isinstance(item, str):
            parts = item.split()
            if len(parts) not in {3, 4}:
                raise ValueError(
                    "Range-bias string rows must be 'STATION start/end biasCm' "
                    f"or 'STATION start/end biasCm source', got {item!r}"
                )
            station, interval, value = parts[:3]
            if len(parts) == 4:
                source = parts[3]
            start, end = _parse_interval(interval)
            return cls(station=station, start=start, end=end, bias_two_way_cm=float(value), source=source)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            values = list(item)
            if len(values) == 3:
                station, interval, value = values
                start, end = _parse_interval(interval)
                return cls(station=str(station), start=start, end=end, bias_two_way_cm=float(value), source=source)
            if len(values) in {4, 5}:
                station, start, end, value = values[:4]
                if len(values) == 5:
                    source = str(values[4])
                return cls(station=str(station), start=_parse_date(start), end=_parse_date(end), bias_two_way_cm=float(value), source=source)
            raise ValueError(
                "Range-bias sequence rows must be [station, start/end, biasCm] "
                "or [station, start, end, biasCm]."
            )
        if isinstance(item, Mapping):
            if "station" not in item:
                raise ValueError(f"Range-bias row requires 'station': {item!r}")
            if "interval" in item:
                start, end = _parse_interval(item["interval"])
            else:
                if "start" not in item or "end" not in item:
                    raise ValueError(f"Range-bias row requires 'interval' or start/end: {item!r}")
                start, end = _parse_date(item["start"]), _parse_date(item["end"])
            if "biasCm" not in item:
                raise ValueError(f"Range-bias row requires 'biasCm': {item!r}")
            item_source = item.get("source")
            return cls(
                station=str(item["station"]),
                start=start,
                end=end,
                bias_two_way_cm=float(item["biasCm"]),
                source=source if item_source is None else str(item_source) or source,
            )
        raise TypeError(f"Unsupported range-bias row: {item!r}")


def _parse_interval(value: object) -> tuple[date, date]:
    text = str(value).strip()
    if "/" not in text:
        raise ValueError(f"Interval must be written as 'start/end', got {value!r}")
    start, end = [part.strip() for part in text.split("/", 1)]
    if not start or not end:
        raise ValueError(f"Interval must be written as 'start/end', got {value!r}")
    return _parse_date(start), _parse_date(end)

def _d(value: str) -> date:
    return datetime.strptime(value, "%Y/%m/%d").date()


# INPOP21a Table 8, INPOP21a column: station biases over different periods.
INPOP21_RANGE_BIASES: tuple[RangeBiasEntry, ...] = (
    RangeBiasEntry("APOLLO", _d("2006/04/07"), _d("2010/11/01"), 0.03, "INPOP21a Table 8"),
    RangeBiasEntry("APOLLO", _d("2007/12/15"), _d("2008/06/30"), -3.93, "INPOP21a Table 8"),
    RangeBiasEntry("APOLLO", _d("2008/09/20"), _d("2009/06/20"), 3.22, "INPOP21a Table 8"),
    RangeBiasEntry("APOLLO", _d("2010/11/01"), _d("2012/04/07"), -6.28, "INPOP21a Table 8"),
    RangeBiasEntry("APOLLO", _d("2012/04/07"), _d("2013/09/02"), 8.85, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("1984/06/01"), _d("1986/06/13"), -17.12, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("1987/10/01"), _d("2005/08/01"), -5.41, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("1993/03/01"), _d("1996/10/01"), 9.81, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("1996/12/10"), _d("1997/01/18"), 14.32, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("1997/02/08"), _d("1998/06/24"), 20.79, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("2004/12/04"), _d("2004/12/07"), -5.53, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("2005/01/03"), _d("2005/01/06"), -4.53, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("2009/11/01"), _d("2014/01/01"), -0.99, "INPOP21a Table 8"),
    RangeBiasEntry("GRASSE", _d("2015/12/20"), _d("2015/12/21"), -88.05, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1984/11/01"), _d("1990/09/01"), 10.07, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1984/11/01"), _d("1986/04/01"), -0.72, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1986/04/02"), _d("1987/07/30"), 9.81, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1987/07/31"), _d("1987/08/14"), 1.86, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1985/06/09"), _d("1985/06/10"), -11.18, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1987/11/10"), _d("1988/02/18"), 18.57, "INPOP21a Table 8"),
    RangeBiasEntry("HALEAKALA", _d("1990/02/06"), _d("1990/09/01"), 13.36, "INPOP21a Table 8"),
    RangeBiasEntry("MATERA", _d("2003/01/01"), _d("2016/01/01"), 0.34, "INPOP21a Table 8"),
    RangeBiasEntry("MCDONALD", _d("1969/01/01"), _d("1985/07/01"), -46.56, "INPOP21a Table 8"),
    RangeBiasEntry("MCDONALD", _d("1971/12/01"), _d("1972/12/05"), 40.23, "INPOP21a Table 8"),
    RangeBiasEntry("MCDONALD", _d("1972/04/21"), _d("1972/04/27"), 129.56, "INPOP21a Table 8"),
    RangeBiasEntry("MCDONALD", _d("1974/08/18"), _d("1974/10/16"), -114.07, "INPOP21a Table 8"),
    RangeBiasEntry("MCDONALD", _d("1975/10/05"), _d("1976/03/01"), 26.87, "INPOP21a Table 8"),
    RangeBiasEntry("MCDONALD", _d("1983/12/01"), _d("1984/01/17"), -12.80, "INPOP21a Table 8"),
    RangeBiasEntry("MLRS1", _d("1983/08/01"), _d("1988/01/28"), 14.42, "INPOP21a Table 8"),
    RangeBiasEntry("WETTZELL", _d("2018/01/01"), _d("2025/01/01"), 0.0, "INPOP21a Table 8"),
)


@dataclass(frozen=True, slots=True)
class RangeBiasTable:
    """Station-indexed deterministic two-way range-bias table."""

    entries: tuple[RangeBiasEntry, ...]
    units: str = "cm two-way light distance"
    source: str | None = None
    _by_station: dict[str, tuple[RangeBiasEntry, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        by_station: dict[str, list[RangeBiasEntry]] = {}
        for entry in entries:
            by_station.setdefault(entry.station, []).append(entry)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "units", str(self.units).strip() or "cm two-way light distance")
        object.__setattr__(self, "source", None if self.source is None else str(self.source).strip() or None)
        object.__setattr__(
            self,
            "_by_station",
            {station: tuple(sorted(items, key=lambda item: (item.start, item.end))) for station, items in by_station.items()},
        )

    def canonical_station(self, value: object) -> Optional[str]:
        return normalize_station(value)

    def station_from_candidates(self, station_values: Sequence[object] | object) -> Optional[str]:
        for candidate in _candidate_list(station_values):
            station = self.canonical_station(candidate)
            if station is not None:
                return station
        return None

    def active_entries(self, station_values: Sequence[object] | object, obs_epoch_utc: Epoch) -> tuple[RangeBiasEntry, ...]:
        station = self.station_from_candidates(station_values)
        if station is None:
            return ()
        day = _utc_date(obs_epoch_utc)
        return tuple(entry for entry in self._by_station.get(station, ()) if entry.start <= day < entry.end)

    def two_way_cm(self, station_values: Sequence[object] | object, obs_epoch_utc: Epoch) -> float:
        return float(sum(entry.bias_two_way_cm for entry in self.active_entries(station_values, obs_epoch_utc)))

    def coverage_summary(self) -> dict[str, list[tuple[str, str]]]:
        return {
            station: [(entry.start.isoformat(), entry.end.isoformat()) for entry in entries]
            for station, entries in self._by_station.items()
        }

    @classmethod
    def from_mapping(cls, config: Mapping[str, object], *, source_file: str | Path | None = None) -> "RangeBiasTable":
        forbidden = sorted({"name", "aliases"} & set(config))
        if forbidden:
            raise ValueError(
                "Range-bias table config no longer accepts top-level "
                f"{', '.join(repr(key) for key in forbidden)}; use only 'file', 'source', and 'biases'."
            )
        source = config.get("source") or (str(source_file) if source_file else None)
        raw_biases = config.get("biases")
        if raw_biases is None:
            raise ValueError("Range-bias table config requires a 'biases' list.")
        if not isinstance(raw_biases, Sequence) or isinstance(raw_biases, (str, bytes)):
            raise TypeError("Range-bias 'biases' must be a list of rows.")
        entries = tuple(
            RangeBiasEntry.from_config_item(item, default_source=str(source) if source else None)
            for item in raw_biases
        )
        return cls(entries=entries, source=None if source is None else str(source))


def _utc_date(epoch: Epoch) -> date:
    if not isinstance(epoch, Epoch):
        raise TypeError("obs_epoch_utc must be an Epoch.")
    return date.fromisoformat(epoch.date_iso())


INPOP21_RANGE_BIAS_TABLE = RangeBiasTable(
    entries=INPOP21_RANGE_BIASES,
    source="INPOP21a Table 8",
)

BUILTIN_RANGE_BIAS_TABLES: dict[str, RangeBiasTable] = {
    "inpop21": INPOP21_RANGE_BIAS_TABLE,
}


def builtin_range_bias_table(name: object) -> RangeBiasTable:
    key = str(name).strip().lower()
    try:
        return BUILTIN_RANGE_BIAS_TABLES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown built-in range-bias table {name!r}. Available: {sorted(BUILTIN_RANGE_BIAS_TABLES)}"
        ) from exc


def load_range_bias_table(path: str | Path) -> RangeBiasTable:
    file = Path(path).expanduser()
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Range-bias table file must contain a mapping: {file}")
    return RangeBiasTable.from_mapping(data, source_file=file)


__all__ = [
    "BUILTIN_RANGE_BIAS_TABLES",
    "DEFAULT_STATION_ALIASES",
    "INPOP21_RANGE_BIASES",
    "INPOP21_RANGE_BIAS_TABLE",
    "RangeBiasEntry",
    "RangeBiasTable",
    "builtin_range_bias_table",
    "load_range_bias_table",
    "normalize_station",
    "station_token",
]
