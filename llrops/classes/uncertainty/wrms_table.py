from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from llrops.base.constants import C
from llrops.base.epoch import Epoch
from llrops.classes.range_bias.table import normalize_station


def _parse_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip().replace("/", "-")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return datetime.strptime(str(value).strip(), "%Y/%m/%d").date()


@dataclass(frozen=True, slots=True)
class WrmsUncertaintyEntry:
    """Empirical normal-point uncertainty segment.

    ``wrms_two_way_m`` is a two-way range WRMS in metres.  The one-way range
    sigma used in LLROPS weighted residuals is ``0.5 * wrms_two_way_m``.
    Interval ends are exclusive.
    """

    group: str
    station: str
    start: date
    end: date
    wrms_two_way_m: float
    source: str | None = None

    def __post_init__(self) -> None:
        group = str(self.group).strip()
        station = normalize_station(self.station) or str(self.station).strip().upper()
        start = _parse_date(self.start)
        end = _parse_date(self.end)
        value = float(self.wrms_two_way_m)
        if not group or not station:
            raise ValueError("WRMS uncertainty group/station must not be empty.")
        if end <= start:
            raise ValueError(f"WRMS uncertainty end must be after start for {group}.")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"WRMS uncertainty must be positive and finite for {group}.")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "wrms_two_way_m", value)
        object.__setattr__(self, "source", None if self.source is None else str(self.source).strip() or None)

    @property
    def wrms_one_way_m(self) -> float:
        return 0.5 * self.wrms_two_way_m

    @property
    def uncertainty_two_way_s(self) -> float:
        return self.wrms_two_way_m / C

    @property
    def uncertainty_two_way_ps(self) -> float:
        return self.uncertainty_two_way_s * 1.0e12

    @property
    def uncertainty_raw_0p1ps(self) -> int:
        return int(round(self.uncertainty_two_way_ps * 10.0))

    @classmethod
    def from_config_item(cls, item: object, *, default_source: str | None = None) -> "WrmsUncertaintyEntry":
        """Parse one declarative WRMS row.

        Supported row forms are deliberately small and explicit::

            APOLLO 2020-01-01/2021-01-01 0.020 APO-test
            [APOLLO, 2020-01-01/2021-01-01, 0.020, APO-test]
            [APOLLO, 2020-01-01, 2021-01-01, 0.020, APO-test]
            {station: APOLLO, interval: 2020-01-01/2021-01-01, wrmsM: 0.020, group: APO-test}
        """
        source = default_source
        if isinstance(item, str):
            parts = item.split()
            if len(parts) not in {3, 4, 5}:
                raise ValueError(
                    "WRMS string rows must be 'STATION start/end wrmsM [group]' "
                    f"or 'STATION start/end wrmsM group source', got {item!r}"
                )
            station, interval, value = parts[:3]
            group = parts[3] if len(parts) >= 4 else station
            if len(parts) == 5:
                source = parts[4]
            start, end = _parse_interval(interval)
            return cls(group=str(group), station=station, start=start, end=end, wrms_two_way_m=float(value), source=source)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            values = list(item)
            if len(values) in {3, 4}:
                station, interval, value = values[:3]
                group = values[3] if len(values) == 4 else station
                start, end = _parse_interval(interval)
                return cls(group=str(group), station=str(station), start=start, end=end, wrms_two_way_m=float(value), source=source)
            if len(values) in {5, 6}:
                station, start, end, value, group = values[:5]
                if len(values) == 6:
                    source = str(values[5])
                return cls(group=str(group), station=str(station), start=_parse_date(start), end=_parse_date(end), wrms_two_way_m=float(value), source=source)
            raise ValueError(
                "WRMS sequence rows must be [station, start/end, wrmsM, group] "
                "or [station, start, end, wrmsM, group]."
            )
        if isinstance(item, Mapping):
            if "station" not in item:
                raise ValueError(f"WRMS row requires 'station': {item!r}")
            if "interval" in item:
                start, end = _parse_interval(item["interval"])
            else:
                if "start" not in item or "end" not in item:
                    raise ValueError(f"WRMS row requires 'interval' or start/end: {item!r}")
                start, end = _parse_date(item["start"]), _parse_date(item["end"])
            if "wrmsM" not in item:
                raise ValueError(f"WRMS row requires 'wrmsM': {item!r}")
            item_source = item.get("source")
            return cls(
                group=str(item.get("group") or item["station"]),
                station=str(item["station"]),
                start=start,
                end=end,
                wrms_two_way_m=float(item["wrmsM"]),
                source=source if item_source is None else str(item_source) or source,
            )
        raise TypeError(f"Unsupported WRMS uncertainty row: {item!r}")


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


# Empirical WRMS uncertainty table supplied with the project.  Intervals are
# start-inclusive and end-exclusive.
DEFAULT_WRMS_UNCERTAINTY_SEGMENTS: tuple[WrmsUncertaintyEntry, ...] = (
    WrmsUncertaintyEntry("APO-a", "APOLLO", _d("2006/01/01"), _d("2010/01/01"), 0.025),
    WrmsUncertaintyEntry("APO-b", "APOLLO", _d("2010/01/01"), _d("2012/01/01"), 0.044),
    WrmsUncertaintyEntry("APO-c", "APOLLO", _d("2012/01/01"), _d("2013/01/01"), 0.040),
    WrmsUncertaintyEntry("APO-d", "APOLLO", _d("2013/01/01"), _d("2016/01/01"), 0.032),
    WrmsUncertaintyEntry("APO-e", "APOLLO", _d("2016/01/01"), _d("2019/01/01"), 0.028),
    WrmsUncertaintyEntry("APO-f", "APOLLO", _d("2019/01/01"), _d("2025/01/01"), 0.019),
    WrmsUncertaintyEntry("MLRS2", "MLRS2", _d("1988/01/01"), _d("2015/03/26"), 0.072),
    WrmsUncertaintyEntry("MLRS1-a", "MLRS1", _d("1983/01/01"), _d("1985/01/01"), 1.320),
    WrmsUncertaintyEntry("MLRS1-b", "MLRS1", _d("1985/01/01"), _d("1989/01/01"), 0.101),
    WrmsUncertaintyEntry("Haleakala", "HALEAKALA", _d("1984/01/01"), _d("1991/01/01"), 0.125),
    WrmsUncertaintyEntry("Matera", "MATERA", _d("2003/01/01"), _d("2025/01/01"), 0.069),
    WrmsUncertaintyEntry("Grasse-a", "GRASSE", _d("1984/01/01"), _d("1987/01/01"), 0.314),
    WrmsUncertaintyEntry("Grasse-b", "GRASSE", _d("1987/01/01"), _d("2006/01/01"), 0.061),
    WrmsUncertaintyEntry("Grasse-c", "GRASSE", _d("2009/01/01"), _d("2025/01/01"), 0.023),
    WrmsUncertaintyEntry("MCD2.7", "MCDONALD", _d("1969/01/01"), _d("1986/01/01"), 0.616),
    WrmsUncertaintyEntry("Wettzell", "WETTZELL", _d("2018/01/01"), _d("2025/01/01"), 0.018),
)


def _utc_date(epoch: Epoch) -> date:
    if not isinstance(epoch, Epoch):
        raise TypeError("obs_epoch_utc must be an Epoch.")
    return date.fromisoformat(epoch.date_iso())


def _candidate_list(station_values: Sequence[object] | object) -> list[object]:
    if isinstance(station_values, (str, bytes)) or not isinstance(station_values, Iterable):
        return [station_values]
    return list(station_values)


@dataclass(frozen=True, slots=True)
class WrmsUncertaintyTable:
    """Station-indexed empirical two-way WRMS uncertainty table."""

    entries: tuple[WrmsUncertaintyEntry, ...]
    units: str = "metres two-way range WRMS"
    source: str | None = None
    _by_station: dict[str, tuple[WrmsUncertaintyEntry, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        by_station: dict[str, list[WrmsUncertaintyEntry]] = {}
        for entry in entries:
            by_station.setdefault(entry.station, []).append(entry)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "units", str(self.units).strip() or "metres two-way range WRMS")
        object.__setattr__(self, "source", None if self.source is None else str(self.source).strip() or None)
        object.__setattr__(
            self,
            "_by_station",
            {station: tuple(sorted(items, key=lambda item: (item.start, item.end))) for station, items in by_station.items()},
        )

    def station_from_candidates(self, station_values: Sequence[object] | object) -> Optional[str]:
        for candidate in _candidate_list(station_values):
            station = normalize_station(candidate)
            if station is not None:
                return station
        return None

    def entry(self, station_values: Sequence[object] | object, obs_epoch_utc: Epoch) -> Optional[WrmsUncertaintyEntry]:
        station = self.station_from_candidates(station_values)
        if station is None:
            return None
        day = _utc_date(obs_epoch_utc)
        for entry in self._by_station.get(station, ()):  # first match wins
            if entry.start <= day < entry.end:
                return entry
        return None

    def coverage_summary(self) -> dict[str, list[tuple[str, str, str]]]:
        return {
            station: [(entry.group, entry.start.isoformat(), entry.end.isoformat()) for entry in entries]
            for station, entries in self._by_station.items()
        }

    @classmethod
    def from_mapping(cls, config: Mapping[str, object], *, source_file: str | Path | None = None) -> "WrmsUncertaintyTable":
        forbidden = sorted({"name", "aliases"} & set(config))
        if forbidden:
            raise ValueError(
                "WRMS uncertainty table config no longer accepts top-level "
                f"{', '.join(repr(key) for key in forbidden)}; use only 'file', 'source', and 'uncertainties'."
            )
        source = config.get("source") or (str(source_file) if source_file else None)
        raw_uncertainties = config.get("uncertainties")
        if raw_uncertainties is None:
            raise ValueError("WRMS uncertainty table config requires an 'uncertainties' list.")
        if not isinstance(raw_uncertainties, Sequence) or isinstance(raw_uncertainties, (str, bytes)):
            raise TypeError("WRMS uncertainty 'uncertainties' must be a list of rows.")
        entries = tuple(
            WrmsUncertaintyEntry.from_config_item(item, default_source=str(source) if source else None)
            for item in raw_uncertainties
        )
        return cls(entries=entries, source=None if source is None else str(source))


DEFAULT_WRMS_UNCERTAINTY_TABLE = WrmsUncertaintyTable(
    entries=DEFAULT_WRMS_UNCERTAINTY_SEGMENTS,
    source="default",
)

BUILTIN_WRMS_UNCERTAINTY_TABLES: dict[str, WrmsUncertaintyTable] = {
    "default": DEFAULT_WRMS_UNCERTAINTY_TABLE,
}


def builtin_wrms_uncertainty_table(name: object) -> WrmsUncertaintyTable:
    key = str(name).strip().lower()
    try:
        return BUILTIN_WRMS_UNCERTAINTY_TABLES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown built-in WRMS uncertainty table {name!r}. Available: {sorted(BUILTIN_WRMS_UNCERTAINTY_TABLES)}"
        ) from exc


def load_wrms_uncertainty_table(path: str | Path) -> WrmsUncertaintyTable:
    file = Path(path).expanduser()
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() in {".yml", ".yaml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"WRMS uncertainty table file must contain a mapping: {file}")
    return WrmsUncertaintyTable.from_mapping(data, source_file=file)


__all__ = [
    "BUILTIN_WRMS_UNCERTAINTY_TABLES",
    "DEFAULT_WRMS_UNCERTAINTY_SEGMENTS",
    "DEFAULT_WRMS_UNCERTAINTY_TABLE",
    "WrmsUncertaintyEntry",
    "WrmsUncertaintyTable",
    "builtin_wrms_uncertainty_table",
    "load_wrms_uncertainty_table",
]
