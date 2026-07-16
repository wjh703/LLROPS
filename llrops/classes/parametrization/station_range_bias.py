"""Parametrization: additive one-way station range-bias parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from llrops.base.parameter_name import ParameterName
from llrops.base.epoch import Epoch, TimeScale
from llrops.base.validation import parameter_vector
from llrops.config.registry import register
from llrops.classes.range_bias.table import normalize_station
from llrops.classes.observation.equations import ObservationEquation
from .base import Parametrization


_STATION_ALIASES = {"station", "stations", "perstation", "constant"}
_INTERVAL_ALIASES = {
    "stationinterval",
    "station+interval",
    "station_interval",
    "station-interval",
    "interval",
    "period",
}


def _normalize_per(value: object) -> str:
    text = str(value or "station").strip()
    compact = text.replace("_", "").replace("-", "").replace(" ", "").lower()
    if text.lower() in _INTERVAL_ALIASES or compact in _INTERVAL_ALIASES:
        return "station+interval"
    if text.lower() in _STATION_ALIASES or compact in _STATION_ALIASES:
        return "station"
    raise ValueError(
        "Unsupported stationRangeBias per={!r}; expected 'station' or 'station+interval'.".format(value)
    )


def _as_nonempty_strings(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _canonical_station_or_text(value: object) -> str:
    text = str(value).strip()
    return normalize_station(text) or text.upper()


@dataclass(frozen=True, slots=True)
class StationBiasInterval:
    station: str
    start: str
    end_exclusive: Optional[str]
    name: Optional[str] = None

    @property
    def key(self) -> str:
        end = self.end_exclusive or "present"
        return str(self.name or f"{self.station}_{self.start}_{end}")

    def active_at(self, epoch: Epoch) -> bool:
        if not isinstance(epoch, Epoch):
            raise TypeError("epoch must be an Epoch.")
        epoch.require_scale(TimeScale.UTC, name="epoch")
        date_text = epoch.date_iso()
        start = str(self.start).replace("/", "-")[:10]
        end = None if self.end_exclusive is None else str(self.end_exclusive).replace("/", "-")[:10]
        return start <= date_text and (end is None or date_text < end)


def _parse_interval_value(station: str, value: object) -> StationBiasInterval:
    if isinstance(value, str):
        if "/" not in value:
            raise ValueError(f"stationRangeBias interval string must be 'start/end', got {value!r}")
        start, end = [part.strip() for part in value.split("/", 1)]
        return StationBiasInterval(station=station, start=start, end_exclusive=end)
    if isinstance(value, Mapping):
        start = value.get("start") or value.get("from") or value.get("begin")
        has_end_exclusive = "end_exclusive" in value
        if has_end_exclusive:
            end = value.get("end_exclusive")
        else:
            end = value.get("end") or value.get("to") or value.get("until")
        if not start or (not has_end_exclusive and not end):
            raise ValueError(f"stationRangeBias interval mapping needs start/end, got {value!r}")
        return StationBiasInterval(
            station=station,
            start=str(start),
            end_exclusive=None if end is None else str(end),
            name=None if value.get("name") is None else str(value.get("name")),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return StationBiasInterval(station=station, start=str(value[0]), end_exclusive=str(value[1]))
    raise ValueError(f"Unsupported stationRangeBias interval item: {value!r}")


def parse_station_bias_intervals(config_value: object) -> List[StationBiasInterval]:
    """Parse explicit station×interval bias definitions from config.

    Accepted forms::

        intervals:
          APOLLO:
            - 2006-04-07/2010-11-01
            - {start: 2010-11-01, end: 2014-01-01}

        intervals:
          - {station: APOLLO, start: 2006-04-07, end: 2010-11-01}
    """
    if not config_value:
        return []
    intervals: List[StationBiasInterval] = []
    if isinstance(config_value, Mapping):
        for station_raw, items in config_value.items():
            station = _canonical_station_or_text(station_raw)
            if isinstance(items, (str, Mapping)):
                items = [items]
            for item in items:
                intervals.append(_parse_interval_value(station, item))
        return intervals
    if isinstance(config_value, Sequence) and not isinstance(config_value, (str, bytes)):
        for item in config_value:
            if not isinstance(item, Mapping) or "station" not in item:
                raise ValueError(
                    "List-form stationRangeBias intervals must be mappings with station/start/end."
                )
            station = _canonical_station_or_text(item["station"])
            intervals.append(_parse_interval_value(station, item))
        return intervals
    raise ValueError(f"Unsupported stationRangeBias intervals config: {config_value!r}")


def station_candidates_for_equation(eq: ObservationEquation) -> List[str]:
    metadata = eq.metadata or {}
    return _as_nonempty_strings(
        [
            eq.station_key,
            metadata.get("station_catalog_key"),
            metadata.get("station_name"),
            metadata.get("station_full_name"),
            metadata.get("station_id"),
        ]
    )


def canonical_station_for_equation(eq: ObservationEquation) -> Optional[str]:
    for candidate in station_candidates_for_equation(eq):
        canonical = normalize_station(candidate)
        if canonical is not None:
            return canonical
    if eq.station_key:
        return str(eq.station_key).upper()
    return None


def active_station_bias_interval_keys(
    intervals: Sequence[StationBiasInterval],
    eq: ObservationEquation,
    *,
    requested_canonical: Optional[set[str]] = None,
) -> Tuple[str, ...]:
    canonical = canonical_station_for_equation(eq)
    if canonical is None:
        return ()
    out: List[str] = []
    for interval in intervals:
        if interval.station != canonical:
            continue
        if requested_canonical and interval.station not in requested_canonical:
            continue
        if interval.active_at(eq.epoch):
            out.append(interval.key)
    return tuple(out)


@register("parametrization", "stationRangeBias")
class StationRangeBiasParametrization(Parametrization):
    """Estimate additive one-way station range biases.

    ``per='station'`` estimates one constant per station.  ``per='station+interval'``
    now uses explicit intervals from the parameter-estimation config; it does
    not use INPOP21a bias values or the forward-model ``rangeBias``.
    """

    def __init__(
        self,
        *,
        stations: Optional[Sequence[str]] = None,
        per: str = "station",
        intervals: Optional[object] = None,
    ) -> None:
        self.per = _normalize_per(per)
        self.intervals = parse_station_bias_intervals(intervals)
        self.requested = list(stations) if stations else None
        self._requested_exact = {str(s).strip() for s in self.requested or [] if str(s).strip()}
        self._requested_upper = {s.upper() for s in self._requested_exact}
        self._requested_canonical = {
            canonical
            for canonical in (normalize_station(s) for s in self._requested_exact)
            if canonical is not None
        }
        self.keys: List[str] = []
        self._index_by_key: Dict[str, int] = {}
        self._names: List[ParameterName] = []
        self.values: Dict[str, float] = {}

    @classmethod
    def from_config(cls, config: dict, context) -> "StationRangeBiasParametrization":
        return cls(
            stations=config.get("stations"),
            per=config.get("per", "station"),
            intervals=config.get("intervals") or config.get("periods"),
        )

    def _station_allowed(self, eq: ObservationEquation) -> bool:
        if self.requested is None:
            return True
        candidates = station_candidates_for_equation(eq)
        if any(c in self._requested_exact or c.upper() in self._requested_upper for c in candidates):
            return True
        canonical = canonical_station_for_equation(eq)
        return canonical is not None and canonical in self._requested_canonical

    def _interval_keys_for(self, eq: ObservationEquation) -> Tuple[str, ...]:
        if not self._station_allowed(eq):
            return ()
        return active_station_bias_interval_keys(
            self.intervals,
            eq,
            requested_canonical=self._requested_canonical or None,
        )

    def _station_key_for(self, eq: ObservationEquation) -> Optional[str]:
        if not self._station_allowed(eq):
            return None
        return canonical_station_for_equation(eq) or eq.station_key

    def _active_keys_for(self, eq: ObservationEquation) -> Tuple[str, ...]:
        if self.per == "station":
            key = self._station_key_for(eq)
            return (key,) if key else ()
        return self._interval_keys_for(eq)

    def setup(self, equations: Sequence[ObservationEquation], context) -> None:
        if self.per == "station":
            observed = sorted(
                {
                    key
                    for eq in equations
                    for key in [self._station_key_for(eq)]
                    if key is not None
                }
            )
            self.keys = sorted(self._requested_canonical) if self.requested is not None and self._requested_canonical else list(self.requested or observed)
        else:
            if not self.intervals:
                raise ValueError(
                    "stationRangeBias per='station+interval' requires explicit intervals/periods in the parametrization config."
                )
            keys = sorted({key for eq in equations for key in self._interval_keys_for(eq)})
            self.keys = keys
        for key in self.keys:
            self.values.setdefault(key, 0.0)
        self._index_by_key = {key: index for index, key in enumerate(self.keys)}
        self._names = self._build_parameter_names()

    def _build_parameter_names(self) -> List[ParameterName]:
        if self.per == "station":
            return [ParameterName(key, "rangeBias") for key in self.keys]

        interval_by_key = {interval.key: interval for interval in self.intervals}
        names: List[ParameterName] = []
        for key in self.keys:
            interval = interval_by_key.get(key)
            if interval is not None:
                end = interval.end_exclusive or "present"
                names.append(ParameterName(interval.station, "rangeBias", "interval", f"{interval.start}/{end}"))
            else:
                names.append(ParameterName(key, "rangeBias", "interval"))
        return names

    def parameter_names(self) -> List[ParameterName]:
        return list(self._names)

    def design_columns(self, eq: ObservationEquation) -> np.ndarray:
        cols = np.zeros(len(self.keys), dtype=float)
        for index, coeff in self.design_entries(eq):
            cols[index] += coeff
        return cols

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        coeff = float(np.asarray(eq.partials.get("station_range_bias", [1.0]), dtype=float).ravel()[0])
        if not coeff:
            return []
        entries: list[tuple[int, float]] = []
        for key in self._active_keys_for(eq):
            index = self._index_by_key.get(key)
            if index is not None:
                entries.append((index, coeff))
        return entries

    def reduce_observation(self, eq: ObservationEquation) -> float:
        return float(sum(self.values.get(key, 0.0) for key in self._active_keys_for(eq)))

    def apply_update(self, delta: np.ndarray) -> None:
        values = parameter_vector(delta, expected_size=len(self.keys), name="stationRangeBias update")
        for key, d in zip(self.keys, values):
            self.values[key] = self.values.get(key, 0.0) + float(d)

    def state(self) -> Dict[str, object]:
        return {
            "per": self.per,
            "intervals": [
                {
                    "station": interval.station,
                    "start": interval.start,
                    "end_exclusive": interval.end_exclusive,
                    "name": interval.name,
                }
                for interval in self.intervals
            ],
            "values": {key: float(self.values.get(key, 0.0)) for key in self.keys},
        }
