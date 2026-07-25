"""Additive one-way station range-bias parametrization."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from llrops.base.array_validation import parameter_vector
from llrops.base.epoch import Epoch, TimeScale
from llrops.base.parameter_name import ParameterName
from llrops.base.station_identity import canonical_station_id
from llrops.classes.observation.equations import ObservationEquation
from llrops.config.registry import register

from .base import Parametrization


_MODES = {"station", "station+interval"}


def _iso_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        raise TypeError(f"{field} must be a date, not a datetime.")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field} must be an ISO date string.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be written as YYYY-MM-DD.") from exc


@dataclass(frozen=True, slots=True)
class StationBiasInterval:
    station: str
    start: date
    end_exclusive: Optional[date]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.station, str):
            raise TypeError("stationRangeBias interval station must be a string.")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("stationRangeBias interval name must be a string or null.")
        station = canonical_station_id(self.station)
        start = _iso_date(self.start, "stationRangeBias interval start")
        end = (
            None
            if self.end_exclusive is None
            else _iso_date(
                self.end_exclusive,
                "stationRangeBias interval end_exclusive",
            )
        )
        if end is not None and end <= start:
            raise ValueError("stationRangeBias interval end_exclusive must be after start.")
        name = None if self.name is None else self.name.strip()
        if self.name is not None and not name:
            raise ValueError("stationRangeBias interval name must not be empty.")
        object.__setattr__(self, "station", station)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "name", name)

    @property
    def key(self) -> str:
        end = self.end_exclusive.isoformat() if self.end_exclusive else "present"
        return self.name or f"{self.station}_{self.start.isoformat()}_{end}"

    def active_at(self, epoch: Epoch) -> bool:
        if not isinstance(epoch, Epoch):
            raise TypeError("epoch must be an Epoch.")
        epoch.require_scale(TimeScale.UTC, name="epoch")
        epoch_date = date.fromisoformat(epoch.date_iso())
        return self.start <= epoch_date and (
            self.end_exclusive is None or epoch_date < self.end_exclusive
        )


def parse_station_bias_intervals(config_value: object) -> List[StationBiasInterval]:
    """Parse the canonical list-of-mappings interval schema."""
    if config_value is None:
        return []
    if not isinstance(config_value, list):
        raise TypeError("stationRangeBias intervals must be a list of mappings.")
    intervals: List[StationBiasInterval] = []
    for index, item in enumerate(config_value):
        if not isinstance(item, Mapping):
            raise TypeError(f"stationRangeBias intervals[{index}] must be a mapping.")
        allowed = {"station", "start", "end_exclusive", "name"}
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(
                f"stationRangeBias intervals[{index}] has unknown key(s) "
                f"{sorted(unknown)}."
            )
        missing = {"station", "start", "end_exclusive"} - set(item)
        if missing:
            raise ValueError(
                f"stationRangeBias intervals[{index}] is missing key(s) "
                f"{sorted(missing)}."
            )
        station = item["station"]
        if not isinstance(station, str):
            raise TypeError(
                f"stationRangeBias intervals[{index}].station must be a string."
            )
        intervals.append(
            StationBiasInterval(
                station=station,
                start=item["start"],
                end_exclusive=item["end_exclusive"],
                name=item.get("name"),
            )
        )
    keys = [interval.key for interval in intervals]
    if len(set(keys)) != len(keys):
        raise ValueError("stationRangeBias interval keys must be unique.")
    return intervals


def canonical_station_for_equation(eq: ObservationEquation) -> str:
    return canonical_station_id(eq.station_key)


def active_station_bias_interval_keys(
    intervals: Sequence[StationBiasInterval],
    eq: ObservationEquation,
    *,
    requested: Optional[set[str]] = None,
) -> Tuple[str, ...]:
    station = canonical_station_for_equation(eq)
    if requested is not None and station not in requested:
        return ()
    return tuple(
        interval.key
        for interval in intervals
        if interval.station == station and interval.active_at(eq.epoch)
    )


@register("parametrization", "stationRangeBias")
class StationRangeBiasParametrization(Parametrization):
    """Estimate one-way station biases by station or explicit interval."""

    def __init__(
        self,
        *,
        stations: Optional[Sequence[str]] = None,
        per: str = "station",
        intervals: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> None:
        if per not in _MODES:
            raise ValueError(
                f"stationRangeBias per must be one of {sorted(_MODES)}, got {per!r}."
            )
        if isinstance(stations, (str, bytes)):
            raise TypeError("stationRangeBias stations must be a list.")
        if stations is not None and not isinstance(stations, list):
            raise TypeError("stationRangeBias stations must be a list.")
        if stations is not None and any(
            not isinstance(station, str) for station in stations
        ):
            raise TypeError("stationRangeBias stations must contain only strings.")
        self.per = per
        self.intervals = parse_station_bias_intervals(intervals)
        self.requested = (
            None
            if stations is None
            else [canonical_station_id(station) for station in stations]
        )
        if self.requested is not None and len(set(self.requested)) != len(self.requested):
            raise ValueError("stationRangeBias stations must be unique.")
        self.keys: List[str] = []
        self._index_by_key: Dict[str, int] = {}
        self._names: List[ParameterName] = []
        self.values: Dict[str, float] = {}

    @classmethod
    def from_config(cls, config: dict, context) -> "StationRangeBiasParametrization":
        unknown = set(config) - {"type", "stations", "per", "intervals"}
        if unknown:
            raise ValueError(
                f"stationRangeBias has unknown key(s) {sorted(unknown)}."
            )
        return cls(
            stations=config.get("stations"),
            per=config.get("per", "station"),
            intervals=config.get("intervals"),
        )

    def _station_key_for(self, eq: ObservationEquation) -> Optional[str]:
        station = canonical_station_for_equation(eq)
        if self.requested is not None and station not in self.requested:
            return None
        return station

    def _active_keys_for(self, eq: ObservationEquation) -> Tuple[str, ...]:
        if self.per == "station":
            key = self._station_key_for(eq)
            return () if key is None else (key,)
        return active_station_bias_interval_keys(
            self.intervals,
            eq,
            requested=None if self.requested is None else set(self.requested),
        )

    def setup(self, equations: Sequence[ObservationEquation], model_state) -> None:
        observed_stations = {
            canonical_station_for_equation(equation) for equation in equations
        }
        if self.requested is not None:
            missing = set(self.requested) - observed_stations
            if missing:
                raise ValueError(
                    f"stationRangeBias requested station(s) have no observations: "
                    f"{sorted(missing)}."
                )
        if self.per == "station":
            self.keys = sorted(self.requested or observed_stations)
        else:
            if not self.intervals:
                raise ValueError(
                    "stationRangeBias per='station+interval' requires intervals."
                )
            self.keys = sorted(
                {
                    key
                    for equation in equations
                    for key in self._active_keys_for(equation)
                }
            )
        for key in self.keys:
            self.values.setdefault(key, 0.0)
        self._index_by_key = {key: index for index, key in enumerate(self.keys)}
        self._names = self._build_parameter_names()

    def _build_parameter_names(self) -> List[ParameterName]:
        if self.per == "station":
            return [ParameterName(key, "rangeBias") for key in self.keys]
        interval_by_key = {interval.key: interval for interval in self.intervals}
        return [
            ParameterName(
                interval_by_key[key].station,
                "rangeBias",
                "interval",
                f"{interval_by_key[key].start.isoformat()}/"
                f"{interval_by_key[key].end_exclusive.isoformat() if interval_by_key[key].end_exclusive else 'present'}",
            )
            for key in self.keys
        ]

    def parameter_names(self) -> List[ParameterName]:
        return list(self._names)

    def design_columns(self, eq: ObservationEquation) -> np.ndarray:
        columns = np.zeros(len(self.keys), dtype=float)
        for index, coefficient in self.design_entries(eq):
            columns[index] += coefficient
        return columns

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        coefficient = float(
            np.asarray(
                eq.partials.get("station_range_bias", [1.0]), dtype=float
            ).reshape(-1)[0]
        )
        if not coefficient:
            return []
        return [
            (self._index_by_key[key], coefficient)
            for key in self._active_keys_for(eq)
            if key in self._index_by_key
        ]

    def reduce_observation(self, eq: ObservationEquation) -> float:
        return float(
            sum(self.values.get(key, 0.0) for key in self._active_keys_for(eq))
        )

    def apply_update(self, delta: np.ndarray) -> None:
        updates = parameter_vector(
            delta,
            expected_size=len(self.keys),
            name="stationRangeBias update",
        )
        for key, update in zip(self.keys, updates):
            self.values[key] = self.values.get(key, 0.0) + float(update)

    def state(self) -> Dict[str, object]:
        return {
            "per": self.per,
            "intervals": [
                {
                    "station": interval.station,
                    "start": interval.start.isoformat(),
                    "end_exclusive": (
                        None
                        if interval.end_exclusive is None
                        else interval.end_exclusive.isoformat()
                    ),
                    "name": interval.name,
                }
                for interval in self.intervals
            ],
            "values": {key: float(self.values.get(key, 0.0)) for key in self.keys},
        }


__all__ = [
    "StationBiasInterval",
    "StationRangeBiasParametrization",
    "active_station_bias_interval_keys",
    "canonical_station_for_equation",
    "parse_station_bias_intervals",
]
