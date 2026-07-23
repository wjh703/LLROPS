"""Canonical source-independent LLR normal-point records."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, List, Optional, Sequence

from llrops.base.constants import C
from llrops.base.epoch import Epoch, TimeScale


@dataclass
class NptRecord:
    station_name: str
    reflector_name: str
    transmit_epoch: Epoch
    round_trip_time_s: float
    uncertainty_two_way_s: float
    pressure_hpa: float
    temperature_k: float
    humidity_percent: float
    wavelength_nm: float
    index: int = 0
    station_code: Optional[str] = None
    reflector_code: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.transmit_epoch, Epoch):
            raise TypeError("transmit_epoch must be an Epoch.")
        self.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")
        self.station_name = str(self.station_name).strip()
        self.reflector_name = str(self.reflector_name).strip()
        if not self.station_name or not self.reflector_name:
            raise ValueError("station_name and reflector_name must not be empty.")
        positive_fields = (
            "round_trip_time_s",
            "uncertainty_two_way_s",
            "pressure_hpa",
            "temperature_k",
            "wavelength_nm",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
            setattr(self, name, value)
        humidity = float(self.humidity_percent)
        if not math.isfinite(humidity) or not 0.0 <= humidity <= 100.0:
            raise ValueError("humidity_percent must be finite and in [0, 100].")
        self.humidity_percent = humidity
        self.index = int(self.index)
        self.station_code = _optional_text(self.station_code)
        self.reflector_code = _optional_text(self.reflector_code)

    @property
    def observed_round_trip_time_s(self) -> float:
        return float(self.round_trip_time_s)

    @property
    def observed_range_m(self) -> float:
        return 0.5 * C * self.observed_round_trip_time_s

    @property
    def uncertainty_two_way_ps(self) -> float:
        return float(self.uncertainty_two_way_s) * 1.0e12

    @property
    def range_uncertainty_one_way_m(self) -> float:
        return 0.5 * C * float(self.uncertainty_two_way_s)

    @property
    def temperature_c(self) -> float:
        return float(self.temperature_k) - 273.15

    @property
    def wavelength_um(self) -> float:
        return float(self.wavelength_nm) / 1000.0


@dataclass
class NptDataset:
    records: List[NptRecord]
    name: Optional[str] = None
    n_input_records: int = 0
    n_invalid_records: int = 0

    @property
    def n_valid_records(self) -> int:
        return len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[NptRecord]:
        return iter(self.records)

    def assign_indices(self, *, start: int = 0) -> "NptDataset":
        for offset, rec in enumerate(self.records):
            rec.index = int(start) + offset
        return self

    def filter_time(self, start_time_utc=None, end_time_utc=None) -> "NptDataset":
        start = parse_time_filter(start_time_utc)
        end = parse_time_filter(end_time_utc)
        if start is None and end is None:
            return self

        kept: List[NptRecord] = []
        for rec in self.records:
            epoch = rec.transmit_epoch
            if start is not None and epoch < start:
                continue
            if end is not None and epoch >= end:
                continue
            kept.append(rec)
        self.records = kept
        self.assign_indices(start=0)
        return self


def _optional_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_time_filter(value):
    """Parse an optional lower/upper UTC filter into an Epoch."""
    if value is None:
        return None
    if isinstance(value, Epoch):
        return value.require_scale(TimeScale.UTC, name="time filter")
    text = str(value).strip()
    if not text:
        return None
    return Epoch.from_isot(text, scale=TimeScale.UTC)


def npt_record_from_mini(record) -> NptRecord:
    return NptRecord(
        station_name=record.station_name,
        reflector_name=record.reflector_name,
        transmit_epoch=record.launch_epoch,
        round_trip_time_s=record.observed_round_trip_time_s,
        uncertainty_two_way_s=record.uncertainty_two_way_s,
        pressure_hpa=record.pressure_hpa,
        temperature_k=record.temperature_k,
        humidity_percent=float(record.humidity_percent),
        wavelength_nm=record.wavelength_nm,
        index=int(record.index),
        station_code=record.station_id,
        reflector_code=str(record.reflector_id),
    )


def npt_records_from_mini(
    records: Sequence[object],
) -> List[NptRecord]:
    out = [npt_record_from_mini(record) for record in records]
    for index, record in enumerate(out):
        record.index = index
    return out


def combine_npt_datasets(
    datasets: Sequence[NptDataset],
    *,
    name: Optional[str] = None,
) -> NptDataset:
    merged: List[NptRecord] = []
    n_input_records = 0
    n_invalid_records = 0
    for dataset in datasets:
        n_input_records += int(dataset.n_input_records)
        n_invalid_records += int(dataset.n_invalid_records)
        for record in dataset.records:
            record.index = len(merged)
            merged.append(record)
    return NptDataset(
        records=merged,
        name=name,
        n_input_records=n_input_records,
        n_invalid_records=n_invalid_records,
    )


__all__ = [
    "NptDataset",
    "NptRecord",
    "combine_npt_datasets",
    "npt_record_from_mini",
    "npt_records_from_mini",
    "parse_time_filter",
]
