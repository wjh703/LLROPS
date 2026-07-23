"""Observation-level uncertainty models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from llrops.base.constants import C
from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.uncertainty.wrms_table import WrmsUncertaintyTable


class UncertaintyKind(str, Enum):
    WRMS_TABLE = "wrms-table"
    MINI = "mini"

    @classmethod
    def parse(cls, value: object) -> "UncertaintyKind":
        if isinstance(value, cls):
            return value
        raw = str(value or cls.WRMS_TABLE.value).strip().lower()
        try:
            return cls(raw)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown uncertainty model {value!r}; expected one of: {allowed}."
            ) from exc


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    kind: UncertaintyKind
    source: str
    group: str | None
    sigma_one_way_m: float
    uncertainty_two_way_s: float
    uncertainty_two_way_ps: float
    uncertainty_raw: float | None
    wrms_two_way_m: float | None = None

    def __post_init__(self) -> None:
        kind = UncertaintyKind.parse(self.kind)
        source = str(self.source).strip()
        sigma = float(self.sigma_one_way_m)
        two_way_s = float(self.uncertainty_two_way_s)
        two_way_ps = float(self.uncertainty_two_way_ps)
        if not source:
            raise ValueError("Uncertainty source must not be empty.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_one_way_m must be positive and finite.")
        if not np.isfinite(two_way_s) or two_way_s <= 0.0:
            raise ValueError("uncertainty_two_way_s must be positive and finite.")
        if not np.isfinite(two_way_ps) or two_way_ps <= 0.0:
            raise ValueError("uncertainty_two_way_ps must be positive and finite.")
        raw = None if self.uncertainty_raw is None else float(self.uncertainty_raw)
        wrms = None if self.wrms_two_way_m is None else float(self.wrms_two_way_m)
        if raw is not None and not np.isfinite(raw):
            raise ValueError("uncertainty_raw must be finite when supplied.")
        if wrms is not None and (not np.isfinite(wrms) or wrms <= 0.0):
            raise ValueError("wrms_two_way_m must be positive and finite when supplied.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "group",
            None if self.group is None else str(self.group),
        )
        object.__setattr__(self, "sigma_one_way_m", sigma)
        object.__setattr__(self, "uncertainty_two_way_s", two_way_s)
        object.__setattr__(self, "uncertainty_two_way_ps", two_way_ps)
        object.__setattr__(self, "uncertainty_raw", raw)
        object.__setattr__(self, "wrms_two_way_m", wrms)


class UncertaintyModel(ABC):
    kind: UncertaintyKind

    @abstractmethod
    def estimate(
        self,
        *,
        record,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> UncertaintyEstimate:
        raise NotImplementedError

    def validate(
        self,
        *,
        record,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> None:
        self.estimate(
            record=record,
            station_candidates=station_candidates,
            epoch_utc=epoch_utc,
        )


class MiniUncertainty(UncertaintyModel):
    kind = UncertaintyKind.MINI

    def estimate(
        self,
        *,
        record,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> UncertaintyEstimate:
        two_way_s = float(record.uncertainty_two_way_s)
        return UncertaintyEstimate(
            kind=self.kind,
            source="mini_uncertainty_two_way",
            group="MINI",
            sigma_one_way_m=0.5 * C * two_way_s,
            uncertainty_two_way_s=two_way_s,
            uncertainty_two_way_ps=two_way_s * 1.0e12,
            uncertainty_raw=None,
            wrms_two_way_m=None,
        )


class WrmsTableUncertainty(UncertaintyModel):
    kind = UncertaintyKind.WRMS_TABLE

    def __init__(self, table: WrmsUncertaintyTable) -> None:
        if not isinstance(table, WrmsUncertaintyTable):
            raise TypeError("table must be a WrmsUncertaintyTable.")
        self.table = table

    def estimate(
        self,
        *,
        record,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> UncertaintyEstimate:
        entry = self.table.entry(list(station_candidates), epoch_utc)
        if entry is None:
            raise ValueError(
                f"WRMS uncertainty table {self.table.source or 'wrms-table'!r} "
                f"has no match for station_candidates={list(station_candidates)!r}; "
                f"obs_time_utc={epoch_utc.isot(scale=TimeScale.UTC)}"
            )
        return UncertaintyEstimate(
            kind=self.kind,
            source=self.table.source or "wrms-table",
            group=entry.group,
            sigma_one_way_m=float(entry.wrms_one_way_m),
            uncertainty_two_way_s=float(entry.uncertainty_two_way_s),
            uncertainty_two_way_ps=float(entry.uncertainty_two_way_ps),
            uncertainty_raw=float(entry.uncertainty_raw_0p1ps),
            wrms_two_way_m=float(entry.wrms_two_way_m),
        )


__all__ = [
    "MiniUncertainty",
    "UncertaintyEstimate",
    "UncertaintyKind",
    "UncertaintyModel",
    "WrmsTableUncertainty",
]
