"""Observation-level range-bias models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from llrops.base.constants import C
from llrops.base.epoch import Epoch
from llrops.classes.range_bias.table import RangeBiasTable


@dataclass(frozen=True, slots=True)
class RangeBiasCorrection:
    model: str
    two_way_cm: float

    def __post_init__(self) -> None:
        model = str(self.model).strip()
        value = float(self.two_way_cm)
        if not model:
            raise ValueError("Range-bias model name must not be empty.")
        if not np.isfinite(value):
            raise ValueError("two_way_cm must be finite.")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "two_way_cm", value)

    @property
    def two_way_m(self) -> float:
        return 0.01 * self.two_way_cm

    @property
    def two_way_s(self) -> float:
        return self.two_way_m / C

    @property
    def one_way_m(self) -> float:
        return 0.5 * self.two_way_m


class RangeBiasModel(ABC):
    @abstractmethod
    def correction(
        self,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> RangeBiasCorrection:
        """Return the correction applied to the calculated two-way observable."""


class ZeroRangeBiasModel(RangeBiasModel):
    def __init__(self, name: str = "none") -> None:
        self.name = name

    def correction(
        self,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> RangeBiasCorrection:
        return RangeBiasCorrection(self.name, 0.0)


class TableRangeBiasModel(RangeBiasModel):
    def __init__(self, table: RangeBiasTable) -> None:
        if not isinstance(table, RangeBiasTable):
            raise TypeError("table must be a RangeBiasTable.")
        self.table = table

    @property
    def model_label(self) -> str:
        return self.table.source or "range-bias table"

    def correction(
        self,
        station_candidates: Sequence[str],
        epoch_utc: Epoch,
    ) -> RangeBiasCorrection:
        return RangeBiasCorrection(
            self.model_label,
            self.table.two_way_cm(list(station_candidates), epoch_utc),
        )


__all__ = [
    "RangeBiasCorrection",
    "RangeBiasModel",
    "TableRangeBiasModel",
    "ZeroRangeBiasModel",
]
