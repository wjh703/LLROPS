"""Catalog resolution for source-independent normal-point observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from llrops.base.epoch import Epoch, TimeScale
from llrops.fileio.catalogs import ReflectorRecord, StationRecord, first_resolvable_key
from llrops.fileio.npt import NptRecord


@dataclass(frozen=True, slots=True)
class CatalogSelection:
    station_name: str | None = None
    reflector_name: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedObservation:
    record: NptRecord
    station_key: str
    station: StationRecord
    reflector_key: str
    reflector: ReflectorRecord
    transmit_epoch: Epoch

    def __post_init__(self) -> None:
        if not isinstance(self.transmit_epoch, Epoch):
            raise TypeError("transmit_epoch must be an Epoch.")
        self.transmit_epoch.require_scale(TimeScale.UTC, name="transmit_epoch")

    @property
    def station_candidates(self) -> tuple[str, ...]:
        values = (
            self.station_key,
            self.station.name,
            self.record.station_name,
            self.record.station_code,
        )
        return tuple(str(value) for value in values if value is not None and str(value).strip())


class ObservationResolver:
    def __init__(
        self,
        station_catalog: Mapping[str, StationRecord],
        reflector_catalog: Mapping[str, ReflectorRecord],
    ) -> None:
        self.station_catalog = dict(station_catalog)
        self.reflector_catalog = dict(reflector_catalog)

    def replace_catalogs(
        self,
        *,
        station_catalog: Mapping[str, StationRecord] | None = None,
        reflector_catalog: Mapping[str, ReflectorRecord] | None = None,
    ) -> None:
        if station_catalog is not None:
            self.station_catalog = dict(station_catalog)
        if reflector_catalog is not None:
            self.reflector_catalog = dict(reflector_catalog)

    @staticmethod
    def _candidates(
        record: NptRecord,
        selection: CatalogSelection,
    ) -> tuple[list[str | None], list[str | None]]:
        station_candidates = (
            [selection.station_name]
            if selection.station_name
            else [record.station_name, record.station_code]
        )
        reflector_candidates = (
            [selection.reflector_name]
            if selection.reflector_name
            else [record.reflector_name, record.reflector_code]
        )
        return station_candidates, reflector_candidates

    def resolve(
        self,
        record: NptRecord,
        selection: CatalogSelection = CatalogSelection(),
    ) -> ResolvedObservation:
        station_candidates, reflector_candidates = self._candidates(record, selection)
        station_key = first_resolvable_key(station_candidates, self.station_catalog, "Station")
        reflector_key = first_resolvable_key(
            reflector_candidates,
            self.reflector_catalog,
            "Reflector",
        )
        return ResolvedObservation(
            record=record,
            station_key=station_key,
            station=self.station_catalog[station_key],
            reflector_key=reflector_key,
            reflector=self.reflector_catalog[reflector_key],
            transmit_epoch=record.transmit_epoch,
        )

    def validate(
        self,
        records: Sequence[NptRecord],
        selection: CatalogSelection = CatalogSelection(),
    ) -> list[ResolvedObservation]:
        resolved: list[ResolvedObservation] = []
        problems: list[str] = []
        for position, record in enumerate(records):
            try:
                resolved.append(self.resolve(record, selection))
            except KeyError as exc:
                problems.append(f"record_index={position}: {exc}")
        if problems:
            detail = "\n  ".join(problems)
            raise ValueError(
                f"Catalog resolution failed for {len(problems)} record(s):\n  {detail}"
            )
        return resolved


__all__ = ["CatalogSelection", "ObservationResolver", "ResolvedObservation"]
