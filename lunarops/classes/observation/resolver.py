"""Catalog resolution for source-independent normal-point observations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from lunarops.base.epoch import Epoch, TimeScale
from lunarops.fileio.catalogs import ReflectorRecord, StationRecord, first_resolvable_key
from lunarops.fileio.normal_points import NptRecord


@dataclass(frozen=True, slots=True)
class CatalogSelection:
    station_name: str | None = None
    reflector_name: str | None = None


@dataclass(frozen=True, slots=True, eq=False, repr=False)
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


class ObservationModelState:
    """Mutable catalog state shared explicitly by model and parametrizations."""

    __slots__ = ("station_catalog", "reflector_catalog")

    def __init__(
        self,
        station_catalog: Mapping[str, StationRecord],
        reflector_catalog: Mapping[str, ReflectorRecord],
    ) -> None:
        self.station_catalog = dict(station_catalog)
        self.reflector_catalog = dict(reflector_catalog)

    def reflector_positions(self) -> dict[str, tuple[float, float, float]]:
        return {
            key: tuple(float(value) for value in record.moon_fixed_xyz_m)
            for key, record in self.reflector_catalog.items()
        }

    def apply_reflector_positions(
        self,
        positions: Mapping[str, Sequence[float]],
    ) -> None:
        unknown = set(positions) - set(self.reflector_catalog)
        if unknown:
            raise KeyError(f"Unknown reflector state key(s): {sorted(unknown)}")
        for key, values in positions.items():
            self.reflector_catalog[key] = replace(
                self.reflector_catalog[key],
                moon_fixed_xyz_m=values,
            )


class ObservationResolver:
    def __init__(
        self,
        model_state: ObservationModelState,
    ) -> None:
        if not isinstance(model_state, ObservationModelState):
            raise TypeError("model_state must be an ObservationModelState.")
        self.model_state = model_state

    @property
    def station_catalog(self) -> dict[str, StationRecord]:
        return self.model_state.station_catalog

    @property
    def reflector_catalog(self) -> dict[str, ReflectorRecord]:
        return self.model_state.reflector_catalog

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


__all__ = [
    "CatalogSelection",
    "ObservationModelState",
    "ObservationResolver",
    "ResolvedObservation",
]
