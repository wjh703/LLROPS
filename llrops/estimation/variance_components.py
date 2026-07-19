"""Variance-component definitions and observation assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence

from llrops.classes.observation.equations import ObservationEquation

ObsKey = Hashable


def _normalise(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")


def _as_texts(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_normalise(value),)
    return tuple(_normalise(item) for item in value if _normalise(item))


def _metadata_candidates(eq: ObservationEquation, *keys: str) -> set[str]:
    metadata = eq.metadata or {}
    values = [eq.station_key]
    values.extend(metadata.get(key) for key in keys)
    return {_normalise(value) for value in values if _normalise(value)}


@dataclass(frozen=True)
class VarianceComponentDefinition:
    id: str
    station_system: str
    start: str
    end_exclusive: Optional[str]
    station_aliases: tuple[str, ...]
    system_aliases: tuple[str, ...]
    wavelength_min_nm: Optional[float] = None
    wavelength_max_nm: Optional[float] = None

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "VarianceComponentDefinition":
        component_id = str(value.get("id") or "").strip()
        station_system = str(value.get("station_system") or "").strip()
        start = str(value.get("start") or "").strip()
        if not component_id or not station_system or not start:
            raise ValueError("Each variance component requires id, station_system, and start.")
        end = value.get("end_exclusive")
        return cls(
            id=component_id,
            station_system=station_system,
            start=start[:10],
            end_exclusive=None if end in (None, "", "present") else str(end)[:10],
            station_aliases=_as_texts(value.get("station_aliases")) or (_normalise(station_system),),
            system_aliases=_as_texts(value.get("system_aliases")),
            wavelength_min_nm=None if value.get("wavelength_min_nm") is None else float(value["wavelength_min_nm"]),
            wavelength_max_nm=None if value.get("wavelength_max_nm") is None else float(value["wavelength_max_nm"]),
        )

    def matches(self, equation: ObservationEquation) -> bool:
        date = equation.epoch.date_iso()
        if date < self.start or (self.end_exclusive is not None and date >= self.end_exclusive):
            return False
        stations = _metadata_candidates(equation, "station_catalog_key", "station_name", "station_full_name", "station_id")
        if not stations.intersection(self.station_aliases):
            return False
        if self.system_aliases:
            systems = _metadata_candidates(equation, "system_config_id", "system_name", "station_system", "observation_mode")
            if not systems.intersection(self.system_aliases):
                return False
        wavelength = (equation.metadata or {}).get("wavelength_nm")
        if self.wavelength_min_nm is not None or self.wavelength_max_nm is not None:
            if wavelength is None:
                return False
            wavelength = float(wavelength)
            if self.wavelength_min_nm is not None and wavelength < self.wavelength_min_nm:
                return False
            if self.wavelength_max_nm is not None and wavelength > self.wavelength_max_nm:
                return False
        return True


def assign_variance_components(equations: Sequence[ObservationEquation], components: Sequence[VarianceComponentDefinition]) -> dict[ObsKey, str]:
    if not components:
        raise ValueError("At least one variance component is required.")
    assignments: dict[ObsKey, str] = {}
    for equation in equations:
        matches = [component.id for component in components if component.matches(equation)]
        if len(matches) != 1:
            detail = "no matching component" if not matches else f"multiple matching components {matches!r}"
            raise ValueError(f"Observation {equation.identity!r} at {equation.epoch.date_iso()} has {detail}.")
        assignments[equation.identity] = matches[0]
    return assignments


__all__ = ["VarianceComponentDefinition", "assign_variance_components"]
