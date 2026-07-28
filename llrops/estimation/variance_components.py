"""Variance-component definitions and observation assignment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Hashable, Mapping, Optional, Sequence

from llrops.base.station_identity import canonical_station_id
from llrops.classes.observation.equations import ObservationEquation

ObsKey = Hashable


_COMPONENT_CONFIG_KEYS = {
    "endExclusive",
    "id",
    "start",
    "station",
    "wavelengthMaxExclusiveNm",
    "wavelengthMinNm",
}


def _date_text(value: object, field: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Variance component {field} must be YYYY-MM-DD.") from exc


def _optional_wavelength(value: object, field: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Variance component {field} must be a number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Variance component {field} must be finite and positive.")
    return result


@dataclass(frozen=True)
class VarianceComponentDefinition:
    id: str
    station: str
    start: str
    end_exclusive: Optional[str]
    wavelength_min_nm: Optional[float] = None
    wavelength_max_exclusive_nm: Optional[float] = None

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "VarianceComponentDefinition":
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError("Variance-component keys must be strings.")
        unknown = set(value) - _COMPONENT_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Variance component: unknown key(s) {sorted(unknown)}."
            )
        component_id_value = value.get("id")
        station_value = value.get("station")
        if not isinstance(component_id_value, str):
            raise TypeError("Variance component id must be a string.")
        if not isinstance(station_value, str):
            raise TypeError("Variance component station must be a string.")
        component_id = component_id_value.strip()
        station = canonical_station_id(station_value)
        start_value = value.get("start")
        start = "" if start_value is None else _date_text(start_value, "start")
        if not component_id or not start:
            raise ValueError(
                "Each variance component requires id, station, and start."
            )
        end_value = value.get("endExclusive")
        end = None if end_value is None else _date_text(end_value, "endExclusive")
        if end is not None and end <= start:
            raise ValueError(
                f"Variance component {component_id!r} endExclusive must be after start."
            )
        wavelength_min = _optional_wavelength(
            value.get("wavelengthMinNm"), "wavelengthMinNm"
        )
        wavelength_max = _optional_wavelength(
            value.get("wavelengthMaxExclusiveNm"), "wavelengthMaxExclusiveNm"
        )
        if (
            wavelength_min is not None
            and wavelength_max is not None
            and wavelength_min >= wavelength_max
        ):
            raise ValueError(
                f"Variance component {component_id!r} wavelength range is empty."
            )
        component = cls(
            id=component_id,
            station=station,
            start=start,
            end_exclusive=end,
            wavelength_min_nm=wavelength_min,
            wavelength_max_exclusive_nm=wavelength_max,
        )
        return component

    def matches(self, equation: ObservationEquation) -> bool:
        date = equation.epoch.date_iso()
        if date < self.start or (self.end_exclusive is not None and date >= self.end_exclusive):
            return False
        if canonical_station_id(equation.station_key) != self.station:
            return False
        wavelength = equation.wavelength_nm
        if self.wavelength_min_nm is not None or self.wavelength_max_exclusive_nm is not None:
            if wavelength is None:
                return False
            wavelength = float(wavelength)
            if self.wavelength_min_nm is not None and wavelength < self.wavelength_min_nm:
                return False
            if self.wavelength_max_exclusive_nm is not None and wavelength >= self.wavelength_max_exclusive_nm:
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
