"""GROOPS-style parameter names.

GROOPS identifies every estimated parameter by a structured name
``object:type:temporal:interval``.  Structured names are what make normal
equations *combinable across programs*: two normal-equation files can be
merged by aligning parameter names instead of hoping the column order agrees.

Examples
--------
``apollo15:position.x::``                     reflector PA x-coordinate
``GRASSE:rangeBias::``                        per-station range bias
``earth:polarMotion.xp:trend:``               (future) EOP parameter
``moon:orbitState.x0::``                      (future) integrated orbit ICs
``moon:loveNumber.h2::``                      (future) lunar tide parameter
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True, order=True, slots=True)
class ParameterName:
    object: str = ""
    type: str = ""
    temporal: str = ""
    interval: str = ""

    def __post_init__(self) -> None:
        for field_name in ("object", "type", "temporal", "interval"):
            value = getattr(self, field_name)
            text = str(value or "").strip()
            if ":" in text:
                raise ValueError(f"ParameterName.{field_name} must not contain ':' characters.")
            object.__setattr__(self, field_name, text)
        if not self.type:
            raise ValueError("ParameterName.type must not be empty.")

    def __str__(self) -> str:
        return f"{self.object}:{self.type}:{self.temporal}:{self.interval}"

    @classmethod
    def parse(cls, text: str) -> "ParameterName":
        parts = str(text).split(":")
        if len(parts) > 4:
            raise ValueError(f"Structured parameter name has too many fields: {text!r}")
        return cls(*(parts + ["", "", "", ""])[:4])


def names_to_strings(names: Sequence[ParameterName]) -> List[str]:
    return [str(n) for n in names]


def strings_to_names(strings: Sequence[str]) -> List[ParameterName]:
    return [ParameterName.parse(s) for s in strings]


__all__ = [
    "ParameterName",
    "names_to_strings",
    "strings_to_names",
]
