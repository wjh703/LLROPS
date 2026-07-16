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
from typing import Iterable, List, Sequence


KNOWN_PARAMETER_TYPES: frozenset[str] = frozenset(
    {
        # Implemented parameter blocks.
        "position.x",
        "position.y",
        "position.z",
        "rangeBias",
        # Reserved names for planned blocks; keeping the registry here makes
        # normal-equation files self-checking as new blocks are introduced.
        "eop.xp",
        "eop.yp",
        "eop.ut1MinusUtc",
        "orbitState.x0",
        "orbitState.y0",
        "orbitState.z0",
        "orbitState.vx0",
        "orbitState.vy0",
        "orbitState.vz0",
        "loveNumber.h2",
        "loveNumber.l2",
    }
)


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


def validate_parameter_types(
    names: Iterable[ParameterName],
    *,
    allowed_types: Iterable[str] = KNOWN_PARAMETER_TYPES,
) -> None:
    """Validate parameter-name ``type`` fields against the LLROPS schema."""
    allowed = set(allowed_types)
    unknown = sorted({name.type for name in names if name.type and name.type not in allowed})
    if unknown:
        raise ValueError(
            "Unknown parameter type(s) in structured names: "
            f"{unknown!r}. Register the type in llrops.base.parameter_name."
        )


__all__ = [
    "KNOWN_PARAMETER_TYPES",
    "ParameterName",
    "names_to_strings",
    "strings_to_names",
    "validate_parameter_types",
]
