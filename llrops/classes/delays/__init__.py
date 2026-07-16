from __future__ import annotations

from llrops.classes.delays.base import (
    GravitationalDelay,
    TroposphereDelay,
    TroposphereInput,
    ZeroGravitationalDelay,
    ZeroTroposphereDelay,
)

__all__ = [
    "GravitationalDelay",
    "TroposphereDelay",
    "TroposphereInput",
    "ZeroGravitationalDelay",
    "ZeroTroposphereDelay",
    "Iers2010ShapiroDelay",
    "Iers2010MendesPavlisTroposphere",
]


def __getattr__(name: str):
    """Load concrete delay models only when requested.

    This keeps the lightweight base/troposphere API importable in environments
    where optional astronomy dependencies used by the Shapiro model are absent.
    """
    if name == "Iers2010ShapiroDelay":
        from llrops.classes.delays.shapiro import Iers2010ShapiroDelay

        return Iers2010ShapiroDelay
    if name == "Iers2010MendesPavlisTroposphere":
        from llrops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere

        return Iers2010MendesPavlisTroposphere
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
