"""LLR observation modelling, processing, and linearized equations.

Heavy model modules are imported lazily so the generic estimation framework can
use :class:`ObservationEquation` in minimal environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .equations import ObservationEquation

if TYPE_CHECKING:
    from .equations import ObservationResultDetail  # noqa: F401
    from .light_time import (
        LightTimeLeg,  # noqa: F401
        LightTimeRequest,  # noqa: F401
        LightTimeSolution,  # noqa: F401
        LightTimeSolver,  # noqa: F401
        TroposphereEnvironment,  # noqa: F401
    )
    from .measurement import LlrObservationEvaluation, LlrObservationModel  # noqa: F401
    from .processor import LlrObservationProcessor, ObservationProcessingOptions  # noqa: F401
    from .resolver import (
        ObservationCatalogSelection,  # noqa: F401
        ObservationCatalogState,  # noqa: F401
        ObservationResolver,  # noqa: F401
        ResolvedObservation,  # noqa: F401
    )

_LAZY_EXPORTS = {
    "ObservationCatalogSelection": ("resolver", "ObservationCatalogSelection"),
    "ResolvedObservation": ("resolver", "ResolvedObservation"),
    "ObservationCatalogState": ("resolver", "ObservationCatalogState"),
    "ObservationResolver": ("resolver", "ObservationResolver"),
    "LightTimeLeg": ("light_time", "LightTimeLeg"),
    "LightTimeRequest": ("light_time", "LightTimeRequest"),
    "LightTimeSolution": ("light_time", "LightTimeSolution"),
    "LightTimeSolver": ("light_time", "LightTimeSolver"),
    "TroposphereEnvironment": ("light_time", "TroposphereEnvironment"),
    "LlrObservationModel": ("measurement", "LlrObservationModel"),
    "LlrObservationEvaluation": ("measurement", "LlrObservationEvaluation"),
    "LlrObservationProcessor": ("processor", "LlrObservationProcessor"),
    "ObservationProcessingOptions": ("processor", "ObservationProcessingOptions"),
    "ObservationResultDetail": ("equations", "ObservationResultDetail"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ObservationEquation",
    "ObservationCatalogSelection",
    "ResolvedObservation",
    "ObservationCatalogState",
    "ObservationResolver",
    "LightTimeLeg",
    "LightTimeRequest",
    "LightTimeSolution",
    "LightTimeSolver",
    "TroposphereEnvironment",
    "LlrObservationModel",
    "LlrObservationEvaluation",
    "LlrObservationProcessor",
    "ObservationProcessingOptions",
    "ObservationResultDetail",
]
