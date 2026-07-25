"""LLR observation modelling, processing, and linearized equations.

Heavy model modules are imported lazily so the generic estimation framework can
use :class:`ObservationEquation` in minimal environments.
"""
from __future__ import annotations

from importlib import import_module

from .equations import ObservationEquation

_LAZY_EXPORTS = {
    "FrozenMapping": ("frozen_mapping", "FrozenMapping"),
    "CatalogSelection": ("resolver", "CatalogSelection"),
    "ResolvedObservation": ("resolver", "ResolvedObservation"),
    "ObservationModelState": ("resolver", "ObservationModelState"),
    "ObservationResolver": ("resolver", "ObservationResolver"),
    "LightTimeLeg": ("light_time", "LightTimeLeg"),
    "LightTimeRequest": ("light_time", "LightTimeRequest"),
    "LightTimeSolution": ("light_time", "LightTimeSolution"),
    "LightTimeSolver": ("light_time", "LightTimeSolver"),
    "OpticalAtmosphere": ("light_time", "OpticalAtmosphere"),
    "LlrObservationModel": ("model", "LlrObservationModel"),
    "LlrPrediction": ("model", "LlrPrediction"),
    "LlrObservationReducer": ("reduction", "LlrObservationReducer"),
    "ObservationReduction": ("reduction", "ObservationReduction"),
    "LlrObservationProcessor": ("processor", "LlrObservationProcessor"),
    "ObservationProcessingOptions": ("processor", "ObservationProcessingOptions"),
    "ObservationOutputLevel": ("equations", "ObservationOutputLevel"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


__all__ = ["ObservationEquation", *_LAZY_EXPORTS]
