"""Shared declarative schema fragments for LLR programs."""

from __future__ import annotations

OBSERVATION_MODEL_KEYS = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
    "stationCatalog",
    "reflectorCatalog",
)

NORMAL_POINT_SELECTION_KEYS = (
    "combineInputs",
    "combinedName",
    "startTime",
    "endTime",
    "stationName",
    "reflectorName",
    "minElevationDeg",
    "showProgress",
    "mpi",
)

OBSERVATION_PROGRAM_KEYS = (
    *NORMAL_POINT_SELECTION_KEYS,
    *OBSERVATION_MODEL_KEYS,
)

PARAMETRIZED_OBSERVATION_KEYS = (
    *OBSERVATION_PROGRAM_KEYS,
    "parametrization",
)


__all__ = [
    "NORMAL_POINT_SELECTION_KEYS",
    "OBSERVATION_MODEL_KEYS",
    "OBSERVATION_PROGRAM_KEYS",
    "PARAMETRIZED_OBSERVATION_KEYS",
]
