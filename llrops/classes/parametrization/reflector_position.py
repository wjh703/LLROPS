"""Parametrization: lunar reflector PA-frame coordinates (3 per reflector)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName
from llrops.base.validation import parameter_vector
from llrops.config.registry import register
from llrops.classes.observation.equations import ObservationEquation
from .base import Parametrization

_AXES = ("x", "y", "z")


@register("parametrization", "reflectorPosition")
class ReflectorPositionParametrization(Parametrization):
    """Estimate corrections to reflector moon-fixed (PA) coordinates.

    Options
    -------
    reflectors :
        Optional explicit list of reflector catalog keys to estimate; default
        is every reflector present in the observation set.
    catalog :
        Injected at setup time through ``context.shared["reflectorCatalog"]``;
        :meth:`apply_update` writes the corrected positions back into it, so
        the forward model of the next Gauss–Newton iteration directly
        relinearizes about the updated coordinates.
    """

    def __init__(self, *, reflectors: Optional[Sequence[str]] = None) -> None:
        self.requested = list(reflectors) if reflectors else None
        self.keys: List[str] = []
        self._index_by_key: Dict[str, int] = {}
        self._names: List[ParameterName] = []
        self._catalog: Dict[str, object] = {}

    @classmethod
    def from_config(cls, config: dict, context) -> "ReflectorPositionParametrization":
        return cls(reflectors=config.get("reflectors"))

    def setup(self, equations: Sequence[ObservationEquation], context) -> None:
        self._catalog = context.shared["reflectorCatalog"]
        observed = sorted({eq.reflector_key for eq in equations})
        self.keys = [k for k in (self.requested or observed) if k in self._catalog]
        missing = set(self.requested or []) - set(self.keys)
        if missing:
            raise KeyError(f"reflectorPosition: unknown reflector key(s) {sorted(missing)}")
        self._index_by_key = {key: index for index, key in enumerate(self.keys)}
        self._names = [ParameterName(key, f"position.{axis}") for key in self.keys for axis in _AXES]

    def parameter_names(self) -> List[ParameterName]:
        return list(self._names)

    def _partial_block(self, eq: ObservationEquation) -> tuple[int | None, np.ndarray | None]:
        j = self._index_by_key.get(eq.reflector_key)
        if j is None:
            return None, None
        block = eq.partials.get("reflector_position_pa")
        if block is None:
            raise KeyError(
                "Observation equation lacks the 'reflector_position_pa' partial "
                "block; run the forward model with include_reflector_design=True."
            )
        return j, np.asarray(block, dtype=float).reshape(3)

    def design_columns(self, eq: ObservationEquation) -> np.ndarray:
        cols = np.zeros(3 * len(self.keys), dtype=float)
        j, block = self._partial_block(eq)
        if j is not None and block is not None:
            cols[3 * j : 3 * j + 3] = block
        return cols

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        j, block = self._partial_block(eq)
        if j is None or block is None:
            return []
        start = 3 * j
        return [(start + axis, float(value)) for axis, value in enumerate(block) if float(value)]

    def apply_update(self, delta: np.ndarray) -> None:
        delta = parameter_vector(delta, expected_size=3 * len(self.keys), name="reflectorPosition update")
        for j, key in enumerate(self.keys):
            record = self._catalog[key]
            record.moon_fixed_xyz_m = (
                np.asarray(record.moon_fixed_xyz_m, dtype=float) + delta[3 * j : 3 * j + 3]
            )

    def max_update_norm(self, delta: np.ndarray) -> float:
        if not len(delta):
            return 0.0
        return max(
            float(np.linalg.norm(delta[3 * j : 3 * j + 3])) for j in range(len(self.keys))
        )

    def state(self) -> Dict[str, object]:
        return {
            key: [float(v) for v in np.asarray(self._catalog[key].moon_fixed_xyz_m, dtype=float)]
            for key in self.keys
        }
