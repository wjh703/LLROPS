"""Parametrization base class (GROOPS ``parametrization`` analogue).

A parametrization owns a contiguous block of design-matrix columns.  It

* declares structured :class:`ParameterName`s,
* fills its columns of an observation-equation row from the equation's
  named partial blocks,
* maps solved corrections back into model state (catalogs, bias tables,
  force-model coefficients, integrator initial conditions, ...),
* reports its current values for output.

The estimator (:mod:`llrops.estimation.adjustment`) and the normal-equation
builder are completely generic over the parametrization list — adding EOP,
Love-number or orbit-state parameters never touches them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName, validate_parameter_types
from llrops.base.validation import parameter_vector
from llrops.classes.observation.equations import ObservationEquation


class Parametrization:
    """One block of estimated parameters."""

    #: overridden by subclasses
    category = "parametrization"

    def setup(self, equations: Sequence[ObservationEquation], context) -> None:
        """Inspect the dataset once before the first iteration (e.g. discover
        which reflectors / stations actually occur).  Default: no-op."""

    def parameter_names(self) -> List[ParameterName]:
        raise NotImplementedError

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names())

    def design_columns(self, eq: ObservationEquation) -> np.ndarray:
        """Return this block's row segment of the design matrix, shape (p_block,)."""
        raise NotImplementedError

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        """Return non-zero local design entries as ``(column, value)`` pairs.

        Blocks with naturally sparse rows should override this to avoid
        allocating a dense block vector for every observation.
        """
        columns = np.asarray(self.design_columns(eq), dtype=float).reshape(-1)
        return [(int(index), float(columns[index])) for index in np.flatnonzero(columns)]

    def reduce_observation(self, eq: ObservationEquation) -> float:
        """Amount to subtract from ``eq.observed_minus_computed_m`` for the *current* parameter
        values (linearization point), e.g. the currently accumulated station
        bias.  Default 0."""
        return 0.0

    def apply_update(self, delta: np.ndarray) -> None:
        """Absorb solved corrections (same order as :meth:`parameter_names`)."""
        raise NotImplementedError

    def max_update_norm(self, delta: np.ndarray) -> float:
        """Convergence metric for this block; default max |delta_i|."""
        return float(np.max(np.abs(delta))) if len(delta) else 0.0

    def state(self) -> Dict[str, object]:
        """Current parameter values for reporting."""
        return {}


class ParametrizationList:
    """Concatenation of parametrization blocks into one design matrix.

    The block layout is cached after :meth:`setup`, so streaming normal-equation
    accumulation does not repeatedly rebuild parameter-name lists or search for
    block slice boundaries row by row.
    """

    def __init__(self, blocks: Sequence[Parametrization]) -> None:
        self.blocks: List[Parametrization] = list(blocks)
        self._parameter_names: List[ParameterName] | None = None
        self._slices: List[slice] = []

    def _ensure_layout(self) -> None:
        if self._parameter_names is not None:
            return
        names: List[ParameterName] = []
        slices: List[slice] = []
        offset = 0
        for block in self.blocks:
            block_names = list(block.parameter_names())
            names.extend(block_names)
            next_offset = offset + len(block_names)
            slices.append(slice(offset, next_offset))
            offset = next_offset
        self._parameter_names = names
        validate_parameter_types(self._parameter_names)
        self._slices = slices

    def setup(self, equations: Sequence[ObservationEquation], context) -> None:
        self._parameter_names = None
        self._slices = []
        for block in self.blocks:
            block.setup(equations, context)
        self._ensure_layout()

    def parameter_names(self) -> List[ParameterName]:
        self._ensure_layout()
        return list(self._parameter_names or [])

    def select_blocks(self, selectors: Sequence[str]) -> "ParametrizationList":
        """Return a view over selected parameter blocks, reusing block state.

        Selectors are exact class names, for example
        ``ReflectorPositionParametrization``. An empty selector list is invalid
        so a processing step cannot silently solve a zero-parameter system.
        """
        requested = {str(value).strip() for value in selectors if str(value).strip()}
        if not requested:
            raise ValueError("At least one parametrization block selector is required.")
        selected = [block for block in self.blocks if type(block).__name__ in requested]
        found = {type(block).__name__ for block in selected}
        missing = requested - found
        if missing:
            raise KeyError(f"Unknown parametrization block selector(s): {sorted(missing)}")
        return ParametrizationList(selected)

    @property
    def parameter_count(self) -> int:
        self._ensure_layout()
        return len(self._parameter_names or [])

    def _block_design_entries(
        self,
        block: Parametrization,
        block_slice: slice,
        eq: ObservationEquation,
    ) -> list[tuple[int, float]]:
        expected = block_slice.stop - block_slice.start
        if type(block).design_entries is Parametrization.design_entries:
            columns = np.asarray(block.design_columns(eq), dtype=float).reshape(-1)
            if columns.size != expected:
                raise ValueError(
                    f"{type(block).__name__}.design_columns() returned {columns.size} "
                    f"columns, expected {expected}."
                )
            return [
                (block_slice.start + int(index), float(columns[index]))
                for index in np.flatnonzero(columns)
            ]

        entries: list[tuple[int, float]] = []
        for local_index, value in block.design_entries(eq):
            index = int(local_index)
            if index < 0 or index >= expected:
                raise ValueError(
                    f"{type(block).__name__}.design_entries() returned local "
                    f"column {index}, expected [0, {expected})."
                )
            scalar = float(value)
            if scalar:
                entries.append((block_slice.start + index, scalar))
        return entries

    def design_entries(self, eq: ObservationEquation) -> list[tuple[int, float]]:
        self._ensure_layout()
        entries: list[tuple[int, float]] = []
        for block, block_slice in zip(self.blocks, self._slices):
            entries.extend(self._block_design_entries(block, block_slice, eq))
        return entries

    def design_row(self, eq: ObservationEquation) -> np.ndarray:
        self._ensure_layout()
        row = np.zeros(self.parameter_count, dtype=float)
        for index, value in self.design_entries(eq):
            row[index] += value
        return row

    def design_value(self, eq: ObservationEquation, coefficients: np.ndarray) -> float:
        values = parameter_vector(coefficients, expected_size=self.parameter_count, name="coefficients")
        return float(sum(values[index] * value for index, value in self.design_entries(eq)))

    def reduced_observation(self, eq: ObservationEquation) -> float:
        return float(eq.observed_minus_computed_m) - sum(block.reduce_observation(eq) for block in self.blocks)

    def split(self, delta: np.ndarray) -> List[np.ndarray]:
        self._ensure_layout()
        values = parameter_vector(delta, expected_size=self.parameter_count, name="delta")
        return [values[block_slice] for block_slice in self._slices]

    def apply_update(self, delta: np.ndarray) -> Dict[str, float]:
        """Apply all block updates; returns per-block max update norms."""
        norms: Dict[str, float] = {}
        duplicate_types = {
            type(block).__name__
            for block in self.blocks
            if sum(type(other).__name__ == type(block).__name__ for other in self.blocks) > 1
        }
        for index, (block, block_delta) in enumerate(zip(self.blocks, self.split(delta))):
            block.apply_update(block_delta)
            label = type(block).__name__
            if label in duplicate_types:
                label = f"{index}:{label}"
            norms[label] = block.max_update_norm(block_delta)
        return norms

    def state(self) -> Dict[str, object]:
        merged: Dict[str, object] = {}
        duplicate_types = {
            type(block).__name__
            for block in self.blocks
            if sum(type(other).__name__ == type(block).__name__ for other in self.blocks) > 1
        }
        for index, block in enumerate(self.blocks):
            label = type(block).__name__
            if label in duplicate_types:
                label = f"{index}:{label}"
            merged[label] = block.state()
        return merged
