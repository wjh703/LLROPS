"""Array validation helpers shared by physical-model modules."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def finite_array(
    value,
    *,
    shape: tuple[int, ...] | None = None,
    size: int | None = None,
    name: str,
    copy: bool = True,
    readonly: bool = False,
) -> np.ndarray:
    """Return a finite float array with one explicit shape/size contract.

    ``shape`` is preferred for matrix-like values.  ``size`` is useful when the
    accepted input may be any shape but must contain a fixed number of values.
    The returned array is always reshaped when either contract is supplied.
    """
    if shape is not None and size is not None:
        raise ValueError("finite_array() accepts either shape or size, not both.")
    array = np.array(value, dtype=float, copy=copy)
    if shape is not None:
        expected = int(np.prod(shape))
        if array.size != expected:
            raise ValueError(
                f"{name} must contain exactly {expected} values, got shape {array.shape}."
            )
        array = array.reshape(shape)
    elif size is not None:
        if array.size != int(size):
            raise ValueError(
                f"{name} must contain exactly {int(size)} values, got shape {array.shape}."
            )
        array = array.reshape(int(size))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if readonly:
        array.setflags(write=False)
    return array


def vector3(
    value: Sequence[float],
    *,
    name: str,
    copy: bool = True,
    readonly: bool = False,
) -> np.ndarray:
    return finite_array(value, size=3, name=name, copy=copy, readonly=readonly)


def readonly_vector3(value: Sequence[float], *, name: str) -> np.ndarray:
    return vector3(value, name=name, copy=True, readonly=True)


def matrix3x3(value, *, name: str, copy: bool = True, readonly: bool = False) -> np.ndarray:
    return finite_array(value, shape=(3, 3), name=name, copy=copy, readonly=readonly)


def readonly_matrix3x3(value, *, name: str) -> np.ndarray:
    return matrix3x3(value, name=name, copy=True, readonly=True)


def parameter_vector(value, *, expected_size: int, name: str = "delta") -> np.ndarray:
    """Return a finite 1-D parameter vector with an exact length."""
    size = int(expected_size)
    if size < 0:
        raise ValueError("expected_size must be non-negative.")
    return finite_array(value, size=size, name=name, copy=True, readonly=False)


def catalog_vector3(value: Sequence[float], *, name: str) -> np.ndarray:
    """Validate mutable catalog position/velocity triples."""
    return vector3(value, name=name, copy=True, readonly=False)


__all__ = [
    "finite_array",
    "catalog_vector3",
    "matrix3x3",
    "parameter_vector",
    "readonly_matrix3x3",
    "readonly_vector3",
    "vector3",
]
