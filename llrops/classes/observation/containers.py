"""Small immutable, pickle-friendly containers used by observation objects."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class FrozenMapping(Mapping[K, V], Generic[K, V]):
    """Read-only mapping that remains safe to transport through MPI/pickle."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[K, V] | None = None, /, **kwargs: V) -> None:
        data = dict(values or {})
        data.update(kwargs)
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __reduce__(self):
        return type(self), (self._data,)

    def to_dict(self) -> dict[K, V]:
        return dict(self._data)


__all__ = ["FrozenMapping"]
