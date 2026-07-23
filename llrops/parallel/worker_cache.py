"""Lifecycle management for process-local worker caches."""

from __future__ import annotations

from typing import Any

from llrops.resource_lifecycle import close_resource


def close_cached_objects(cache: dict[Any, Any]) -> None:
    """Close unique resources reachable from a nested worker cache."""
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        close_resource(value, owner="mpi-worker-cache")

    for value in cache.values():
        walk(value)


__all__ = ["close_cached_objects"]
