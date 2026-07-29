"""Small lifecycle helpers shared by run contexts and execution backends."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def close_resource(resource: Any, *, owner: str) -> None:
    """Close one optional resource and report cleanup failures."""
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.warning(
            "Failed to close %s resource %s",
            owner,
            type(resource).__name__,
            exc_info=True,
        )


def close_resources(resources: Iterable[Any], *, owner: str) -> None:
    """Close each unique resource, logging but isolating failures."""
    seen: set[int] = set()
    for resource in resources:
        identity = id(resource)
        if identity in seen:
            continue
        seen.add(identity)
        close_resource(resource, owner=owner)


__all__ = ["close_resource", "close_resources"]
