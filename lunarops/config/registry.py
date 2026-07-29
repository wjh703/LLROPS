"""GROOPS-style class registry.

In GROOPS every configurable concept (ephemerides, troposphere, tides,
parametrization, ...) is an abstract *class category*; concrete
implementations register themselves under a ``type`` name and are
instantiated from the config file.  This module provides that mechanism.

Usage
-----
Registering an implementation::

    @register("troposphere", "mendesPavlis")
    class Iers2010MendesPavlisTroposphere: ...

or, when the class lives in an unmodified physics module::

    register_factory("troposphere", "mendesPavlis",
                     lambda cfg, ctx: Iers2010MendesPavlisTroposphere())

Instantiating from config::

    model = create("troposphere", {"type": "mendesPavlis"}, context)

Config conventions
------------------
* A class config is either a plain string ``"mendesPavlis"`` (no options) or a
  mapping ``{"type": "mendesPavlis", ...options...}``.
* A *list* of class configs is allowed for categories whose base class
  supports composition (e.g. stationDisplacement); ``create_list`` returns the
  instantiated list.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

Factory = Callable[[dict, "object"], Any]

_REGISTRY: Dict[str, Dict[str, Factory]] = {}


class UnknownClassError(KeyError):
    pass


def register_factory(category: str, type_name: str, factory: Factory) -> None:
    _REGISTRY.setdefault(category, {})[type_name.lower()] = factory


def register(category: str, type_name: str):
    """Decorator form.  The class must accept ``**options`` in ``__init__`` or
    provide ``from_config(cls, config, context)``."""

    def _wrap(cls):
        def _factory(config: dict, context) -> Any:
            if hasattr(cls, "from_config"):
                return cls.from_config(config, context)
            options = {k: v for k, v in config.items() if k != "type"}
            return cls(**options)

        register_factory(category, type_name, _factory)
        cls._registry_category = category
        cls._registry_type = type_name
        return cls

    return _wrap


def normalize_class_config(config) -> dict:
    if config is None:
        return {"type": "none"}
    if isinstance(config, str):
        return {"type": config}
    if isinstance(config, dict):
        if "type" not in config:
            raise ValueError(f"Class config mapping requires a 'type' key: {config!r}")
        return config
    raise TypeError(f"Unsupported class config: {config!r}")


def create(category: str, config, context=None):
    """Instantiate one implementation of *category* from *config*."""
    cfg = normalize_class_config(config)
    type_name = str(cfg["type"]).lower()
    try:
        factory = _REGISTRY[category][type_name]
    except KeyError:
        raise UnknownClassError(
            f"No implementation {cfg['type']!r} registered for category {category!r}. "
            f"Available: {sorted(_REGISTRY.get(category, {}))}"
        ) from None
    return factory(cfg, context)


def create_list(category: str, configs, context=None) -> List[Any]:
    if configs is None:
        return []
    if isinstance(configs, (str, dict)):
        configs = [configs]
    return [create(category, cfg, context) for cfg in configs]


def available(category: Optional[str] = None):
    if category is None:
        return {cat: sorted(types) for cat, types in _REGISTRY.items()}
    return sorted(_REGISTRY.get(category, {}))
