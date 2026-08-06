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

from contextlib import contextmanager
from threading import RLock
from typing import Any, Callable, Dict, Iterator, List, Optional

Factory = Callable[[dict, "object"], Any]

_REGISTRY: Dict[str, Dict[str, Factory]] = {}
_REGISTRY_LOCK = RLock()


class UnknownClassError(KeyError):
    pass


class DuplicateClassRegistrationError(ValueError):
    """Raised when a factory would replace an existing type implicitly."""


def register_factory(
    category: str,
    type_name: str,
    factory: Factory,
    *,
    replace: bool = False,
) -> None:
    """Register one config factory.

    Replacing an existing ``(category, type)`` is opt-in.  This prevents a
    plugin or an import-order change from silently changing a configured
    physical model.
    """
    normalized_type_name = type_name.lower()
    with _REGISTRY_LOCK:
        category_factories = _REGISTRY.setdefault(category, {})
        if normalized_type_name in category_factories and not replace:
            raise DuplicateClassRegistrationError(
                f"Implementation {type_name!r} is already registered for category {category!r}. "
                "Pass replace=True to replace it explicitly."
            )
        category_factories[normalized_type_name] = factory


@contextmanager
def _registration_transaction() -> Iterator[None]:
    """Restore the registry if a built-in registration batch fails."""
    with _REGISTRY_LOCK:
        snapshot = {category: factories.copy() for category, factories in _REGISTRY.items()}
        try:
            yield
        except Exception:
            _REGISTRY.clear()
            _REGISTRY.update(snapshot)
            raise


def register(category: str, type_name: str, *, replace: bool = False):
    """Decorator form.  The class must accept ``**options`` in ``__init__`` or
    provide ``from_config(cls, config, context)``."""

    def _wrap(cls):
        def _factory(config: dict, context) -> Any:
            if hasattr(cls, "from_config"):
                return cls.from_config(config, context)
            options = {k: v for k, v in config.items() if k != "type"}
            return cls(**options)

        register_factory(category, type_name, _factory, replace=replace)
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
    with _REGISTRY_LOCK:
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
    with _REGISTRY_LOCK:
        if category is None:
            return {cat: sorted(types) for cat, types in _REGISTRY.items()}
        return sorted(_REGISTRY.get(category, {}))
