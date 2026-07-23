"""Run context shared across programs in one config run.

Analogue of GROOPS' global config elements: heavyweight objects (ephemeris
backend, frame system, catalogs, IERS table) are declared once under
``globals:`` in the config and lazily constructed on first use; subsequent
programs in the same run reuse the same instance.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .registry import create, normalize_class_config
from llrops.resource_lifecycle import close_resources


def _config_key(category: str, config) -> str:
    payload = json.dumps({"category": category, "config": normalize_class_config(config)}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class RunContext:
    """Carries variables, global class configs and a shared object cache."""

    def __init__(
        self,
        *,
        variables: Optional[Dict[str, Any]] = None,
        global_class_configs: Optional[Dict[str, Any]] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        self.variables: Dict[str, Any] = dict(variables or {})
        self.global_class_configs: Dict[str, Any] = dict(global_class_configs or {})
        self.working_dir = Path(working_dir or ".").expanduser()
        self._cache: Dict[str, Any] = {}
        # Free-form slots that programs may publish for later programs
        # (e.g. "stationCatalog", "reflectorCatalog", "observationProcessor").
        self.shared: Dict[str, Any] = {}

    # -- class instantiation ------------------------------------------------
    def create_class(self, category: str, config=None, *, cache: bool = True):
        """Instantiate a class; ``config=None`` falls back to ``globals:``.

        With ``cache=True`` (default) identical (category, config) pairs share
        one instance for the lifetime of the run — this is how the CALCEPH
        ephemeris or Earth-orientation source is opened once and reused by every program.
        """
        if config is None:
            if category not in self.global_class_configs:
                raise KeyError(
                    f"Program requires class {category!r} but neither the program "
                    f"config nor the run 'globals:' section defines it."
                )
            config = self.global_class_configs[category]
        if not cache:
            return create(category, config, self)
        key = _config_key(category, config)
        if key not in self._cache:
            self._cache[key] = create(category, config, self)
        return self._cache[key]

    def class_config(self, category: str, program_config: dict, key: Optional[str] = None):
        """Return the class config for *category*: program entry overrides globals."""
        key = key or category
        if key in program_config:
            return program_config[key]
        return self.global_class_configs.get(category)

    # -- paths ---------------------------------------------------------------
    def resolve_path(self, value) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.working_dir / path
        return path

    def close(self) -> None:
        close_resources(self._cache.values(), owner="run-context")
        self._cache.clear()
