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
from typing import TYPE_CHECKING, Any, Dict, Mapping, MutableMapping, Optional

if TYPE_CHECKING:
    from lunarops.parallel.mpi import MpiRuntime

from .registry import create, normalize_class_config
from lunarops.resource_lifecycle import close_resources


def _config_key(category: str, config) -> str:
    payload = json.dumps({"category": category, "config": normalize_class_config(config)}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class RunContext:
    """Own run configuration, runtime services, and cached class instances."""

    def __init__(
        self,
        *,
        global_class_configs: Optional[Dict[str, Any]] = None,
        working_dir: Optional[str] = None,
        runtime: MpiRuntime | None = None,
        mpi_resources: Mapping[str, object] | None = None,
        class_cache: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.global_class_configs: Dict[str, Any] = dict(global_class_configs or {})
        self.working_dir = Path(working_dir or ".").expanduser()
        self.runtime = runtime
        self.mpi_resources: Dict[str, object] = dict(mpi_resources or {})
        self._cache: MutableMapping[str, Any] = class_cache if class_cache is not None else {}
        self._observation_spec_sequence = 0

    # -- class instantiation ------------------------------------------------
    def create_class(
        self,
        category: str,
        config=None,
        *,
        cache: bool = True,
        factory_context=None,
        cache_namespace: str = "",
    ):
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
        target_context = self if factory_context is None else factory_context
        if not cache:
            return create(category, config, target_context)
        key = f"{cache_namespace}:{_config_key(category, config)}"
        if key not in self._cache:
            self._cache[key] = create(category, config, target_context)
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

    def next_observation_spec_id(self) -> str:
        self._observation_spec_sequence += 1
        return f"{id(self)}:{self._observation_spec_sequence}"

    def close(self) -> None:
        close_resources(self._cache.values(), owner="run-context")
        self._cache.clear()
