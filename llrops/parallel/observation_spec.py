"""Picklable observation specifications shared by serial and MPI execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
    "uncertaintyModel",
)


# ---------------------------------------------------------------------------
# observation spec: everything a worker needs to build its own observation processor
# ---------------------------------------------------------------------------


def _prepare_shared_resources(merged: dict, context) -> dict:
    """Build rank-0-only immutable resources for one observation spec.

    Text tables are parsed once on rank 0 and serialized as compact arrays.
    Process-local native handles (notably CALCEPH) are intentionally excluded;
    they cannot be safely pickled or shared between MPI processes.
    """
    resources: dict = {}
    earth_rotation_config = merged.get("earthRotation")
    if earth_rotation_config is not None:
        from llrops.classes.builders import ensure_registered
        from llrops.classes.frames import C04EarthOrientation
        from llrops.config.registry import normalize_class_config

        cfg = normalize_class_config(earth_rotation_config)
        if str(cfg["type"]).strip().lower() == "iersc04":
            ensure_registered()
            earth_orientation = context.create_class(
                "earthRotation",
                earth_rotation_config,
                cache=True,
            )
            if not isinstance(earth_orientation, C04EarthOrientation):
                raise TypeError(
                    "MPI earthRotation resource preparation expected "
                    "C04EarthOrientation."
                )
            context.shared["earthOrientation"] = earth_orientation
            resources["earthRotation"] = earth_orientation.to_mpi_payload()
    return resources


def make_observation_spec(config: dict, context, datasets) -> dict:
    """Resolve one picklable observation specification on rank 0.

    The complete specification is broadcast once to every worker and then
    referenced by ``specId`` in individual tasks.  This avoids repeatedly
    pickling catalogs and EOP arrays for every small chunk.
    """
    merged: dict = {}
    for category in _MODEL_CATEGORIES:
        value = context.class_config(category, config)
        if value is not None:
            merged[category] = value

    station_catalog = context.shared.get("stationCatalog")
    if station_catalog is None:
        from llrops.fileio.catalogs import load_station_catalog

        station_catalog = load_station_catalog(
            config.get(
                "stationCatalog", context.global_class_configs.get("stationCatalog")
            )
        )
        context.shared["stationCatalog"] = station_catalog
    reflector_catalog = context.shared.get("reflectorCatalog")
    if reflector_catalog is None:
        from llrops.fileio.catalogs import load_reflector_catalog

        reflector_catalog = load_reflector_catalog(
            config.get(
                "reflectorCatalog", context.global_class_configs.get("reflectorCatalog")
            )
        )
        context.shared["reflectorCatalog"] = reflector_catalog

    spec_id = (
        f"{id(context)}-{hash(repr(sorted(merged.items(), key=lambda kv: kv[0])))}"
    )
    return {
        "specId": spec_id,
        "programConfig": merged,
        "workingDir": str(context.working_dir),
        "stationCatalog": station_catalog,
        "reflectorCatalog": reflector_catalog,
        "sharedResources": _prepare_shared_resources(merged, context),
    }


def build_worker_processor(spec: dict, shared_class_cache: Optional[dict] = None):
    from llrops.config.context import RunContext
    from llrops.classes.builders import build_observation_processor

    context = RunContext(
        variables={},
        global_class_configs={},
        working_dir=spec.get("workingDir", "."),
    )
    context.shared["mpiResources"] = dict(spec.get("sharedResources") or {})
    if shared_class_cache is not None:
        # Reuse immutable/heavy classes (CALCEPH ephemeris, Earth-orientation source, immutable
        # observation components) without caching the mutable LlrObservationProcessor itself.
        context._cache = shared_class_cache
    return build_observation_processor(
        context,
        spec["programConfig"],
        station_catalog=spec["stationCatalog"],
        reflector_catalog=spec["reflectorCatalog"],
    )


def snapshot_catalog_state(context) -> dict:
    """Pickle-light snapshot of the mutable per-iteration model state."""
    reflectors = context.shared.get("reflectorCatalog") or {}
    return {
        "reflectorPositions": {
            str(key): [
                float(x)
                for x in np.asarray(rec.moon_fixed_xyz_m, dtype=float).reshape(3)
            ]
            for key, rec in reflectors.items()
        }
    }


def apply_catalog_state(processor, catalog_state: Optional[dict]) -> None:
    if not catalog_state:
        return
    positions = catalog_state.get("reflectorPositions") or {}
    if positions:
        new_catalog = {}
        for key, rec in processor.reflector_catalog.items():
            if key in positions:
                rec = replace(
                    rec, moon_fixed_xyz_m=np.asarray(positions[key], dtype=float)
                )
            new_catalog[key] = rec
        processor.reflector_catalog = new_catalog


__all__ = [
    "apply_catalog_state",
    "build_worker_processor",
    "make_observation_spec",
    "snapshot_catalog_state",
]
