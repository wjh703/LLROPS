"""Register model implementations and assemble the LLR observation workflow.

This is the single place where config ``type`` names map to physics classes.
The physics modules themselves are untouched ports of v24; registration is
purely additive, so validated numerics stay validated.

Registered categories and types
-------------------------------
ephemerides            : calceph
earthRotation          : iersC04
troposphere            : none | mendesPavlis
relativity             : none | iersShapiro
stationDisplacement    : none | sum | iers2010SolidEarthTide | iers2010PoleTide | iers2010OceanPoleTide | iers2010OceanTidalLoading
reflectorDisplacement  : none | lunarSolidTide
rangeBias             : none | inpop21 | table
parametrization        : reflectorPosition | stationRangeBias   (registered in their modules)

``RunContext.create_class(..., cache=True)`` is intentionally used here for
immutable/heavy backends such as CALCEPH and Earth-orientation sources.  Mutable model state
(catalog coordinates, station-bias values, future EOP/orbit corrections) stays
inside the returned ``LlrObservationProcessor`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

from lunarops.config.registry import register_factory, normalize_class_config

if TYPE_CHECKING:
    from lunarops.classes.ephemerides import Ephemeris
    from lunarops.classes.frames import EarthOrientationProvider, ReferenceFrameSystem
    from lunarops.config.context import RunContext
    from lunarops.fileio.catalogs import ReflectorRecord, StationRecord


_REMOVED_UNCERTAINTY_CONFIG_KEYS = frozenset({"uncertainty", "uncertaintyModel"})
_MODEL_CATEGORIES = (
    "ephemerides",
    "earthRotation",
    "troposphere",
    "relativity",
    "stationDisplacement",
    "reflectorDisplacement",
    "rangeBias",
)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ObservationAssembly:
    """Resolved model configuration and catalogs shared by serial and MPI."""

    program_config: dict
    station_catalog: Mapping[str, "StationRecord"]
    reflector_catalog: Mapping[str, "ReflectorRecord"]


class _PathResolver(Protocol):
    def resolve_path(self, value: object) -> Path: ...


@dataclass(frozen=True, slots=True)
class _ObservationDependencies:
    run_context: "RunContext"
    ephemeris: "Ephemeris"
    earth_orientation_provider: "EarthOrientationProvider"
    frames: "ReferenceFrameSystem"
    cache_namespace: str

    @property
    def mpi_resources(self):
        return self.run_context.mpi_resources

    def resolve_path(self, value: object) -> Path:
        return self.run_context.resolve_path(value)

    def create_class(self, category: str, config=None, *, cache: bool = True):
        return self.run_context.create_class(
            category,
            config,
            cache=cache,
            factory_context=self,
            cache_namespace=self.cache_namespace,
        )


def validate_observation_config(
    program_config: dict,
    global_config: dict | None = None,
) -> None:
    """Reject uncertainty selectors now owned by normal-point records."""
    for scope, config in (
        ("program", program_config),
        ("globals", global_config or {}),
    ):
        removed = sorted(_REMOVED_UNCERTAINTY_CONFIG_KEYS.intersection(config))
        if removed:
            raise ValueError(
                f"{scope} contains removed uncertainty configuration key(s) "
                f"{removed}; every normal-point record must provide "
                "uncertainty_two_way_s."
            )


def _resolve_optional_path(ctx: _PathResolver | None, value: object):
    if value in (None, ""):
        return None
    if ctx is not None:
        return ctx.resolve_path(value)
    return Path(str(value)).expanduser()


def _resolve_required_path(ctx: _PathResolver | None, value: object, *, name: str) -> Path:
    path = _resolve_optional_path(ctx, value)
    if path is None:
        raise ValueError(f"{name} must be a non-empty path.")
    return path


def _register_all() -> None:
    # Imports are local so that merely importing the registry does not load
    # CALCEPH or optional physical-model backends.
    from lunarops.classes.delays.base import ZeroGravitationalDelay, ZeroTroposphereDelay
    from lunarops.classes.delays.shapiro import Iers2010ShapiroDelay
    from lunarops.classes.delays.troposphere import Iers2010MendesPavlisTroposphere
    from lunarops.classes.displacement import (
        CompositeStationDisplacement,
        Iers2010OceanPoleTide,
        Iers2010OceanTidalLoading,
        Iers2010SolidEarthPoleTide,
        Iers2010SolidEarthTide,
        LunarSolidTide,
        OceanPoleTideGrid,
        OceanTidalLoadingCatalog,
        ZeroReflectorDisplacement,
        ZeroStationDisplacement,
    )
    from lunarops.classes.ephemerides import load_calceph_ephemeris
    from lunarops.classes.frames import TabulatedEarthOrientation, load_iers_eop
    from lunarops.classes.range_bias.models import (
        TableRangeBiasModel,
        ZeroRangeBiasModel,
    )
    from lunarops.classes.range_bias.table import (
        RangeBiasTable,
        builtin_range_bias_table,
        load_range_bias_table,
    )

    def _calceph(cfg: dict, ctx):
        if "file" not in cfg:
            raise ValueError("ephemerides/calceph requires 'file'.")
        return load_calceph_ephemeris(
            _resolve_required_path(ctx, cfg["file"], name="ephemerides/calceph file"),
            longitude_libration_correction_type=cfg.get(
                "longitudeLibrationCorrection",
                "none",
            ),
        )

    def _iers_c04(cfg: dict, ctx):
        payload = ctx.mpi_resources.get("earthRotation")
        if payload is not None:
            return TabulatedEarthOrientation.from_mpi_payload(payload)
        if "file" not in cfg:
            raise ValueError("earthRotation/iersC04 requires 'file'.")
        return load_iers_eop(
            _resolve_required_path(ctx, cfg["file"], name="earthRotation/iersC04 file"),
            duplicate_mjd_policy=cfg.get("duplicateMjdPolicy", "error"),
        )

    register_factory("ephemerides", "calceph", _calceph)
    register_factory("earthRotation", "iersc04", _iers_c04)

    register_factory("troposphere", "none", lambda cfg, ctx: ZeroTroposphereDelay())
    register_factory("troposphere", "mendespavlis", lambda cfg, ctx: Iers2010MendesPavlisTroposphere())

    register_factory("relativity", "none", lambda cfg, ctx: ZeroGravitationalDelay())
    register_factory(
        "relativity",
        "iersshapiro",
        lambda cfg, ctx: Iers2010ShapiroDelay(ephemeris=_required_ephemeris(ctx)),
    )

    def _required_earth_orientation(ctx):
        return ctx.earth_orientation_provider

    def _required_ephemeris(ctx):
        return ctx.ephemeris

    def _required_frames(ctx):
        return ctx.frames

    def _station_sum(cfg: dict, ctx) -> CompositeStationDisplacement:
        components_cfg = cfg.get("components", [])
        if isinstance(components_cfg, (str, dict)):
            components_cfg = [components_cfg]
        components = tuple(
            ctx.create_class("stationDisplacement", component, cache=True) for component in components_cfg
        )
        return CompositeStationDisplacement(components)

    def _station_ocean_pole_tide(cfg: dict, ctx) -> Iers2010OceanPoleTide:
        coefficient_file = _resolve_optional_path(ctx, cfg.get("coefficientFile"))
        if coefficient_file is None:
            raise ValueError("stationDisplacement/iers2010OceanPoleTide requires 'coefficientFile'.")
        return Iers2010OceanPoleTide(
            grid=OceanPoleTideGrid(coefficient_file),
            earth_orientation_provider=_required_earth_orientation(ctx),
        )

    def _station_ocean_tidal_loading(cfg: dict, ctx) -> Iers2010OceanTidalLoading:
        coefficient_file = _resolve_optional_path(ctx, cfg.get("coefficientFile"))
        if coefficient_file is None:
            raise ValueError("stationDisplacement/iers2010OceanTidalLoading requires 'coefficientFile'.")
        catalog = OceanTidalLoadingCatalog(coefficient_file)
        expected_model = cfg.get("model")
        actual_model = catalog.info.tidal_model
        if expected_model is not None and (
            actual_model is None or str(expected_model).casefold() != actual_model.casefold()
        ):
            raise ValueError(
                "stationDisplacement/iers2010OceanTidalLoading model mismatch: "
                f"config requests {expected_model!r}, BLQ file declares {actual_model!r}."
            )
        return Iers2010OceanTidalLoading(catalog=catalog)

    register_factory(
        "stationDisplacement",
        "none",
        lambda cfg, ctx: ZeroStationDisplacement(),
    )
    register_factory("stationDisplacement", "sum", _station_sum)
    register_factory(
        "stationDisplacement",
        "iers2010solidearthtide",
        lambda cfg, ctx: Iers2010SolidEarthTide(frames=_required_frames(ctx)),
    )
    register_factory(
        "stationDisplacement",
        "iers2010poletide",
        lambda cfg, ctx: Iers2010SolidEarthPoleTide(earth_orientation_provider=_required_earth_orientation(ctx)),
    )
    register_factory(
        "stationDisplacement",
        "iers2010oceanpoletide",
        _station_ocean_pole_tide,
    )
    register_factory(
        "stationDisplacement",
        "iers2010oceantidalloading",
        _station_ocean_tidal_loading,
    )

    register_factory(
        "reflectorDisplacement",
        "none",
        lambda cfg, ctx: ZeroReflectorDisplacement(),
    )
    register_factory(
        "reflectorDisplacement",
        "lunarsolidtide",
        lambda cfg, ctx: LunarSolidTide(
            ephemeris=_required_ephemeris(ctx),
            h2=float(cfg.get("h2", 0.0423)),
            l2=float(cfg.get("l2", 0.0107)),
            moon_radius_m=float(cfg.get("moonRadiusM", 1_737_400.0)),
        ),
    )

    def _range_bias_table(cfg: dict, ctx) -> TableRangeBiasModel:
        has_file = "file" in cfg
        has_biases = "biases" in cfg
        if has_file == has_biases:
            raise ValueError("rangeBias/table requires exactly one of 'file' or 'biases'.")
        if has_file:
            table = load_range_bias_table(_resolve_required_path(ctx, cfg["file"], name="rangeBias/table file"))
        else:
            table = RangeBiasTable.from_mapping(cfg)
        return TableRangeBiasModel(table)

    register_factory("rangeBias", "none", lambda cfg, ctx: ZeroRangeBiasModel())
    register_factory("rangeBias", "inpop21", lambda cfg, ctx: TableRangeBiasModel(builtin_range_bias_table("inpop21")))
    register_factory(
        "rangeBias", "inpop21a", lambda cfg, ctx: TableRangeBiasModel(builtin_range_bias_table("inpop21a"))
    )
    register_factory("rangeBias", "table", _range_bias_table)

    # Parametrizations register themselves on import.
    import lunarops.classes.parametrization.reflector_position  # noqa: F401
    import lunarops.classes.parametrization.station_range_bias  # noqa: F401


_REGISTERED = False


def ensure_registered() -> None:
    global _REGISTERED
    if not _REGISTERED:
        _register_all()
        _REGISTERED = True


def resolve_observation_assembly(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> ObservationAssembly:
    """Resolve configs, paths, and catalogs once for every execution backend."""
    from lunarops.fileio.catalogs import load_reflector_catalog, load_station_catalog

    validate_observation_config(program_config, context.global_class_configs)
    merged = {
        category: value
        for category in _MODEL_CATEGORIES
        if (value := context.class_config(category, program_config)) is not None
    }

    def catalog_source(name: str):
        value = program_config.get(name, context.global_class_configs.get(name))
        if isinstance(value, str) and value not in ("builtin", ""):
            return context.resolve_path(value)
        return value

    stations = load_station_catalog(catalog_source("stationCatalog")) if station_catalog is None else station_catalog
    reflectors = (
        load_reflector_catalog(catalog_source("reflectorCatalog")) if reflector_catalog is None else reflector_catalog
    )
    return ObservationAssembly(merged, stations, reflectors)


def build_observation_processor(
    context,
    program_config: dict,
    *,
    station_catalog=None,
    reflector_catalog=None,
):
    """Assemble :class:`LlrObservationProcessor` from config.

    Expected class configs (program entry overrides ``globals:``)::

        ephemerides:           {type: calceph, file: ..., longitudeLibrationCorrection: none}
        earthRotation:         {type: iersC04, file: ..., duplicateMjdPolicy: error|first|last|mean}
        troposphere:           mendesPavlis
        relativity:            iersShapiro
        stationDisplacement:   {type: sum, components: [...]} | none
        reflectorDisplacement: lunarSolidTide | none
        rangeBias:             none | inpop21 | {type: table, file: ...} | {type: table, biases: [...]}

    Observation uncertainty is read directly from each normal-point record.
    """
    ensure_registered()
    from lunarops.classes.frames import ReferenceFrameSystem
    from lunarops.classes.observation import (
        LightTimeSolver,
        LlrObservationModel,
        LlrObservationProcessor,
        ObservationCatalogState,
        ObservationResolver,
    )

    assembly = resolve_observation_assembly(
        context,
        program_config,
        station_catalog=station_catalog,
        reflector_catalog=reflector_catalog,
    )
    program_config = assembly.program_config

    def cfg(category: str):
        return normalize_class_config(context.class_config(category, program_config))

    eph_cfg = cfg("ephemerides")
    if eph_cfg["type"].lower() != "calceph":
        raise ValueError(f"Only ephemerides type 'calceph' is available, got {eph_cfg['type']!r}")
    eop_cfg = cfg("earthRotation")

    ephemeris = context.create_class("ephemerides", eph_cfg, cache=True)
    earth_orientation_provider = context.create_class("earthRotation", eop_cfg, cache=True)
    frames = ReferenceFrameSystem(
        ephemeris=ephemeris,
        earth_orientation_provider=earth_orientation_provider,
    )
    factory_context = _ObservationDependencies(
        run_context=context,
        ephemeris=ephemeris,
        earth_orientation_provider=earth_orientation_provider,
        frames=frames,
        cache_namespace=f"observation:{id(ephemeris)}:{id(earth_orientation_provider)}",
    )
    station_displacement = factory_context.create_class(
        "stationDisplacement",
        normalize_class_config(context.class_config("stationDisplacement", program_config)),
        cache=True,
    )
    reflector_displacement = factory_context.create_class(
        "reflectorDisplacement",
        cfg("reflectorDisplacement"),
        cache=True,
    )
    solver = LightTimeSolver(
        frame_system=frames,
        gravitational_delay_model=factory_context.create_class("relativity", cfg("relativity"), cache=False),
        troposphere_delay_model=factory_context.create_class("troposphere", cfg("troposphere"), cache=True),
        station_displacement_model=station_displacement,
        reflector_displacement_model=reflector_displacement,
    )
    model_state = ObservationCatalogState(
        assembly.station_catalog,
        assembly.reflector_catalog,
    )
    resolver = ObservationResolver(model_state)
    range_bias_cfg = context.class_config("rangeBias", program_config)
    if range_bias_cfg is None:
        raise KeyError("Observation processing requires explicit 'rangeBias' in the program or globals config.")
    range_bias = factory_context.create_class(
        "rangeBias",
        normalize_class_config(range_bias_cfg),
        cache=True,
    )
    observation_model = LlrObservationModel(
        frame_system=frames,
        light_time_solver=solver,
        range_bias_model=range_bias,
    )
    processor = LlrObservationProcessor(
        resolver,
        observation_model,
    )
    return processor


__all__ = [
    "ObservationAssembly",
    "build_observation_processor",
    "ensure_registered",
    "resolve_observation_assembly",
    "validate_observation_config",
]
