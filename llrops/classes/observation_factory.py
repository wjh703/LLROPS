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
stationDisplacement    : none | sum | iers2010SolidEarthTide | iers2010PoleTide | iers2010OceanPoleTide
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
from typing import Mapping

from llrops.config.registry import register_factory, normalize_class_config


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
    station_catalog: Mapping[str, object]
    reflector_catalog: Mapping[str, object]


class _ObservationFactoryContext:
    __slots__ = ("run_context", "ephemeris", "earth_orientation", "cache_namespace")

    def __init__(
        self,
        run_context: object,
        ephemeris: object,
        earth_orientation: object,
        cache_namespace: str,
    ) -> None:
        self.run_context = run_context
        self.ephemeris = ephemeris
        self.earth_orientation = earth_orientation
        self.cache_namespace = cache_namespace

    @property
    def mpi_resources(self):
        return self.run_context.mpi_resources

    def resolve_path(self, value):
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


def _resolve_optional_path(ctx, value):
    if value in (None, ""):
        return None
    if ctx is not None and hasattr(ctx, "resolve_path"):
        return ctx.resolve_path(value)
    return Path(str(value)).expanduser()


def _register_all() -> None:
    # Imports are local so that merely importing the registry does not load
    # CALCEPH or optional physical-model backends.
    from llrops.classes.delays import (
        Iers2010MendesPavlisTroposphere,
        Iers2010ShapiroDelay,
        ZeroGravitationalDelay,
        ZeroTroposphereDelay,
    )
    from llrops.classes.displacement import (
        CompositeStationDisplacement,
        Iers2010OceanPoleTide,
        Iers2010PoleTide,
        Iers2010SolidEarthTide,
        LunarSolidTide,
        OceanPoleTideGrid,
        ZeroReflectorDisplacement,
        ZeroStationDisplacement,
    )
    from llrops.classes.ephemerides import load_calceph_ephemeris
    from llrops.classes.frames import C04EarthOrientation, load_iers_c04
    from llrops.classes.range_bias.models import (
        TableRangeBiasModel,
        ZeroRangeBiasModel,
    )
    from llrops.classes.range_bias.table import (
        RangeBiasTable,
        builtin_range_bias_table,
        load_range_bias_table,
    )

    def _calceph(cfg: dict, ctx):
        if "file" not in cfg:
            raise ValueError("ephemerides/calceph requires 'file'.")
        return load_calceph_ephemeris(
            _resolve_optional_path(ctx, cfg["file"]),
            longitude_libration=cfg.get("longitudeLibrationCorrection", "none"),
        )

    def _iers_c04(cfg: dict, ctx):
        payload = ctx.mpi_resources.get("earthRotation")
        if payload is not None:
            return C04EarthOrientation.from_mpi_payload(payload)
        if "file" not in cfg:
            raise ValueError("earthRotation/iersC04 requires 'file'.")
        return load_iers_c04(
            _resolve_optional_path(ctx, cfg["file"]),
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
        try:
            return ctx.earth_orientation
        except AttributeError as exc:
            raise RuntimeError(
                "stationDisplacement requires an explicit Earth-orientation source."
            ) from exc

    def _required_ephemeris(ctx):
        try:
            return ctx.ephemeris
        except AttributeError as exc:
            raise RuntimeError(
                "reflectorDisplacement requires an explicit ephemeris."
            ) from exc

    def _station_sum(cfg: dict, ctx) -> CompositeStationDisplacement:
        components_cfg = cfg.get("components", [])
        if isinstance(components_cfg, (str, dict)):
            components_cfg = [components_cfg]
        components = tuple(
            ctx.create_class("stationDisplacement", component, cache=True)
            for component in components_cfg
        )
        return CompositeStationDisplacement(components)

    def _station_ocean_pole_tide(cfg: dict, ctx) -> Iers2010OceanPoleTide:
        coefficient_file = _resolve_optional_path(ctx, cfg.get("coefficientFile"))
        if coefficient_file is None:
            raise ValueError(
                "stationDisplacement/iers2010OceanPoleTide requires 'coefficientFile'."
            )
        return Iers2010OceanPoleTide(
            grid=OceanPoleTideGrid(coefficient_file),
            earth_orientation=_required_earth_orientation(ctx),
        )

    register_factory(
        "stationDisplacement",
        "none",
        lambda cfg, ctx: ZeroStationDisplacement(),
    )
    register_factory("stationDisplacement", "sum", _station_sum)
    register_factory(
        "stationDisplacement",
        "iers2010solidearthtide",
        lambda cfg, ctx: Iers2010SolidEarthTide(
            sampling_interval_s=float(cfg.get("samplingIntervalSeconds", 60.0))
        ),
    )
    register_factory(
        "stationDisplacement",
        "iers2010poletide",
        lambda cfg, ctx: Iers2010PoleTide(
            earth_orientation=_required_earth_orientation(ctx)
        ),
    )
    register_factory(
        "stationDisplacement",
        "iers2010oceanpoletide",
        _station_ocean_pole_tide,
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
            table = load_range_bias_table(_resolve_optional_path(ctx, cfg["file"]))
        else:
            table = RangeBiasTable.from_mapping(cfg)
        return TableRangeBiasModel(table)

    register_factory("rangeBias", "none", lambda cfg, ctx: ZeroRangeBiasModel())
    register_factory("rangeBias", "inpop21", lambda cfg, ctx: TableRangeBiasModel(builtin_range_bias_table("inpop21")))
    register_factory("rangeBias", "inpop21a", lambda cfg, ctx: TableRangeBiasModel(builtin_range_bias_table("inpop21a")))
    register_factory("rangeBias", "table", _range_bias_table)

    # Parametrizations register themselves on import.
    import llrops.classes.parametrization.reflector_position  # noqa: F401
    import llrops.classes.parametrization.station_range_bias  # noqa: F401


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
    from llrops.fileio.catalogs import load_reflector_catalog, load_station_catalog

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

    stations = (
        load_station_catalog(catalog_source("stationCatalog"))
        if station_catalog is None
        else station_catalog
    )
    reflectors = (
        load_reflector_catalog(catalog_source("reflectorCatalog"))
        if reflector_catalog is None
        else reflector_catalog
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
    from llrops.classes.frames import ReferenceFrameSystem
    from llrops.classes.observation import (
        LightTimeSolver,
        LlrObservationModel,
        LlrObservationProcessor,
        LlrObservationReducer,
        ObservationModelState,
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
    earth_orientation = context.create_class("earthRotation", eop_cfg, cache=True)
    factory_context = _ObservationFactoryContext(
        run_context=context,
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
        cache_namespace=f"observation:{id(ephemeris)}:{id(earth_orientation)}",
    )

    frames = ReferenceFrameSystem(
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
        owns_ephemeris=False,
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
        frames,
        gravitational_delay=factory_context.create_class(
            "relativity", cfg("relativity"), cache=False
        ),
        troposphere_delay=factory_context.create_class(
            "troposphere", cfg("troposphere"), cache=True
        ),
        station_displacement=station_displacement,
        reflector_displacement=reflector_displacement,
    )
    model = LlrObservationModel(frames, solver)
    model_state = ObservationModelState.from_catalogs(
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
    reducer = LlrObservationReducer(
        ephemeris=ephemeris,
        range_bias=range_bias,
    )
    processor = LlrObservationProcessor(
        resolver=resolver,
        model=model,
        reducer=reducer,
    )
    return processor


__all__ = [
    "ObservationAssembly",
    "build_observation_processor",
    "ensure_registered",
    "resolve_observation_assembly",
    "validate_observation_config",
]
