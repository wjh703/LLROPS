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
uncertaintyModel      : wrms-table | table
parametrization        : reflectorPosition | stationRangeBias   (registered in their modules)

``RunContext.create_class(..., cache=True)`` is intentionally used here for
immutable/heavy backends such as CALCEPH and Earth-orientation sources.  Mutable model state
(catalog coordinates, station-bias values, future EOP/orbit corrections) stays
inside the returned ``LlrObservationProcessor`` instance.
"""
from __future__ import annotations

from pathlib import Path

from llrops.config.registry import register_factory, normalize_class_config



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
    from llrops.classes.uncertainty.wrms_table import (
        WrmsUncertaintyTable,
        builtin_wrms_uncertainty_table,
        load_wrms_uncertainty_table,
    )
    from llrops.classes.uncertainty.models import WrmsTableUncertainty

    def _calceph(cfg: dict, ctx):
        if "file" not in cfg:
            raise ValueError("ephemerides/calceph requires 'file'.")
        return load_calceph_ephemeris(
            _resolve_optional_path(ctx, cfg["file"]),
            longitude_libration=cfg.get("longitudeLibrationCorrection", "none"),
        )

    def _iers_c04(cfg: dict, ctx):
        mpi_resources = getattr(ctx, "shared", {}).get("mpiResources", {})
        payload = mpi_resources.get("earthRotation")
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
        lambda cfg, ctx: Iers2010ShapiroDelay(ephemeris=_shared_ephemeris(ctx)),
    )

    def _shared_earth_orientation(ctx):
        try:
            return ctx.shared["earthOrientation"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "stationDisplacement requires the shared Earth-orientation source."
            ) from exc

    def _shared_ephemeris(ctx):
        try:
            return ctx.shared["ephemeris"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "reflectorDisplacement requires the shared ephemeris."
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
            earth_orientation=_shared_earth_orientation(ctx),
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
        lambda cfg, ctx: Iers2010PoleTide(earth_orientation=_shared_earth_orientation(ctx)),
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
            ephemeris=_shared_ephemeris(ctx),
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
    register_factory("rangeBias", "table", _range_bias_table)

    def _builtin_wrms_uncertainty(cfg: dict, ctx) -> WrmsTableUncertainty:
        if "model" not in cfg:
            raise ValueError("uncertaintyModel/wrms-table requires explicit 'model'.")
        return WrmsTableUncertainty(builtin_wrms_uncertainty_table(cfg["model"]))

    def _table_wrms_uncertainty(cfg: dict, ctx) -> WrmsTableUncertainty:
        has_file = "file" in cfg
        has_uncertainties = "uncertainties" in cfg
        if has_file == has_uncertainties:
            raise ValueError("uncertaintyModel/table requires exactly one of 'file' or 'uncertainties'.")
        if has_file:
            table = load_wrms_uncertainty_table(_resolve_optional_path(ctx, cfg["file"]))
        else:
            table = WrmsUncertaintyTable.from_mapping(cfg)
        return WrmsTableUncertainty(table)

    register_factory("uncertaintyModel", "wrms-table", _builtin_wrms_uncertainty)
    register_factory("uncertaintyModel", "table", _table_wrms_uncertainty)

    # Parametrizations register themselves on import.
    import llrops.classes.parametrization.reflector_position  # noqa: F401
    import llrops.classes.parametrization.station_range_bias  # noqa: F401


_REGISTERED = False


def ensure_registered() -> None:
    global _REGISTERED
    if not _REGISTERED:
        _register_all()
        _REGISTERED = True


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
        uncertaintyModel:      {type: wrms-table, model: default} | {type: table, file: ...} | {type: table, uncertainties: [...]}
    """
    ensure_registered()
    from llrops.classes.frames import ReferenceFrameSystem
    from llrops.classes.observation import (
        LightTimeSolver,
        LlrObservationModel,
        LlrObservationProcessor,
        LlrObservationReducer,
        LlrObservationResultBuilder,
        ObservationResolver,
    )
    from llrops.classes.uncertainty.models import MiniUncertainty, UncertaintyKind
    from llrops.fileio.catalogs import load_station_catalog, load_reflector_catalog

    def cfg(category: str):
        return normalize_class_config(context.class_config(category, program_config))

    eph_cfg = cfg("ephemerides")
    if eph_cfg["type"].lower() != "calceph":
        raise ValueError(f"Only ephemerides type 'calceph' is available, got {eph_cfg['type']!r}")
    eop_cfg = cfg("earthRotation")

    def catalog_source(name: str):
        value = program_config.get(name, context.global_class_configs.get(name))
        if isinstance(value, str) and value not in ("builtin", ""):
            return context.resolve_path(value)
        return value

    station_catalog = station_catalog or load_station_catalog(catalog_source("stationCatalog"))
    reflector_catalog = reflector_catalog or load_reflector_catalog(catalog_source("reflectorCatalog"))
    context.shared["stationCatalog"] = station_catalog
    context.shared["reflectorCatalog"] = reflector_catalog

    ephemeris = context.create_class("ephemerides", eph_cfg, cache=True)
    earth_orientation = context.create_class("earthRotation", eop_cfg, cache=True)
    context.shared["ephemeris"] = ephemeris
    context.shared["earthOrientation"] = earth_orientation

    frames = ReferenceFrameSystem(
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
        owns_ephemeris=False,
    )
    station_displacement = context.create_class(
        "stationDisplacement",
        normalize_class_config(context.class_config("stationDisplacement", program_config)),
        cache=True,
    )
    reflector_displacement = context.create_class(
        "reflectorDisplacement",
        cfg("reflectorDisplacement"),
        cache=True,
    )
    solver = LightTimeSolver(
        frames,
        gravitational_delay=context.create_class("relativity", cfg("relativity"), cache=False),
        troposphere_delay=context.create_class("troposphere", cfg("troposphere"), cache=True),
        station_displacement=station_displacement,
        reflector_displacement=reflector_displacement,
    )
    model = LlrObservationModel(frames, solver)
    resolver = ObservationResolver(station_catalog, reflector_catalog)
    range_bias_cfg = context.class_config("rangeBias", program_config)
    if range_bias_cfg is None:
        raise KeyError("Observation processing requires explicit 'rangeBias' in the program or globals config.")
    range_bias = context.create_class(
        "rangeBias",
        normalize_class_config(range_bias_cfg),
        cache=True,
    )
    uncertainty_cfg = context.class_config("uncertaintyModel", program_config)
    if uncertainty_cfg is None:
        raise KeyError("Observation processing requires explicit 'uncertaintyModel' in the program or globals config.")
    wrms_uncertainty = context.create_class(
        "uncertaintyModel",
        normalize_class_config(uncertainty_cfg),
        cache=True,
    )
    reducer = LlrObservationReducer(
        ephemeris=ephemeris,
        range_bias=range_bias,
        uncertainty_models={
            UncertaintyKind.WRMS_TABLE: wrms_uncertainty,
            UncertaintyKind.MINI: MiniUncertainty(),
        },
    )
    processor = LlrObservationProcessor(
        resolver=resolver,
        model=model,
        reducer=reducer,
        result_builder=LlrObservationResultBuilder(),
    )
    context.shared["observationModel"] = model
    context.shared["observationProcessor"] = processor
    return processor
