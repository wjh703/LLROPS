"""Shared application workflow for LLR programs."""
from __future__ import annotations

from pathlib import Path

from llrops.config.context import RunContext


def load_datasets(config: dict, context: RunContext):
    from llrops.fileio.normal_point_inputs import (
        read_normal_points,
        resolve_normal_point_inputs,
    )
    from llrops.fileio.normal_points import combine_npt_datasets

    inputs = config.get("inputNormalPoints")
    if not inputs:
        raise ValueError("inputNormalPoints is required")
    input_values = inputs if isinstance(inputs, list) else [inputs]
    input_files = resolve_normal_point_inputs(
        [context.resolve_path(item) for item in input_values]
    )
    mini_io_log = (
        context.resolve_path(config["miniIoLog"])
        if config.get("miniIoLog")
        else None
    )
    if not input_files:
        raise FileNotFoundError(
            f"No supported normal-point files found under {inputs!r}"
        )

    datasets = {}
    for path in input_files:
        dataset = read_normal_points(path, mini_io_log_path=mini_io_log)
        start, end = config.get("startTime"), config.get("endTime")
        if start or end:
            dataset = dataset.filter_time(start, end)
        if dataset.records:
            datasets[Path(path).stem] = dataset

    if config.get("combineInputs"):
        datasets = {
            config.get("combinedName", "combined"): combine_npt_datasets(
                list(datasets.values())
            )
        }

    next_index = 0
    for dataset in datasets.values():
        dataset.assign_indices(start=next_index)
        next_index += len(dataset.records)
    if not datasets:
        raise ValueError("No normal points remain after time filtering.")
    return datasets


def build_processor(config: dict, context: RunContext):
    from llrops.classes.observation_factory import build_observation_processor

    return build_observation_processor(context, config)


def make_processing_options(config: dict, *, include_design: bool = False):
    from llrops.classes.observation import ObservationProcessingOptions
    from llrops.classes.observation_factory import validate_observation_config

    validate_observation_config(config)
    return ObservationProcessingOptions(
        station_name=config.get("stationName"),
        reflector_name=config.get("reflectorName"),
        min_elevation_deg=float(config.get("minElevationDeg", 0.0)),
        include_reflector_position_partial=bool(
            include_design or config.get("includeReflectorDesign", False)
        ),
        show_progress=bool(config.get("showProgress", True)),
    )


def output_level(config: dict, *, include_design: bool = False):
    from llrops.classes.observation import ObservationOutputLevel

    if include_design:
        return ObservationOutputLevel.FULL
    return ObservationOutputLevel.parse(config.get("outputLevel", "standard"))


def build_parametrization(config: dict, context: RunContext):
    from llrops.classes.observation_factory import ensure_registered
    from llrops.classes.parametrization.base import ParametrizationList
    from llrops.config.registry import create_list

    ensure_registered()
    blocks = create_list("parametrization", config.get("parametrization"), context)
    if not blocks:
        raise ValueError("At least one parametrization block is required.")
    return ParametrizationList(blocks)


def build_equation_source(config, context, datasets, processor):
    """Return a closure that relinearizes all observations per iteration."""
    options = make_processing_options(config, include_design=True)
    runtime = context.runtime
    use_mpi = runtime is not None and runtime.has_workers
    if use_mpi:
        from llrops.parallel.mpi import (
            make_observation_spec,
            mpi_observation_equations,
            snapshot_catalog_state,
        )

        spec = make_observation_spec(
            config,
            context,
            station_catalog=processor.station_catalog,
            reflector_catalog=processor.reflector_catalog,
        )
        chunksize = int((config.get("mpi") or {}).get("chunksize", 8))

    def equation_source(iteration: int):
        if use_mpi:
            equations_by_source = mpi_observation_equations(
                runtime,
                spec,
                datasets,
                options,
                chunksize=chunksize,
                catalog_state=snapshot_catalog_state(processor.model_state),
                progress_desc=f"linearization {iteration}",
                quiet=not bool(config.get("showProgress", True)),
            )
        else:
            iteration_options = options.with_progress(f"linearization {iteration}")
            equations_by_source = {
                source_name: processor.equations(dataset, options=iteration_options)
                for source_name, dataset in datasets.items()
            }
        return [
            equation
            for equations in equations_by_source.values()
            for equation in equations
        ]

    return equation_source


__all__ = [
    "build_equation_source",
    "build_parametrization",
    "build_processor",
    "load_datasets",
    "make_processing_options",
    "output_level",
]
