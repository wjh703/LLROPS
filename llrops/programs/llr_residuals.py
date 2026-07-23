"""Compute LLR residuals from normal-point inputs.

``LlrResiduals`` replaces ``run_llr_np_oc.py``.  Config keys::

    - program: LlrResiduals
      inputNormalPoints: [dir_or_file, ...]     # MINI, CRD, and/or LLROPS JSONL
      combineInputs: false                      # merge all inputs into one dataset
      startTime: null / "2020-01-01T00:00:00"
      endTime:   null
      stationName: null                         # catalog override
      reflectorName: null
      outputLevel: standard | full             # compact O-C table | diagnostics
      includeReflectorDesign: false
      minElevationDeg: 0.0
      showProgress: true
      outputCsv: oc.csv                         # grouped over inputs
      outputJson: null
      # model classes: ephemerides / earthRotation / troposphere / relativity /
      # stationDisplacement / reflectorDisplacement / rangeBias - from globals
      # unless overridden here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from llrops.config.context import RunContext
from llrops.programs.registry import program


def load_datasets(config: dict, context: RunContext):
    """Shared input handling: returns ``{source_name: NptDataset}``."""
    from llrops.fileio.normal_point_inputs import (
        read_normal_points,
        resolve_normal_point_inputs,
    )
    from llrops.fileio.normal_points import combine_npt_datasets

    inputs = config.get("inputNormalPoints")
    if not inputs:
        raise ValueError("inputNormalPoints is required")
    input_values = inputs if isinstance(inputs, list) else [inputs]
    resolved_inputs = [context.resolve_path(item) for item in input_values]
    input_files = resolve_normal_point_inputs(resolved_inputs)
    mini_io_log = context.resolve_path(config["miniIoLog"]) if config.get("miniIoLog") else None
    if not input_files:
        raise FileNotFoundError(f"No supported normal-point files found under {inputs!r}")

    datasets = {}
    for path in input_files:
        dataset = read_normal_points(path, mini_io_log_path=mini_io_log)
        start, end = config.get("startTime"), config.get("endTime")
        if start or end:
            dataset = dataset.filter_time(start, end)
        if dataset.records:
            datasets[Path(path).stem] = dataset

    if config.get("combineInputs"):
        combined = combine_npt_datasets(list(datasets.values()))
        datasets = {config.get("combinedName", "combined"): combined}

    # Normal-point indices are source-independent and globally unique across
    # all input files in processing order. Outlier bookkeeping uses only this
    # row number, never a file name.
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
    from llrops.classes.observation_factory import validate_observation_config
    from llrops.classes.observation import ObservationProcessingOptions

    validate_observation_config(config)
    include_design = bool(include_design or config.get("includeReflectorDesign", False))
    return ObservationProcessingOptions(
        station_name=config.get("stationName"),
        reflector_name=config.get("reflectorName"),
        min_elevation_deg=float(config.get("minElevationDeg", 0.0)),
        include_reflector_position_partial=include_design,
        show_progress=bool(config.get("showProgress", True)),
    )


def output_level(config: dict, *, include_design: bool = False):
    from llrops.classes.observation import ObservationOutputLevel

    if include_design:
        return ObservationOutputLevel.FULL
    return ObservationOutputLevel.parse(config.get("outputLevel", "standard"))


@program("LlrResiduals")
def llr_residuals(config: dict, context: RunContext):
    from llrops.fileio import observation_result_writer

    datasets = load_datasets(config, context)
    options = make_processing_options(config)
    table_level = output_level(config)

    runtime = context.shared.get("mpi")
    if runtime is not None and runtime.has_workers:
        # MPI master-worker (v24 run_llr_np_oc_mpi.py): rank 0 loads/writes,
        # workers hold their own processor and compute NptRecord chunks.
        from llrops.parallel.mpi import make_observation_spec, mpi_observation_rows

        spec = make_observation_spec(config, context)
        results_by_file = mpi_observation_rows(
            runtime,
            spec,
            datasets,
            options,
            output_level=table_level.value,
            chunksize=int((config.get("mpi") or {}).get("chunksize", 8)),
            progress_desc="O-C normal points",
            quiet=not bool(config.get("showProgress", True)),
        )
        total = sum(len(rows) for rows in results_by_file.values())
    else:
        processor = build_processor(config, context)
        results_by_file: Dict[str, list] = {}
        total = 0
        try:
            for source_name, dataset in datasets.items():
                results = processor.process(dataset, options=options)
                results_by_file[source_name] = results
                total += len(results)
        finally:
            processor.close()

    if config.get("outputCsv"):
        observation_result_writer.write_csv_grouped(
            results_by_file,
            context.resolve_path(config["outputCsv"]),
            level=table_level,
        )
    if config.get("outputJson"):
        observation_result_writer.write_json_grouped(
            results_by_file,
            context.resolve_path(config["outputJson"]),
            level=table_level,
        )
    print(f"[LlrResiduals] {total} normal points over {len(results_by_file)} source file(s)")
    return results_by_file


__all__ = [
    "build_processor",
    "llr_residuals",
    "load_datasets",
    "make_processing_options",
    "output_level",
]
