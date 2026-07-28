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

from typing import Dict

from llrops.config.context import RunContext
from llrops.llr_workflow import (
    build_processor,
    load_datasets,
    make_processing_options,
    output_level,
)
from llrops.programs.registry import program


@program("LlrResiduals")
def llr_residuals(config: dict, context: RunContext):
    from llrops.fileio import observation_result_writer

    datasets = load_datasets(config, context)
    options = make_processing_options(config)
    table_level = output_level(config)

    runtime = context.runtime
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
        for source_name, dataset in datasets.items():
            results = processor.rows(
                dataset,
                options=options,
                level=table_level,
            )
            results_by_file[source_name] = results
            total += len(results)

    if config.get("outputCsv"):
        observation_result_writer.write_csv_grouped(
            results_by_file,
            context.resolve_path(config["outputCsv"]),
        )
    if config.get("outputJson"):
        observation_result_writer.write_json_grouped(
            results_by_file,
            context.resolve_path(config["outputJson"]),
        )
    print(f"[LlrResiduals] {total} normal points over {len(results_by_file)} source file(s)")
    return results_by_file


__all__ = ["llr_residuals"]
