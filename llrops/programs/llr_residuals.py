"""Compute typed LLR observation-result tables from canonical normal points."""

from __future__ import annotations

from typing import Dict

from llrops.config.context import RunContext
from llrops.llr_workflow import (
    build_processor,
    load_datasets,
    make_processing_options,
    output_level,
)
from llrops.programs.registry import ArtifactSlot, ProgramSpec, program
from llrops.programs.specs import OBSERVATION_PROGRAM_KEYS


@program(
    ProgramSpec(
        name="LlrResiduals",
        summary="Evaluate LLR O-C residuals and diagnostics.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),),
        outputs=(
            ArtifactSlot("outputFileObservationResults", "ObservationResultFile"),
        ),
        optional_keys=(
            *OBSERVATION_PROGRAM_KEYS,
            "outputLevel",
            "includeReflectorDesign",
        ),
    )
)
def llr_residuals(config: dict, context: RunContext):
    from llrops.fileio.observation_results import write_observation_results

    datasets = load_datasets(config, context)
    options = make_processing_options(config)
    table_level = output_level(config)

    runtime = context.runtime
    if runtime is not None and runtime.has_workers:
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
    else:
        processor = build_processor(config, context)
        results_by_file: Dict[str, list] = {
            source_name: processor.rows(dataset, options=options, level=table_level)
            for source_name, dataset in datasets.items()
        }

    output = context.resolve_path(config["outputFileObservationResults"])
    write_observation_results(results_by_file, output)
    total = sum(len(rows) for rows in results_by_file.values())
    print(
        f"[LlrResiduals] {total} normal point(s) over {len(results_by_file)} source(s) -> {output}"
    )
    return results_by_file


__all__ = ["llr_residuals"]
