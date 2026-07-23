"""Convert normal-point inputs into one canonical LLROPS file."""
from __future__ import annotations

from llrops.config.context import RunContext
from llrops.programs.registry import program


@program("NormalPointsToLlrops")
def normal_points_to_llrops(config: dict, context: RunContext):
    """Convert one or more source files into one canonical LLROPS JSONL file."""
    from llrops.fileio.llrops_normal_point_file import write_llrops_npt
    from llrops.fileio.normal_point_inputs import (
        read_normal_points,
        resolve_normal_point_inputs,
    )
    from llrops.fileio.normal_points import combine_npt_datasets

    inputs = config.get("inputNormalPoints")
    if not inputs:
        raise ValueError("inputNormalPoints is required")
    if not config.get("outputFile"):
        raise ValueError("outputFile is required")
    output_path = context.resolve_path(config["outputFile"])
    values = inputs if isinstance(inputs, list) else [inputs]
    paths = [
        path
        for path in resolve_normal_point_inputs(
            [context.resolve_path(value) for value in values]
        )
        if path.resolve() != output_path.resolve()
    ]
    if not paths:
        raise FileNotFoundError(
            f"No supported normal-point files found under {inputs!r}"
        )
    mini_io_log = (
        context.resolve_path(config["miniIoLog"])
        if config.get("miniIoLog")
        else None
    )
    datasets = [
        read_normal_points(path, mini_io_log_path=mini_io_log) for path in paths
    ]
    combined = combine_npt_datasets(
        datasets,
        name=str(config.get("datasetName", "normal-points")),
    )
    output = write_llrops_npt(combined, output_path)
    print(
        f"[NormalPointsToLlrops] wrote {len(combined.records)} record(s) -> {output}"
    )
    return str(output)


__all__ = ["normal_points_to_llrops"]
