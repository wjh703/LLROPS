"""Build and store LLR normal equations at one linearization point."""
from __future__ import annotations

from llrops.config.context import RunContext
from llrops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    build_processor,
    load_datasets,
)
from llrops.programs.registry import program


@program("LlrNormalEquations")
def llr_normal_equations(config: dict, context: RunContext):
    from llrops.estimation.linearized_least_squares import (
        build_normal_equations_streaming,
    )

    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
    processor = build_processor(config, context)

    try:
        equation_source = build_equation_source(config, context, datasets, processor)
        equations = equation_source(1)
        parametrization.setup(equations, processor.model_state)
        names = parametrization.parameter_names()
        normals = build_normal_equations_streaming(
            equations,
            parametrization,
            parameter_names=names,
            sources=sorted(datasets),
            ephemeris=processor.ephemeris_file,
        )
    finally:
        processor.close()

    out = context.resolve_path(config["outputNormals"])
    normals.save(out)
    print(
        f"[LlrNormalEquations] {normals.obs_count} obs, "
        f"{len(names)} parameters -> {out}"
    )
    return normals


__all__ = ["llr_normal_equations"]
