"""Build and store LLR normal equations at one linearization point."""

from __future__ import annotations

from llrops.config.context import RunContext
from llrops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    build_processor,
    load_datasets,
    model_compatibility_fingerprint,
)
from llrops.programs.registry import ArtifactSlot, ProgramSpec, program
from llrops.programs.specs import PARAMETRIZED_OBSERVATION_KEYS


@program(
    ProgramSpec(
        name="LlrNormalEquations",
        summary="Build normal equations at one fixed LLR linearization.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),),
        outputs=(ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),),
        optional_keys=PARAMETRIZED_OBSERVATION_KEYS,
    )
)
def llr_normal_equations(config: dict, context: RunContext):
    from llrops.estimation.linearized_least_squares import (
        build_normal_equations_streaming,
    )

    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
    processor = build_processor(config, context)

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
        compatibility=model_compatibility_fingerprint(config, context),
    )

    out = context.resolve_path(config["outputFileNormalEquations"])
    normals.save(out)
    print(
        f"[LlrNormalEquations] {normals.obs_count} obs, "
        f"{len(names)} parameters -> {out}"
    )
    return normals


__all__ = ["llr_normal_equations"]
