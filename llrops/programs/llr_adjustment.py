"""Programs: LlrAdjustment, LlrNormalEquations, NormalsCombineSolve.

``LlrAdjustment`` is the generalized, parametrization-driven estimator::

    - program: LlrAdjustment
      inputNormalPoints: [...]
      parametrization:
        - {type: reflectorPosition, reflectors: [apollo15]}
        - {type: stationRangeBias}
        # future: {type: stationPosition}, {type: eop, temporal: ...},
        #         {type: lunarLoveNumbers}, {type: lunarOrbitState}
      adjustment:
        maximumLinearizations: 20
        parameterUpdateFactor: 1.0
        updateToleranceM: 1.0e-3
        prefitGrossThresholdM: 20.0
        prefitGrossThresholdByStationM: {APOLLO: 10.0, GRASSE: 30.0}
      outputJson: adjustment.json
      outputNormals: normals/llr        # optional final-state normals

``LlrNormalEquations`` builds and stores normal equations at the current
linearization point without solving — the GROOPS pattern for combining data
sets (per station, per epoch block, later per technique)::

    - program: LlrNormalEquations
      inputNormalPoints: [...]
      parametrization: [...]
      outputNormals: normals/apollo15_2020

``NormalsCombineSolve`` adds normal-equation files by parameter name and
solves them::

    - program: NormalsCombineSolve
      inputNormals: [normals/a, normals/b]
      outputSolutionJson: solution.json
"""
from __future__ import annotations

import json

import numpy as np

from llrops.config.context import RunContext
from llrops.config.registry import create_list
from llrops.programs.base import program
from llrops.programs.llr_residuals import load_datasets, build_processor, make_processing_options


def _build_equation_source(config, context, datasets, processor):
    """Equation-source closure: recompute typed observations each iteration.

    Under ``--mpi`` the typed results of every iteration are computed by worker ranks;
    the current linearization point (reflector catalog updated by the
    parametrizations through ``context.shared``) is snapshotted and shipped
    with each task, so relinearization matches the serial path.
    """
    options = make_processing_options(config, include_design=True)
    runtime = context.shared.get("mpi")
    progress_prefix = (
        "linearization"
        if config.get("program") == "LlrAdjustment"
        else "adjustment iter"
    )
    use_mpi = runtime is not None and runtime.has_workers
    if use_mpi:
        from llrops.parallel.mpi import make_observation_spec, mpi_observation_results, snapshot_catalog_state

        spec = make_observation_spec(
            config,
            context,
            station_catalog=processor.station_catalog,
            reflector_catalog=processor.reflector_catalog,
        )
        chunksize = int((config.get("mpi") or {}).get("chunksize", 8))

    def equation_source(iteration: int):
        if use_mpi:
            results_by_source = mpi_observation_results(
                runtime,
                spec,
                datasets,
                options,
                chunksize=chunksize,
                catalog_state=snapshot_catalog_state(context),
                progress_desc=f"{progress_prefix} {iteration}",
                quiet=not bool(config.get("showProgress", True)),
            )
        else:
            iteration_options = options.with_progress(f"{progress_prefix} {iteration}")
            results_by_source = {
                source_name: processor.process(
                    dataset, source_name=source_name, options=iteration_options
                )
                for source_name, dataset in datasets.items()
            }
        return [
            result.to_equation()
            for results in results_by_source.values()
            for result in results
        ]

    return equation_source


def _build_parametrization(config: dict, context: RunContext):
    from llrops.classes.builders import ensure_registered
    from llrops.classes.parametrization.base import ParametrizationList

    ensure_registered()
    blocks = create_list("parametrization", config.get("parametrization"), context)
    if not blocks:
        raise ValueError("At least one parametrization block is required.")
    return ParametrizationList(blocks)



@program("LlrAdjustment")
def llr_adjustment(config: dict, context: RunContext):
    """Run nonlinear LLR adjustment with robust weights and VCE."""
    from llrops.estimation.adjustment_config import parse_adjustment_plan
    from llrops.estimation.adjustment_solver import LlrAdjustmentSolver

    plan = parse_adjustment_plan(config)
    options = plan.options
    datasets = load_datasets(config, context)
    parametrization = _build_parametrization(config, context)
    processor = build_processor(config, context)
    active_stage = {"name": "joint"}

    def report_iteration(item):
        print(
            "[LlrAdjustment:HelmertVCE] "
            f"stage={active_stage['name']} "
            f"linearization={item.linearization_iteration} "
            f"stochastic={item.stochastic_iteration} "
            f"elapsed={item.elapsed_seconds:.3f}s "
            f"scaleLogTarget={item.maximum_scale_log_target_change:.3e} "
            f"factorTargetQ={item.robust_factor_target_change_quantile:.3e} "
            f"activeSetChange={item.active_set_change_fraction:.3e} "
            f"targetRejected={item.target_rejected_observation_count} "
            f"active={item.active_observation_count} "
            f"rejected={item.rejected_observation_count} "
            f"converged={item.stochastic_converged}",
            flush=True,
        )

    stage_results = []
    warm_start_stochastic = plan.warm_start_stochastic_model_across_stages
    previous_scales = {}
    previous_factors = {}
    try:
        equation_source = _build_equation_source(config, context, datasets, processor)
        for stage in plan.stages:
            stage_name = stage.name
            active_stage["name"] = stage_name
            stage_parametrization = (
                parametrization
                if not stage.parametrizations
                else parametrization.select_blocks(stage.parametrizations)
            )
            stage_options = stage.apply(options)
            result = LlrAdjustmentSolver(
                equation_source=equation_source,
                parametrization=stage_parametrization,
                options=stage_options,
                context=context,
                initial_scales=(previous_scales if warm_start_stochastic else None),
                initial_factors=(previous_factors if warm_start_stochastic else None),
                iteration_callback=(
                    report_iteration if bool(config.get("showProgress", True)) else None
                ),
            ).run()
            previous_scales = dict(result.scales)
            previous_factors = dict(result.robust_factors)
            performance = result.summary["performance_seconds"]
            print(
                "[LlrAdjustment:Performance] "
                f"stage={stage_name} backend={result.settings['linearization_backend']} "
                f"cache={performance['cache_build']:.3f}s "
                f"solve={performance['normal_solve']:.3f}s "
                f"leverage={performance['leverage']:.3f}s "
                f"vce={performance['vce']:.3f}s "
                f"warmScales={result.settings['warm_started_scale_count']} "
                f"warmFactors={result.settings['warm_started_factor_count']}",
                flush=True,
            )
            print(
                "[LlrAdjustment:UncertaintyQC] "
                f"stage={stage_name} action=floor "
                f"floored={result.summary['uncertainty_sigma_floored_count']} "
                f"retainedFloored={result.summary['retained_uncertainty_sigma_floored_count']}",
                flush=True,
            )
            stage_results.append(
                {
                    "name": stage_name,
                    "parametrizations": [
                        type(block).__name__ for block in stage_parametrization.blocks
                    ],
                    "summary": result.summary,
                    "state": result.state,
                }
            )
    finally:
        processor.close()

    if config.get("outputJson"):
        path = context.resolve_path(config["outputJson"])
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["processing_steps"] = stage_results
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if config.get("outputNormals") and result.normals is not None:
        result.normals.save(context.resolve_path(config["outputNormals"]))
    print(
        f"[LlrAdjustment] converged={result.converged} "
        f"linearizations={len(result.linearizations)} "
        f"stochasticIterations={len(result.iterations)} components={len(result.scales)}"
    )
    return result


@program("LlrNormalEquations")
def llr_normal_equations(config: dict, context: RunContext):
    from llrops.estimation.normal_equation_engine import build_normal_equations_streaming

    datasets = load_datasets(config, context)
    parametrization = _build_parametrization(config, context)
    processor = build_processor(config, context)

    try:
        equation_source = _build_equation_source(config, context, datasets, processor)
        equations = equation_source(1)
        parametrization.setup(equations, context)
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
    print(f"[LlrNormalEquations] {normals.obs_count} obs, {len(names)} parameters -> {out}")
    return normals


@program("NormalsCombineSolve")
def normals_combine_solve(config: dict, context: RunContext):
    from llrops.fileio.normal_equations import NormalEquations

    stems = config.get("inputNormals") or []
    if not stems:
        raise ValueError("inputNormals is required")
    total = NormalEquations.load(context.resolve_path(stems[0]))
    for stem in stems[1:]:
        total = total.add(NormalEquations.load(context.resolve_path(stem)))

    x, Qxx, sigma0 = total.solve()
    solution = {
        "sigma0_post": sigma0,
        "obs_count": total.obs_count,
        "parameters": [
            {
                "name": str(name),
                "estimate": float(xi),
                "cofactor_sigma": float(np.sqrt(Qxx[i, i])),
                "formal_sigma": (
                    None
                    if sigma0 is None
                    else float(sigma0 * np.sqrt(Qxx[i, i]))
                ),
            }
            for i, (name, xi) in enumerate(zip(total.parameter_names, x))
        ],
    }
    if config.get("outputSolutionJson"):
        path = context.resolve_path(config["outputSolutionJson"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(solution, indent=2), encoding="utf-8")
    if config.get("outputNormals"):
        total.save(context.resolve_path(config["outputNormals"]))
    sigma0_text = "undefined" if sigma0 is None else f"{sigma0:.4f}"
    print(
        f"[NormalsCombineSolve] solved {len(total.parameter_names)} parameters, "
        f"sigma0={sigma0_text}"
    )
    return solution
