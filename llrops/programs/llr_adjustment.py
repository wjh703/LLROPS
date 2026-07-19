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
        maxIterations: 20
        damping: 1.0
        updateToleranceM: 1.0e-3
        wrmsToleranceM: 1.0e-4
        prefitGrossThresholdM: 20.0
        prefitGrossThresholdByStationM: {APOLLO: 10.0, GRASSE: 30.0}
        outlier: {enabled: true, sigmaFactor: 3.0, reentry: true}
      outputJson: adjustment.json
      outputNormals: normals/llr        # optional last-iteration normals

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
from dataclasses import replace
from typing import Dict, List

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

        spec = make_observation_spec(config, context, datasets)
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


@program("LlrLeastSquaresAdjustment")
def llr_least_squares_adjustment(config: dict, context: RunContext):
    from llrops.estimation.adjustment import AdjustmentOptions, LeastSquaresAdjustment

    datasets = load_datasets(config, context)
    processor = build_processor(config, context)
    parametrization = _build_parametrization(config, context)

    adj = config.get("adjustment") or {}
    outlier = adj.get("outlier") or {}
    options = AdjustmentOptions(
        max_iterations=int(adj.get("maxIterations", 20)),
        damping=float(adj.get("damping", 1.0)),
        update_tolerance_m=float(adj.get("updateToleranceM", 1.0e-3)),
        wrms_tolerance_m=float(adj.get("wrmsToleranceM", 1.0e-4)),
        prefit_gross_threshold_m=adj.get("prefitGrossThresholdM", 20.0),
        prefit_gross_threshold_by_station_m=adj.get("prefitGrossThresholdByStationM"),
        enable_outlier_rejection=bool(outlier.get("enabled", True)),
        outlier_sigma_factor=float(outlier.get("sigmaFactor", 3.0)),
        allow_outlier_reentry=bool(outlier.get("reentry", True)),
    )

    try:
        estimator = LeastSquaresAdjustment(
            equation_source=_build_equation_source(config, context, datasets, processor),
            parametrization=parametrization,
            options=options,
            context=context,
        )
        result = estimator.run()
    finally:
        processor.close()

    if config.get("outputJson"):
        path = context.resolve_path(config["outputJson"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    if config.get("outputNormals") and result.normals is not None:
        result.normals.save(context.resolve_path(config["outputNormals"]))
    print(
        f"[LlrLeastSquaresAdjustment] converged={result.converged} "
        f"iterations={len(result.iterations)} parameters={len(result.parameter_names)}"
    )
    return result


@program("LlrAdjustment")
def llr_adjustment(config: dict, context: RunContext):
    """Run nonlinear LLR adjustment with robust weights and VCE."""
    from llrops.estimation.adjustment_solver import (
        LlrAdjustmentOptions,
        LlrAdjustmentSolver,
    )
    from llrops.estimation.variance_components import VarianceComponentDefinition

    datasets = load_datasets(config, context)
    processor = build_processor(config, context)
    parametrization = _build_parametrization(config, context)
    adjustment = config.get("adjustment") or {}
    initialization = config.get("initialization") or {}
    robust = config.get("robust_estimation") or config.get("robustEstimation") or {}
    vce = config.get("vce") or {}
    components_config = vce.get("components") or []
    components = tuple(
        VarianceComponentDefinition.from_config(item)
        for item in components_config
    )
    options = LlrAdjustmentOptions(
        components=components,
        prefit_gross_threshold_m=adjustment.get(
            "prefitGrossThresholdM",
            adjustment.get("prefit_gross_threshold_m", 20.0),
        ),
        prefit_gross_threshold_by_station_m=adjustment.get(
            "prefitGrossThresholdByStationM",
            adjustment.get("prefit_gross_threshold_by_station_m"),
        ),
        function_max_iterations=int(adjustment.get("maxIterations", 20)),
        update_tolerance_m=float(
            adjustment.get("updateToleranceM", 1.0e-3)
        ),
        update_tolerance_by_block_m={
            str(key): float(value)
            for key, value in (adjustment.get("updateToleranceByBlockM") or {}).items()
        },
        required_consecutive_converged_linearizations=int(
            adjustment.get(
                "requiredConsecutiveConvergedLinearizations",
                adjustment.get(
                    "required_consecutive_converged_linearizations",
                    2,
                ),
            )
        ),
        wrms_tolerance_m=float(
            adjustment.get("wrmsToleranceM", 1.0e-4)
        ),
        maximum_stochastic_iterations=int(
            vce.get("maximum_iterations", vce.get("maximumIterations", 8))
        ),
        k0=float(robust.get("k0", 1.5)),
        k1=float(robust.get("k1", 6.0)),
        minimum_one_minus_leverage=float(
            robust.get(
                "minimum_one_minus_leverage",
                robust.get("minimumOneMinusLeverage", 1.0e-8),
            )
        ),
        minimum_nonzero_robust_factor=float(
            robust.get(
                "minimum_nonzero_robust_factor",
                robust.get("minimumNonzeroRobustFactor", 1.0e-12),
            )
        ),
        minimum_robust_factor_for_convergence=float(
            robust.get(
                "minimum_robust_factor_for_convergence",
                robust.get("minimumRobustFactorForConvergence", 1.0e-3),
            )
        ),
        minimum_mad_count=int(
            initialization.get(
                "minimum_mad_count",
                initialization.get("minimumMadCount", 10),
            )
        ),
        minimum_initial_scale=float(
            initialization.get(
                "minimum_initial_scale",
                initialization.get("minimumInitialScale", 1.0),
            )
        ),
        bias_weight_cap=float(
            initialization.get(
                "bias_weight_cap",
                initialization.get("biasWeightCap", 1.0e12),
            )
        ),
        bias_maximum_iterations=int(
            initialization.get(
                "bias_maximum_iterations",
                initialization.get("biasMaximumIterations", 30),
            )
        ),
        minimum_effective_redundancy=float(
            vce.get(
                "minimum_effective_redundancy",
                vce.get("minimumEffectiveRedundancy", 20.0),
            )
        ),
        scale_log_tolerance=float(
            vce.get(
                "scale_log_tolerance",
                vce.get(
                    "scaleLogTolerance",
                    vce.get(
                        "variance_ratio_tolerance",
                        vce.get("varianceRatioTolerance", 2.5e-2),
                    ),
                ),
            )
        ),
        robust_factor_change_tolerance=float(
            vce.get(
                "robust_factor_change_tolerance",
                vce.get(
                    "robustFactorChangeTolerance",
                    vce.get(
                        "robust_factor_ratio_tolerance",
                        vce.get("robustFactorRatioTolerance", 2.0e-2),
                    ),
                ),
            )
        ),
        robust_factor_change_quantile=float(
            vce.get(
                "robust_factor_change_quantile",
                vce.get("robustFactorChangeQuantile", 0.999),
            )
        ),
        active_set_change_tolerance=float(
            vce.get(
                "active_set_change_tolerance",
                vce.get("activeSetChangeTolerance", 1.0e-3),
            )
        ),
        minimum_variance_ratio_per_iteration=float(
            vce.get(
                "minimum_variance_ratio_per_iteration",
                vce.get("minimumVarianceRatioPerIteration", 0.25),
            )
        ),
        variance_component_method=str(vce.get("method", "helmert")),
        maximum_variance_ratio_per_iteration=float(
            vce.get(
                "maximum_variance_ratio_per_iteration",
                vce.get("maximumVarianceRatioPerIteration", 4.0),
            )
        ),
    )

    active_stage = {"name": "joint"}

    def report_iteration(item):
        print(
            "[LlrAdjustment:HelmertVCE] "
            f"stage={active_stage['name']} "
            f"linearization={item.linearization_iteration} "
            f"stochastic={item.stochastic_iteration} "
            f"scaleLogTarget={item.maximum_scale_log_target_change:.3e} "
            f"factorTargetQ={item.robust_factor_target_change_quantile:.3e} "
            f"activeSetChange={item.active_set_change_fraction:.3e} "
            f"targetRejected={item.target_rejected_observation_count} "
            f"active={item.active_observation_count} "
            f"rejected={item.rejected_observation_count} "
            f"converged={item.stochastic_converged}",
            flush=True,
        )

    equation_source = _build_equation_source(config, context, datasets, processor)
    stage_configs = adjustment.get("stages") or [{"name": "joint"}]
    stage_results = []
    try:
        for index, stage in enumerate(stage_configs, start=1):
            stage_name = str(stage.get("name") or f"stage-{index}")
            active_stage["name"] = stage_name
            selectors = stage.get("parametrizations")
            stage_parametrization = (
                parametrization
                if not selectors
                else parametrization.select_blocks(selectors)
            )
            stage_options = replace(
                options,
                function_max_iterations=int(
                    stage.get("maxIterations", options.function_max_iterations)
                ),
                update_tolerance_m=float(
                    stage.get("updateToleranceM", options.update_tolerance_m)
                ),
                required_consecutive_converged_linearizations=int(
                    stage.get(
                        "requiredConsecutiveConvergedLinearizations",
                        options.required_consecutive_converged_linearizations,
                    )
                ),
            )
            result = LlrAdjustmentSolver(
                equation_source=equation_source,
                parametrization=stage_parametrization,
                options=stage_options,
                context=context,
                iteration_callback=(
                    report_iteration if bool(config.get("showProgress", True)) else None
                ),
            ).run()
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
    processor = build_processor(config, context)
    parametrization = _build_parametrization(config, context)
    equation_source = _build_equation_source(config, context, datasets, processor)

    try:
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
                "sigma": float(sigma0 * np.sqrt(Qxx[i, i])),
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
    print(f"[NormalsCombineSolve] solved {len(total.parameter_names)} parameters, sigma0={sigma0:.4f}")
    return solution
