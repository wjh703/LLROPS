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
        if config.get("program") == "LlrGroupedVceAdjustment"
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


@program("LlrAdjustment")
def llr_adjustment(config: dict, context: RunContext):
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
        f"[LlrAdjustment] converged={result.converged} "
        f"iterations={len(result.iterations)} parameters={len(result.parameter_names)}"
    )
    return result


@program("LlrGroupedVceAdjustment")
def llr_grouped_vce_adjustment(config: dict, context: RunContext):
    """Run the grouped VCE, interval-bias, and IGGIII joint adjustment."""
    from llrops.estimation.grouped_vce import (
        GroupedVceAdjustment,
        GroupedVceOptions,
        VceGroup,
    )

    datasets = load_datasets(config, context)
    processor = build_processor(config, context)
    parametrization = _build_parametrization(config, context)
    adjustment = config.get("adjustment") or {}
    initialization = config.get("initialization") or {}
    robust = config.get("robust_estimation") or config.get("robustEstimation") or {}
    vce = config.get("vce") or {}
    groups_config = config.get("vce_groups") or config.get("vceGroups") or []
    groups = tuple(VceGroup.from_config(item) for item in groups_config)
    options = GroupedVceOptions(
        groups=groups,
        prefit_gross_threshold_m=adjustment.get("prefitGrossThresholdM", 20.0),
        prefit_gross_threshold_by_station_m=adjustment.get("prefitGrossThresholdByStationM"),
        function_max_iterations=int(adjustment.get("maxIterations", 20)),
        function_damping=float(adjustment.get("damping", 1.0)),
        update_tolerance_m=float(adjustment.get("updateToleranceM", 1.0e-3)),
        wrms_tolerance_m=float(adjustment.get("wrmsToleranceM", 1.0e-4)),
        maximum_stochastic_iterations=int(vce.get("maximum_iterations", vce.get("maximumIterations", 20))),
        k0=float(robust.get("k0", 1.5)),
        k1=float(robust.get("k1", 6.0)),
        minimum_one_minus_leverage=float(
            robust.get("minimum_one_minus_leverage", robust.get("minimumOneMinusLeverage", 1.0e-8))
        ),
        minimum_nonzero_robust_factor=float(
            robust.get("minimum_nonzero_robust_factor", robust.get("minimumNonzeroRobustFactor", 1.0e-12))
        ),
        minimum_mad_count=int(initialization.get("minimum_mad_count", initialization.get("minimumMadCount", 10))),
        minimum_initial_scale=float(
            initialization.get("minimum_initial_scale", initialization.get("minimumInitialScale", 1.0))
        ),
        bias_weight_cap=float(initialization.get("bias_weight_cap", initialization.get("biasWeightCap", 1.0e12))),
        bias_maximum_iterations=int(
            initialization.get("bias_maximum_iterations", initialization.get("biasMaximumIterations", 30))
        ),
        vce_damping=float(vce.get("damping", 0.5)),
        minimum_effective_redundancy=float(
            vce.get("minimum_effective_redundancy", vce.get("minimumEffectiveRedundancy", 20.0))
        ),
        scale_log_tolerance=float(vce.get("scale_log_tolerance", vce.get("scaleLogTolerance", 1.0e-3))),
        robust_weight_tolerance=float(
            vce.get("robust_weight_tolerance", vce.get("robustWeightTolerance", 1.0e-3))
        ),
        minimum_variance_ratio_per_iteration=float(
            vce.get("minimum_variance_ratio_per_iteration", vce.get("minimumVarianceRatioPerIteration", 0.25))
        ),
        maximum_variance_ratio_per_iteration=float(
            vce.get("maximum_variance_ratio_per_iteration", vce.get("maximumVarianceRatioPerIteration", 4.0))
        ),
    )

    def report_iteration(item):
        print(
            "[GroupedVCE] "
            f"linearization={item.linearization_iteration} "
            f"stochastic={item.stochastic_iteration} "
            f"scaleLogChange={item.maximum_scale_log_change:.3e} "
            f"factorChange={item.maximum_robust_factor_change:.3e} "
            f"active={item.active_observation_count} "
            f"rejected={item.rejected_observation_count}",
            flush=True,
        )

    try:
        result = GroupedVceAdjustment(
            equation_source=_build_equation_source(config, context, datasets, processor),
            parametrization=parametrization,
            options=options,
            context=context,
            iteration_callback=(
                report_iteration
                if bool(config.get("showProgress", True))
                else None
            ),
        ).run()
    finally:
        processor.close()

    if config.get("outputJson"):
        path = context.resolve_path(config["outputJson"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    if config.get("outputNormals") and result.normals is not None:
        result.normals.save(context.resolve_path(config["outputNormals"]))
    print(
        f"[LlrGroupedVceAdjustment] converged={result.converged} "
        f"linearizations={len(result.linearizations)} "
        f"stochasticIterations={len(result.iterations)} groups={len(result.scales)}"
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
