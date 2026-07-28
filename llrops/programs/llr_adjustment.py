"""Generalized, parametrization-driven LLR adjustment program.

Example configuration::

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
      outputNormals: normals/llr
"""
from __future__ import annotations

import json

from llrops.config.context import RunContext
from llrops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    build_processor,
    load_datasets,
)
from llrops.programs.registry import program



@program("LlrAdjustment")
def llr_adjustment(config: dict, context: RunContext):
    """Run nonlinear LLR adjustment with robust weights and VCE."""
    from llrops.estimation.adjustment_config import parse_adjustment_plan
    from llrops.estimation.adjustment_solver import LlrAdjustmentSolver

    plan = parse_adjustment_plan(config)
    options = plan.options
    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
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
    equation_source = build_equation_source(config, context, datasets, processor)
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
            model_state=processor.model_state,
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
            f"stage={stage_name} "
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
                    block.block_id for block in stage_parametrization.blocks
                ],
                "summary": result.summary,
                "state": result.state,
            }
        )

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


__all__ = ["llr_adjustment"]
