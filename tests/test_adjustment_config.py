from copy import deepcopy
from pathlib import Path

import pytest

from llrops.config.loader import load_config_file
from llrops.estimation.adjustment_config import parse_adjustment_plan


def _component(component_id="A"):
    return {
        "id": component_id,
        "stationSystem": "A",
        "stationAliases": ["STA_A"],
        "start": "2020-01-01",
        "endExclusive": None,
    }


def _config():
    return {"vce": {"components": [_component()]}}


def test_canonical_adjustment_schema_maps_to_typed_plan():
    config = _config()
    config.update(
        {
            "adjustment": {
                "maximumLinearizations": 9,
                "parameterUpdateFactor": 0.5,
                "linearizationBackend": "streaming",
                "uncertaintyFloor": {
                    "minimumOneWaySigmaM": 0.002,
                    "minimumFractionOfGroupMedian": 0.1,
                },
                "updateToleranceM": 0.003,
                "updateToleranceByBlockM": {"OffsetParametrization": 0.004},
                "requiredConsecutiveConvergedLinearizations": 3,
                "prefitGrossThresholdM": None,
                "prefitGrossThresholdByStationM": {"STA_A": 2.0, "STA_B": None},
                "warmStartStochasticModelAcrossStages": False,
                "stages": [
                    {
                        "name": "offset",
                        "parametrizations": ["OffsetParametrization"],
                        "maximumLinearizations": 4,
                        "parameterUpdateFactor": 0.25,
                        "updateToleranceM": 0.001,
                        "requiredConsecutiveConvergedLinearizations": 1,
                    }
                ],
            },
            "initialization": {
                "biasWeightCap": 1.0e10,
                "biasMaximumIterations": 12,
                "minimumMadCount": 5,
                "minimumInitialScale": 0.5,
            },
            "robustEstimation": {
                "k0": 1.2,
                "k1": 5.0,
                "minimumOneMinusLeverage": 1.0e-7,
                "activeFactorThreshold": 1.0e-10,
                "convergenceFactorFloor": 1.0e-4,
                "changeQuantile": 0.99,
            },
            "vce": {
                "components": [_component()],
                "maximumIterations": 6,
                "minimumEffectiveRedundancy": 5.0,
                "scaleLogTolerance": 0.02,
                "targetFactorChangeTolerance": 0.01,
                "targetActiveSetChangeTolerance": 0.005,
                "minimumVarianceRatioPerIteration": 0.5,
                "maximumVarianceRatioPerIteration": 2.0,
            },
        }
    )

    plan = parse_adjustment_plan(config)
    options = plan.options
    stage_options = plan.stages[0].apply(options)

    assert options.maximum_linearizations == 9
    assert options.parameter_update_factor == pytest.approx(0.5)
    assert options.linearization_backend == "streaming"
    assert options.prefit_gross_threshold_m is None
    assert options.maximum_stochastic_iterations == 6
    assert options.robust_factor_change_quantile == pytest.approx(0.99)
    assert not plan.warm_start_stochastic_model_across_stages
    assert plan.stages[0].parametrizations == ("OffsetParametrization",)
    assert stage_options.maximum_linearizations == 4
    assert stage_options.parameter_update_factor == pytest.approx(0.25)
    assert stage_options.update_tolerance_m == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("section", "obsolete_key"),
    [
        ("adjustment", "maxIterations"),
        ("adjustment", "geometryUpdateFactor"),
        ("adjustment", "wrmsToleranceM"),
        ("initialization", "minimum_mad_count"),
        ("robustEstimation", "minimum_one_minus_leverage"),
        ("vce", "method"),
        ("vce", "maximum_iterations"),
    ],
)
def test_obsolete_or_unknown_keys_are_rejected(section, obsolete_key):
    config = _config()
    config.setdefault(section, {})[obsolete_key] = 1

    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)


def test_obsolete_robust_section_name_is_rejected_explicitly():
    config = _config()
    config["robust_estimation"] = {"k0": 1.5}

    with pytest.raises(ValueError, match="use robustEstimation"):
        parse_adjustment_plan(config)


def test_uncertainty_floor_has_no_fake_strategy_selector():
    config = _config()
    config["adjustment"] = {"uncertaintyFloor": {"action": "floor"}}

    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)


def test_variance_component_schema_is_strict():
    config = _config()
    config["vce"]["components"][0]["station_system"] = "A"

    with pytest.raises(ValueError, match="unknown key"):
        parse_adjustment_plan(config)


def test_duplicate_stage_names_are_rejected():
    config = _config()
    config["adjustment"] = {
        "stages": [{"name": "joint"}, {"name": "joint"}]
    }

    with pytest.raises(ValueError, match="names must be unique"):
        parse_adjustment_plan(config)


def test_stage_override_is_validated_eagerly():
    config = _config()
    config["adjustment"] = {
        "stages": [{"name": "joint", "parameterUpdateFactor": 1.5}]
    }

    with pytest.raises(ValueError, match="Parameter update factor"):
        parse_adjustment_plan(config)


def test_duplicate_variance_component_ids_are_rejected():
    config = _config()
    config["vce"]["components"].append(_component())

    with pytest.raises(ValueError, match="IDs must be unique"):
        parse_adjustment_plan(config)


def test_variance_component_dates_are_validated():
    config = _config()
    config["vce"]["components"][0]["endExclusive"] = "2019-01-01"

    with pytest.raises(ValueError, match="must be after start"):
        parse_adjustment_plan(config)


def test_detailed_adjustment_config_uses_the_canonical_schema():
    root = Path(__file__).resolve().parents[1]
    config = load_config_file(
        root / "configs" / "llrops_reflector_bias_adjustment_detailed.yml"
    )
    adjustment_program = next(
        item for item in config["programs"] if item.get("program") == "LlrAdjustment"
    )

    plan = parse_adjustment_plan(deepcopy(adjustment_program))

    assert [stage.name for stage in plan.stages] == ["reflector", "bias", "joint"]
    assert len(plan.options.components) == 11
    assert plan.stages[-1].parameter_update_factor == pytest.approx(0.5)
