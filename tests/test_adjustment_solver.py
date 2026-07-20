import json
from dataclasses import replace

import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.base.parameter_name import ParameterName
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import Parametrization, ParametrizationList
from llrops.estimation.convergence import ParameterConvergencePolicy
from llrops.estimation.normal_equation_engine import (
    DenseLinearization,
    build_normal_equations_streaming,
    solve_normal_equations,
)
from llrops.estimation.adjustment_solver import (
    LlrAdjustmentOptions,
    LlrAdjustmentSolver,
    floor_prefit_uncertainties,
)
from llrops.estimation.robust_weights import (
    Igg3WeightModel,
    active_set_change_fraction,
    igg3_factors,
    maximum_robust_factor_change,
    robust_factor_change_quantile,
)
from llrops.estimation.vce import HelmertVceEstimator
from llrops.estimation.variance_components import (
    VarianceComponentDefinition,
    assign_variance_components,
)


def _equation(identity, value, station, wavelength=532.0):
    return ObservationEquation(
        observed_minus_computed_m=float(value),
        sigma_m=1.0,
        partials={"test": np.array([1.0])},
        identity=identity,
        station_key=station,
        reflector_key="REF",
        epoch=Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC),
        metadata={"station_name": station, "wavelength_nm": wavelength},
    )


class OffsetParametrization(Parametrization):
    def __init__(self):
        self.value = 0.0

    def parameter_names(self):
        return [ParameterName("test", "position.x")]

    def design_columns(self, equation):
        return np.array([1.0])

    def reduce_observation(self, equation):
        return self.value

    def apply_update(self, delta):
        self.value += float(delta[0])


class AffineParametrization(Parametrization):
    def __init__(self):
        self.value = np.zeros(2)

    def parameter_names(self):
        return [
            ParameterName("test", "position.x"),
            ParameterName("test", "position.y"),
        ]

    def design_columns(self, equation):
        station_offset = 10.0 if equation.station_key == "STA_B" else 0.0
        return np.array([1.0, 1.0e4 * (equation.identity[1] + station_offset)])

    def reduce_observation(self, equation):
        return float(self.design_columns(equation) @ self.value)

    def apply_update(self, delta):
        self.value += np.asarray(delta, dtype=float)


def test_parametrization_selection_reuses_block_state():
    offset = OffsetParametrization()
    parametrization = ParametrizationList([offset])

    selected = parametrization.select_blocks(["OffsetParametrization"])
    selected.blocks[0].apply_update(np.array([2.0]))

    assert offset.value == pytest.approx(2.0)
    with pytest.raises(KeyError, match="Unknown parametrization"):
        parametrization.select_blocks(["MissingParametrization"])


def test_parameter_convergence_policy_supports_block_tolerances():
    policy = ParameterConvergencePolicy(
        default_tolerance_m=1.0e-3,
        tolerance_by_block_m={"StationRangeBiasParametrization": 2.0e-3},
    )
    evaluation = policy.evaluate(
        {
            "0:ReflectorPositionParametrization": 0.9e-3,
            "1:StationRangeBiasParametrization": 1.5e-3,
        }
    )

    assert evaluation.converged
    assert evaluation.tolerances_m["0:ReflectorPositionParametrization"] == pytest.approx(1.0e-3)
    assert evaluation.tolerances_m["1:StationRangeBiasParametrization"] == pytest.approx(2.0e-3)


def test_vce_assignment_distinguishes_overlapping_cerga_systems_by_wavelength():
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "CERGA_MEO",
                "station_system": "CERGA_MEO",
                "station_aliases": ["CERGA"],
                "start": "2015-01-01",
                "end_exclusive": "2023-01-01",
                "wavelength_max_nm": 700.0,
            }
        ),
        VarianceComponentDefinition.from_config(
            {
                "id": "CERGA_IR",
                "station_system": "CERGA_IR",
                "station_aliases": ["CERGA"],
                "start": "2015-01-01",
                "end_exclusive": None,
                "wavelength_min_nm": 700.0,
            }
        ),
    )
    equations = [_equation("green", 0.0, "CERGA", 532.0), _equation("infrared", 0.0, "CERGA", 1064.0)]

    assert assign_variance_components(equations, components) == {"green": "CERGA_MEO", "infrared": "CERGA_IR"}


def test_prefit_uncertainty_qc_floors_only_abnormally_small_sigmas():
    equations = [
        replace(_equation("tiny", 0.0, "STA_A"), sigma_m=1.0e-5),
        replace(_equation("normal-1", 0.0, "STA_A"), sigma_m=0.02),
        replace(_equation("normal-2", 0.0, "STA_A"), sigma_m=0.03),
    ]

    adjusted, records, groups = floor_prefit_uncertainties(
        equations,
        {equation.identity: "A" for equation in equations},
        minimum_sigma_m=1.0e-3,
        minimum_group_median_fraction=0.1,
    )

    assert groups["A"] == {
        "median_reported_sigma_m": pytest.approx(0.02),
        "sigma_floor_m": pytest.approx(0.002),
        "observation_count": 3,
        "floored_count": 1,
    }
    assert [equation.sigma_m for equation in adjusted] == pytest.approx(
        [0.002, 0.02, 0.03]
    )
    assert records["tiny"]["status"] == "FLOORED"
    assert records["tiny"]["reported_sigma_m"] == pytest.approx(1.0e-5)
    assert records["normal-1"]["status"] == "UNCHANGED"
    assert adjusted[0].metadata["uncertainty_quality_control"]["reason"] == (
        "BELOW_PREFIT_UNCERTAINTY_FLOOR"
    )


def test_vce_assignment_rejects_unassigned_observation():
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    with pytest.raises(ValueError, match="no matching component"):
        assign_variance_components([_equation(1, 0.0, "STA_B")], components)


def test_igg3_boundaries():
    factors = igg3_factors(np.array([0.0, 1.5, 1.5001, 6.0, 6.1]), k0=1.5, k1=6.0)

    assert factors[0] == 1.0
    assert factors[1] == 1.0
    assert 0.0 < factors[2] < 1.0
    assert factors[3] == 0.0
    assert factors[4] == 0.0


@pytest.mark.parametrize(
    ("old_factor", "new_factor", "expected"),
    [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.55, 0.05),
        (0.0, 0.2, 0.2),
        (0.0, 1.0, 1.0),
    ],
)
def test_robust_factor_change_is_absolute(
    old_factor, new_factor, expected
):
    assert maximum_robust_factor_change(
        {"observation": old_factor},
        {"observation": new_factor},
        ["observation"],
    ) == pytest.approx(expected)


def test_factor_change_quantile_ignores_one_chattering_observation():
    keys = list(range(1001))
    old = {key: 0.5 for key in keys}
    target = {key: 0.501 for key in keys}
    target[keys[-1]] = 1.0

    assert robust_factor_change_quantile(
        old,
        target,
        keys,
        quantile=0.999,
    ) == pytest.approx(0.001)


def test_active_set_change_fraction_counts_membership_only():
    old = {"stable": 1.0, "removed": 0.5, "added": 0.0}
    new = {"stable": 0.2, "removed": 0.0, "added": 0.3}

    assert active_set_change_fraction(
        old,
        new,
        list(old),
        active_threshold=1.0e-12,
    ) == pytest.approx(2.0 / 3.0)


def test_igg3_update_accepts_observation_missing_from_previous_targets():
    model = Igg3WeightModel()
    update = model.update(
        {"stable": 0.0, "reentered": 0.0},
        {"stable": 1.0, "reentered": 1.0},
        {"stable": 1.0},
        ["stable", "reentered"],
    )

    assert update.target_factors == {"stable": 1.0, "reentered": 1.0}
    assert update.active_set_change_fraction == 0.0


def test_igg3_update_preserves_factor_for_temporarily_missing_observation():
    model = Igg3WeightModel()
    update = model.update(
        {"present": 0.0},
        {"present": 0.5, "temporarily_missing": 0.25},
        {"present": 0.5, "temporarily_missing": 0.25},
        ["present"],
    )

    assert update.applied_factors == {
        "present": 1.0,
        "temporarily_missing": 0.25,
    }


def test_igg3_targets_are_applied_without_damping():
    model = Igg3WeightModel(k0=1.5, k1=6.0)
    update = model.update(
        {"full": 0.0, "downweighted": 3.0, "rejected": 7.0},
        {"full": 0.2, "downweighted": 0.8, "rejected": 1.0},
        {"full": 1.0, "downweighted": 1.0, "rejected": 1.0},
        ["full", "downweighted", "rejected"],
    )

    assert update.applied_factors == update.target_factors
    assert update.applied_factors["full"] == 1.0
    assert 0.0 < update.applied_factors["downweighted"] < 1.0
    assert update.applied_factors["rejected"] == 0.0


def test_factor_change_ignores_insignificant_boundary_crossings():
    old = {"weak": 0.0, "material": 0.0}
    new = {"weak": 1.0e-4, "material": 0.01}

    change = maximum_robust_factor_change(
        old,
        new,
        list(old),
        significance_floor=1.0e-3,
    )

    assert change == pytest.approx(0.01)
    assert maximum_robust_factor_change(
        {"weak": 0.0},
        {"weak": 1.0e-4},
        ["weak"],
        significance_floor=1.0e-3,
    ) == 0.0


def test_vce_direct_update_respects_variance_ratio_limit():
    equations = [
        _equation(index, value, "STA_A")
        for index, value in enumerate([0.0, 100.0, 200.0])
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=6,
            maximum_stochastic_iterations=3,
            required_consecutive_converged_linearizations=1,
            update_tolerance_m=1.0e-6,
            k0=1.0e6,
            k1=2.0e6,
            minimum_mad_count=4,
            minimum_effective_redundancy=1.0,
            scale_log_tolerance=1.0e-6,
        ),
    ).run()

    assert result.converged
    assert result.scales["A"] == pytest.approx(100.0)
    first_component = result.iterations[0].variance_components["A"]
    assert first_component["raw_variance"] == pytest.approx(10000.0)
    assert first_component["limited_variance_ratio"] == pytest.approx(4.0)
    assert first_component["variance_after"] == pytest.approx(4.0)


def _two_component_case():
    equations = [
        replace(_equation(("A", i), value, "STA_A"), sigma_m=0.5 + 0.1 * i)
        for i, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ] + [
        replace(_equation(("B", i), value, "STA_B"), sigma_m=0.8 + 0.1 * i)
        for i, value in enumerate([0.0, 2.0, 1.7, -0.2, 2.4, 0.4])
    ]
    components = tuple(
        VarianceComponentDefinition.from_config(
            {
                "id": name,
                "station_system": name,
                "station_aliases": [f"STA_{name}"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        )
        for name in ("A", "B")
    )
    return equations, components


def test_dense_linearization_matches_streaming_normals_and_vce():
    equations, components = _two_component_case()
    parametrization = ParametrizationList([AffineParametrization()])
    parametrization.setup(equations, None)
    names = parametrization.parameter_names()
    assignments = assign_variance_components(equations, components)
    scales = {"A": 1.3, "B": 0.7}
    factors = {
        equation.identity: 0.2 + 0.8 * (index + 1) / len(equations)
        for index, equation in enumerate(equations)
    }
    weights = np.asarray(
        [
            factors[equation.identity]
            / (scales[assignments[equation.identity]] ** 2 * equation.sigma_m**2)
            for equation in equations
        ]
    )

    dense = DenseLinearization.build(equations, parametrization, names)
    dense_normals = dense.normal_equations(weights)
    streaming_normals = build_normal_equations_streaming(
        equations,
        parametrization,
        parameter_names=names,
        weight_for=lambda equation: weights[equations.index(equation)],
    )
    assert np.array_equal(dense_normals.N, dense_normals.N.T)
    assert dense_normals.N == pytest.approx(streaming_normals.N, rel=1.0e-13)
    assert dense_normals.W == pytest.approx(streaming_normals.W, rel=1.0e-13)
    assert dense_normals.lPl == pytest.approx(streaming_normals.lPl, rel=1.0e-13)

    dense_solved = solve_normal_equations(dense_normals)
    streaming_solved = solve_normal_equations(streaming_normals)
    assert dense_solved.delta == pytest.approx(streaming_solved.delta, rel=1.0e-13)
    assert dense_solved.covariance == pytest.approx(
        streaming_solved.covariance, rel=1.0e-13
    )
    residual_vector = dense.reduced_observations - dense.design @ dense_solved.delta
    residuals = {
        key: float(value) for key, value in zip(dense.identities, residual_vector)
    }
    estimator = HelmertVceEstimator(
        components, minimum_effective_redundancy=1.0
    )
    streaming_estimate = estimator.estimate(
        equations=equations,
        residuals=residuals,
        normals=streaming_normals,
        parametrization=parametrization,
        parameter_names=names,
        assignments=assignments,
        factors=factors,
        scales=scales,
        covariance=streaming_solved.covariance,
    )
    dense_estimate = estimator.estimate_dense(
        design=dense.design,
        sigmas=dense.sigmas,
        residuals=residual_vector,
        component_ids=np.asarray([assignments[key] for key in dense.identities]),
        factors=np.asarray([factors[key] for key in dense.identities]),
        scales=scales,
        normals=dense_normals,
        covariance=dense_solved.covariance,
    )
    assert dense_estimate.scales == pytest.approx(
        streaming_estimate.scales, rel=1.0e-12
    )
    for component in components:
        assert dense_estimate.diagnostics[component.id] == pytest.approx(
            streaming_estimate.diagnostics[component.id], rel=1.0e-12
        )


def _run_backend(backend, *, initial_scales=None, initial_factors=None):
    equations, components = _two_component_case()
    return LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=2,
            maximum_stochastic_iterations=3,
            required_consecutive_converged_linearizations=99,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            linearization_backend=backend,
        ),
        initial_scales=initial_scales,
        initial_factors=initial_factors,
    ).run()


def test_dense_and_streaming_adjustments_are_equivalent_and_warm_startable():
    dense = _run_backend("dense")
    streaming = _run_backend("streaming")

    assert dense.scales == pytest.approx(streaming.scales, rel=1.0e-10)
    assert dense.robust_factors == pytest.approx(
        streaming.robust_factors, rel=1.0e-10
    )
    assert dense.normals.N == pytest.approx(streaming.normals.N, rel=1.0e-10)
    assert dense.normals.W == pytest.approx(streaming.normals.W, rel=1.0e-10)
    assert dense.state == streaming.state

    warm = _run_backend(
        "dense", initial_scales=dense.scales, initial_factors=dense.robust_factors
    )
    assert warm.settings["warm_started_scale_count"] == 2
    assert warm.settings["warm_started_factor_count"] == 12
    assert set(warm.summary["performance_seconds"]) == {
        "cache_build",
        "normal_solve",
        "leverage",
        "vce",
    }


def test_llr_adjustment_runs_joint_helmert_vce_cycle():
    equations = [
        _equation(("A", index), value, "STA_A")
        for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ] + [
        _equation(("B", index), value, "STA_B")
        for index, value in enumerate([0.0, 2.0, 1.7, -0.2, 2.4, 0.4])
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
        VarianceComponentDefinition.from_config(
            {
                "id": "B",
                "station_system": "B",
                "station_aliases": ["STA_B"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=6,
            maximum_stochastic_iterations=4,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
        ),
    ).run()

    assert set(result.scales) == {"A", "B"}
    assert result.normals is not None
    assert len(result.observations) == len(equations)
    for item in result.observations:
        base_sigma = item["base_scale"] * item["sigma_np"]
        assert 0.0 <= item["leverage"] < 1.0
        assert item["residual_sigma"] == pytest.approx(
            base_sigma * np.sqrt(1.0 - item["leverage"])
        )
        assert item["standardized_residual"] == pytest.approx(
            item["postfit_residual"] / item["residual_sigma"]
        )
        assert item["reported_sigma_np"] == pytest.approx(item["sigma_np"])
        assert item["effective_sigma_np"] == pytest.approx(item["sigma_np"])
        assert item["uncertainty_qc_status"] == "UNCHANGED"
    assert all(0.0 <= factor <= 1.0 for factor in result.robust_factors.values())
    assert result.iterations[-1].total_effective_redundancy == pytest.approx(
        result.iterations[-1].expected_total_redundancy
    )
    for iteration in result.iterations:
        expected = max(
            abs(group["variance_after"] / group["variance_before"] - 1.0)
            for group in iteration.variance_components.values()
        )
        assert iteration.maximum_variance_ratio_change == pytest.approx(expected)
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["summary"]["source_observation_count"] == len(equations)
    assert payload["summary"]["equation_evaluation_count"] == len(payload["equation_evaluations"])
    assert payload["parameters"][0]["formal_sigma_m"] is not None
    assert payload["global_residuals"]["residual_m"]["count"] == len(equations)
    assert payload["variance_components"][0]["actual_start_epoch"] is not None
    counts = ("full_weight_count", "downweighted_count", "rejected_count")
    assert sum(payload["variance_components"][0][key] for key in counts) == payload["variance_components"][0]["observation_count"]
    assert payload["iterations"][0]["candidate_update_by_block_m"]
    assert payload["iterations"][0]["variance_components"]
    assert "maximum_scale_log_target_change" in payload["iterations"][0]
    assert "robust_factor_target_change_quantile" in payload["iterations"][0]
    assert "active_set_change_fraction" in payload["iterations"][0]
    assert "target_rejected_observation_count" in payload["iterations"][0]
    assert not any("damping" in key for key in payload["settings"])


def test_adjustment_reports_prefit_uncertainty_floor():
    equations = [
        replace(_equation("tiny", 0.0, "STA_A"), sigma_m=1.0e-5),
        replace(_equation("normal-1", 1.0, "STA_A"), sigma_m=0.02),
        replace(_equation("normal-2", 2.0, "STA_A"), sigma_m=0.03),
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )

    result = LlrAdjustmentSolver(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=1,
            maximum_stochastic_iterations=1,
            required_consecutive_converged_linearizations=1,
            update_tolerance_m=10.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
            uncertainty_floor_minimum_m=1.0e-3,
            uncertainty_floor_group_median_fraction=0.1,
        ),
    ).run()

    records = {item["observation_id"]: item for item in result.observations}
    assert result.summary["uncertainty_sigma_floored_count"] == 1
    assert result.summary["retained_uncertainty_sigma_floored_count"] == 1
    assert records["tiny"]["reported_sigma_np"] == pytest.approx(1.0e-5)
    assert records["tiny"]["effective_sigma_np"] == pytest.approx(0.002)
    assert records["tiny"]["uncertainty_qc_status"] == "FLOORED"
    assert records["normal-1"]["effective_sigma_np"] == pytest.approx(0.02)
    assert result.uncertainty_quality_control["groups"]["A"][
        "floored_count"
    ] == 1


from llrops.classes.parametrization.station_range_bias import StationRangeBiasParametrization
from llrops.estimation.adjustment import AdjustmentOptions, LeastSquaresAdjustment


def test_open_bias_interval_remains_active():
    equation = _equation(1, 0.0, "WETTZELL")
    block = StationRangeBiasParametrization(
        per="station+interval",
        intervals=[
            {
                "station": "WETTZELL",
                "start": "2018-01-01",
                "end_exclusive": None,
                "name": "WETTZELL_PRESENT",
            }
        ],
    )
    block.setup([equation], None)

    assert block.keys == ["WETTZELL_PRESENT"]
    assert np.allclose(block.design_columns(equation), [1.0])


def test_prefit_gross_rejection_never_reenters():
    equations = [_equation("gross", 100.0, "STA"), _equation("good", 1.0, "STA")]
    result = LeastSquaresAdjustment(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=AdjustmentOptions(
            max_iterations=2,
            prefit_gross_threshold_m=20.0,
            enable_outlier_rejection=True,
            outlier_sigma_factor=1000.0,
            allow_outlier_reentry=True,
        ),
    ).run()

    assert result.gross_rejected_keys == {"gross": 1}
    assert result.rejected_keys == {"gross": 1}


def test_extended_variance_components_cover_mcdonald_1969_and_current_meo():
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "MCDONALD_1969_1985",
                "station_system": "MCDONALD",
                "station_aliases": ["MCDONALD"],
                "start": "1969-01-01",
                "end_exclusive": "1986-01-01",
            }
        ),
        VarianceComponentDefinition.from_config(
            {
                "id": "CERGA_MEO_2009_PRESENT",
                "station_system": "CERGA_MEO",
                "station_aliases": ["GRASSE"],
                "start": "2009-01-01",
                "end_exclusive": None,
                "wavelength_max_nm": 700.0,
            }
        ),
    )
    mcdonald = replace(
        _equation("mcdonald-1969", 0.0, "MCDONALD", 694.3),
        epoch=Epoch.from_isot("1969-08-20T00:00:00", scale=TimeScale.UTC),
    )
    meo = replace(
        _equation("meo-2024", 0.0, "GRASSE", 532.1),
        epoch=Epoch.from_isot("2024-12-21T00:00:00", scale=TimeScale.UTC),
    )

    assert assign_variance_components([mcdonald, meo], components) == {
        "mcdonald-1969": "MCDONALD_1969_1985",
        "meo-2024": "CERGA_MEO_2009_PRESENT",
    }



def test_stochastic_iterations_do_not_recompute_observation_equations():
    equations = [
        _equation(index, value, "STA_A")
        for index, value in enumerate([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    source_calls = []

    def source(iteration):
        source_calls.append(iteration)
        return equations

    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=2,
            maximum_stochastic_iterations=5,
            required_consecutive_converged_linearizations=1,
            update_tolerance_m=10.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
            scale_log_tolerance=1.0e-12,
            robust_factor_change_tolerance=1.0e-12,
        ),
    ).run()

    assert len(result.iterations) > 1
    assert source_calls == [1]
    assert result.converged
    assert result.termination_reason == "CONVERGED"
    assert result.summary["equation_evaluation_count"] == 1
    assert result.equation_evaluations[0]["fixed_domain_returned_count"] == 6


def test_stochastic_iteration_limit_still_applies_parameter_update():
    equations = [
        _equation(index, value, "STA_A")
        for index, value in enumerate([0.0, 0.0, 0.0, 0.0, 0.0, 20.0])
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    source_calls = []

    def source(iteration):
        source_calls.append(iteration)
        return equations

    block = OffsetParametrization()
    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([block]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=2,
            geometry_update_factor=0.5,
            maximum_stochastic_iterations=1,
            update_tolerance_m=0.0,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
            scale_log_tolerance=0.0,
            robust_factor_change_tolerance=0.0,
            active_set_change_tolerance=0.0,
        ),
    ).run()

    assert source_calls == [1, 2]
    first = result.linearizations[0]
    assert not first["stochastic_converged"]
    assert first["stochastic_iteration_limit_reached"]
    assert first["geometry_update_factor"] == 0.5
    candidate = first["candidate_update_by_block_m"]["0:OffsetParametrization"]
    assert first["maximum_parameter_update_m"] == pytest.approx(candidate)
    applied = first["applied_update_by_block_m"]["OffsetParametrization"]
    assert applied == pytest.approx(0.5 * candidate)
    assert applied > 0.0
    assert result.settings["geometry_update_factor"] == 0.5
    assert result.termination_reason == "MAXIMUM_GEOMETRY_ITERATIONS_REACHED"


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.01])
def test_geometry_update_factor_must_be_in_unit_interval(factor):
    with pytest.raises(ValueError, match="Geometry update factor"):
        LlrAdjustmentOptions(components=(), geometry_update_factor=factor)


def test_fixed_domain_observation_can_reenter_after_one_failed_linearization():
    equations = [
        _equation(index, value, "STA_A")
        for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ]
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )

    def source(iteration):
        return [
            replace(equation, converged=not (iteration == 2 and index == 0))
            for index, equation in enumerate(equations)
        ]

    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=3,
            maximum_stochastic_iterations=1,
            required_consecutive_converged_linearizations=99,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            k0=1.0e6,
            k1=2.0e6,
        ),
    ).run()

    assert [
        item["fixed_domain_returned_count"]
        for item in result.equation_evaluations
    ] == [6, 5, 6]
    assert set(result.robust_factors) == set(range(6))


def test_parameter_convergence_requires_two_confirmation_linearizations():
    equations = [
        _equation(index, value, "STA_A")
        for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ]
    late = _equation("late", 100.0, "STA_A")
    components = (
        VarianceComponentDefinition.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    source_calls = []

    def source(iteration):
        source_calls.append(iteration)
        return equations + [replace(late, converged=iteration > 1)]

    block = OffsetParametrization()
    result = LlrAdjustmentSolver(
        equation_source=source,
        parametrization=ParametrizationList([block]),
        options=LlrAdjustmentOptions(
            components=components,
            prefit_gross_threshold_m=None,
            function_max_iterations=4,
            maximum_stochastic_iterations=2,
            update_tolerance_m=1.0e-6,
            minimum_mad_count=2,
            minimum_effective_redundancy=1.0,
            scale_log_tolerance=10.0,
            robust_factor_change_tolerance=10.0,
        ),
    ).run()

    assert source_calls == [1, 2, 3]
    assert len(result.linearizations) == 3
    first_linearization = result.linearizations[0]
    assert not first_linearization["parameter_converged"]
    assert first_linearization["applied_update_by_block_m"]["OffsetParametrization"] == pytest.approx(
        first_linearization["candidate_update_by_block_m"]["0:OffsetParametrization"]
    )
    assert "geometry_damping" not in result.settings
    assert result.summary["equation_evaluation_count"] == 3
    assert result.equation_evaluations[1]["fixed_domain_returned_count"] == 6
    assert result.linearizations[-2]["consecutive_converged_linearizations"] == 1
    assert result.linearizations[-1]["consecutive_converged_linearizations"] == 2
    assert result.converged
    assert block.value == pytest.approx(np.mean([0.7, 1.0, 1.2, 0.8, 1.1, 0.9]))
    assert "late" not in result.robust_factors
    assert all(item["observation_id"] != "late" for item in result.observations)
