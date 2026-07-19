import numpy as np

from llrops.base.parameter_name import ParameterName
from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import Parametrization, ParametrizationList
from llrops.estimation.adjustment import AdjustmentOptions, LeastSquaresAdjustment


class GeometryLikeParametrization(Parametrization):
    """A minimal geometry-like block.

    It has design partials, but its update is not absorbed into the already
    constructed ObservationEquation objects.  This mimics reflector coordinates:
    the update affects residuals only after the forward model is rerun at the
    next linearization point.
    """

    def __init__(self):
        self.value = 0.0

    def parameter_names(self):
        return [ParameterName("test", "position.x")]

    def design_columns(self, eq):
        return np.array([float(eq.partials["geometry"][0])])

    def apply_update(self, delta):
        self.value += float(delta[0])

    def state(self):
        return {"value": self.value}


def _eq(i, l_m, sigma_m=1.0):
    return ObservationEquation(
        observed_minus_computed_m=float(l_m),
        sigma_m=float(sigma_m),
        partials={"geometry": np.array([1.0])},
        identity=("synthetic", i),
        station_key="STA",
        reflector_key="REF",
        epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
    )


def test_postfit_outlier_test_uses_linearized_residual_before_apply_update():
    # Linearized system: l=[10, 10], A=[1, 1], delta=10, so postfit residuals
    # are exactly zero.  The old generic-adjustment logic applied the update
    # first and then tested reduced_observation(eq), which for geometry-like
    # parameters still equals the original l and would falsely reject both rows.
    equations = [_eq(1, 10.0), _eq(2, 10.0)]
    block = GeometryLikeParametrization()
    result = LeastSquaresAdjustment(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([block]),
        options=AdjustmentOptions(
            max_iterations=1,
            prefit_gross_threshold_m=None,
            enable_outlier_rejection=True,
            outlier_sigma_factor=3.0,
            allow_outlier_reentry=True,
        ),
    ).run()

    assert np.allclose(result.solution, [10.0])
    assert result.rejected_keys == {}
    assert np.isclose(result.iterations[0].wrms_after_m, 0.0, atol=1.0e-12)
    assert np.isclose(block.value, 10.0, atol=1.0e-12)


def test_prefit_gross_threshold_uses_station_override_then_global_default():
    # Global threshold is 20 m. STA_A has a tighter 5 m station override, so
    # its 10 m prefit residual is rejected. STA_B falls back to the global
    # threshold and stays in the solution.
    equations = [
        ObservationEquation(
            observed_minus_computed_m=10.0,
            sigma_m=1.0,
            partials={"geometry": np.array([1.0])},
            identity=1,
            station_key="STA_A",
            reflector_key="REF",
            epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
        ),
        ObservationEquation(
            observed_minus_computed_m=10.0,
            sigma_m=1.0,
            partials={"geometry": np.array([1.0])},
            identity=2,
            station_key="STA_B",
            reflector_key="REF",
            epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
        ),
        ObservationEquation(
            observed_minus_computed_m=10.0,
            sigma_m=1.0,
            partials={"geometry": np.array([1.0])},
            identity=3,
            station_key="STA_B",
            reflector_key="REF",
            epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
        ),
    ]
    block = GeometryLikeParametrization()
    result = LeastSquaresAdjustment(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([block]),
        options=AdjustmentOptions(
            max_iterations=1,
            prefit_gross_threshold_m=20.0,
            prefit_gross_threshold_by_station_m={"STA_A": 5.0},
            enable_outlier_rejection=False,
        ),
    ).run()

    assert result.rejected_keys == {1: 1}
    assert result.iterations[0].n_used == 2


def test_prefit_gross_threshold_station_null_disables_prefit_for_that_station():
    # STA_A would be rejected by the global 5 m threshold, but its explicit
    # station-specific null override disables first-pass gross rejection.
    equations = [
        ObservationEquation(
            observed_minus_computed_m=10.0,
            sigma_m=1.0,
            partials={"geometry": np.array([1.0])},
            identity=1,
            station_key="STA_A",
            reflector_key="REF",
            epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
        ),
        ObservationEquation(
            observed_minus_computed_m=10.0,
            sigma_m=1.0,
            partials={"geometry": np.array([1.0])},
            identity=2,
            station_key="STA_A",
            reflector_key="REF",
            epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
        ),
    ]
    block = GeometryLikeParametrization()
    result = LeastSquaresAdjustment(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([block]),
        options=AdjustmentOptions(
            max_iterations=1,
            prefit_gross_threshold_m=5.0,
            prefit_gross_threshold_by_station_m={"STA_A": None},
            enable_outlier_rejection=False,
        ),
    ).run()

    assert result.rejected_keys == {}
    assert result.iterations[0].n_used == 2


def test_postfit_outlier_rejection_records_observation_identity():
    equations = [_eq(1, 0.0), _eq(2, 10.0)]
    block = GeometryLikeParametrization()

    result = LeastSquaresAdjustment(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([block]),
        options=AdjustmentOptions(
            max_iterations=1,
            prefit_gross_threshold_m=None,
            enable_outlier_rejection=True,
            outlier_sigma_factor=4.0,
            allow_outlier_reentry=True,
        ),
    ).run()

    assert np.allclose(result.solution, [5.0])
    assert result.rejected_keys == {('synthetic', 1): 1, ('synthetic', 2): 1}
