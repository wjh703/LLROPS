import numpy as np

from llrops.base.epoch import Epoch, TimeScale
from llrops.base.parameter_name import ParameterName
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import Parametrization, ParametrizationList
from llrops.estimation.adjustment import AdjustmentOptions, LeastSquaresAdjustment


class OneParameter(Parametrization):
    def __init__(self):
        self.value = 0.0

    def parameter_names(self):
        return [ParameterName("synthetic", "position.x")]

    def design_columns(self, eq):
        return np.array([1.0])

    def apply_update(self, delta):
        self.value += float(delta[0])


def _eq(value):
    return ObservationEquation(
        observed_minus_computed_m=value,
        sigma_m=1.0,
        partials={},
        identity=value,
        station_key="STA",
        reflector_key="REF",
        epoch=Epoch(2451544.5, 0.0, TimeScale.UTC),
    )


def test_iteration_callback_receives_snapshot():
    snapshots = []
    block = OneParameter()
    result = LeastSquaresAdjustment(
        equation_source=lambda iteration: [_eq(2.0), _eq(2.0)],
        parametrization=ParametrizationList([block]),
        options=AdjustmentOptions(max_iterations=1, prefit_gross_threshold_m=None, enable_outlier_rejection=False),
        iteration_callback=snapshots.append,
    ).run()

    assert len(snapshots) == 1
    assert snapshots[0].iteration.iteration == 1
    assert np.allclose(snapshots[0].delta, result.solution)
    assert snapshots[0].normals.obs_count == 2
