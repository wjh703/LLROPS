import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.base.parameter_name import ParameterName
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import Parametrization, ParametrizationList
from llrops.estimation.grouped_vce import (
    GroupedVceAdjustment,
    GroupedVceOptions,
    VceGroup,
    assign_vce_groups,
    igg3_factors,
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


def test_vce_assignment_distinguishes_overlapping_cerga_systems_by_wavelength():
    groups = (
        VceGroup.from_config(
            {
                "id": "CERGA_MEO",
                "station_system": "CERGA_MEO",
                "station_aliases": ["CERGA"],
                "start": "2015-01-01",
                "end_exclusive": "2023-01-01",
                "wavelength_max_nm": 700.0,
            }
        ),
        VceGroup.from_config(
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

    assert assign_vce_groups(equations, groups) == {"green": "CERGA_MEO", "infrared": "CERGA_IR"}


def test_vce_assignment_rejects_unassigned_observation():
    groups = (
        VceGroup.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    with pytest.raises(ValueError, match="no matching group"):
        assign_vce_groups([_equation(1, 0.0, "STA_B")], groups)


def test_igg3_boundaries():
    factors = igg3_factors(np.array([0.0, 1.5, 1.5001, 6.0, 6.1]), k0=1.5, k1=6.0)

    assert factors[0] == 1.0
    assert factors[1] == 1.0
    assert 0.0 < factors[2] < 1.0
    assert factors[3] == 0.0
    assert factors[4] == 0.0


def test_grouped_adjustment_runs_joint_vce_cycle():
    equations = [
        _equation(("A", index), value, "STA_A")
        for index, value in enumerate([0.7, 1.0, 1.2, 0.8, 1.1, 0.9])
    ] + [
        _equation(("B", index), value, "STA_B")
        for index, value in enumerate([0.0, 2.0, 1.7, -0.2, 2.4, 0.4])
    ]
    groups = (
        VceGroup.from_config(
            {
                "id": "A",
                "station_system": "A",
                "station_aliases": ["STA_A"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
        VceGroup.from_config(
            {
                "id": "B",
                "station_system": "B",
                "station_aliases": ["STA_B"],
                "start": "2010-01-01",
                "end_exclusive": None,
            }
        ),
    )
    result = GroupedVceAdjustment(
        equation_source=lambda iteration: equations,
        parametrization=ParametrizationList([OffsetParametrization()]),
        options=GroupedVceOptions(
            groups=groups,
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
    assert all(0.0 <= factor <= 1.0 for factor in result.robust_factors.values())
    assert result.iterations[-1].total_effective_redundancy == pytest.approx(
        result.iterations[-1].expected_total_redundancy
    )


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


from dataclasses import replace


def test_extended_vce_groups_cover_mcdonald_1969_and_current_meo():
    groups = (
        VceGroup.from_config(
            {
                "id": "MCDONALD_1969_1985",
                "station_system": "MCDONALD",
                "station_aliases": ["MCDONALD"],
                "start": "1969-01-01",
                "end_exclusive": "1986-01-01",
            }
        ),
        VceGroup.from_config(
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

    assert assign_vce_groups([mcdonald, meo], groups) == {
        "mcdonald-1969": "MCDONALD_1969_1985",
        "meo-2024": "CERGA_MEO_2009_PRESENT",
    }

