from pathlib import Path
from types import SimpleNamespace

from llrops.classes.observation import ObservationOutputLevel
from llrops.classes.parametrization.station_range_bias import (
    StationRangeBiasParametrization,
)
from llrops.config.context import RunContext
from llrops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    load_datasets,
    make_processing_options,
    output_level,
)


class _Dataset:
    def __init__(self, size: int) -> None:
        self.records = [SimpleNamespace(index=None) for _ in range(size)]
        self.filter_args = None

    def filter_time(self, start, end):
        self.filter_args = (start, end)
        return self

    def assign_indices(self, *, start: int) -> None:
        for offset, record in enumerate(self.records):
            record.index = start + offset


def test_load_datasets_uses_working_directory_and_assigns_global_indices(
    monkeypatch,
    tmp_path,
):
    from llrops.fileio import normal_point_inputs

    source_paths = [tmp_path / "a.npt", tmp_path / "b.npt"]
    datasets_by_path = {
        source_paths[0]: _Dataset(2),
        source_paths[1]: _Dataset(1),
    }
    captured = {}

    def resolve(paths):
        captured["inputs"] = paths
        return source_paths

    def read(path):
        return datasets_by_path[Path(path)]

    monkeypatch.setattr(normal_point_inputs, "resolve_normal_point_inputs", resolve)
    monkeypatch.setattr(normal_point_inputs, "read_normal_points", read)
    context = RunContext(working_dir=tmp_path)

    datasets = load_datasets(
        {
            "inputFilesNormalPoints": ["input"],
            "startTime": "2020-01-01",
            "endTime": "2021-01-01",
        },
        context,
    )

    assert captured["inputs"] == [tmp_path / "input"]
    assert list(datasets) == ["a", "b"]
    assert [record.index for record in datasets["a"].records] == [0, 1]
    assert [record.index for record in datasets["b"].records] == [2]
    assert datasets["a"].filter_args == ("2020-01-01", "2021-01-01")


def test_workflow_builds_canonical_options_and_parametrization():
    config = {
        "stationName": "APOLLO",
        "reflectorName": "APOLLO15",
        "minElevationDeg": 12.5,
        "showProgress": False,
        "outputLevel": "full",
        "parametrization": [{"type": "stationRangeBias", "per": "station"}],
    }
    context = RunContext()

    options = make_processing_options(config, include_design=True)
    blocks = build_parametrization(config, context)

    assert options.station_name == "APOLLO"
    assert options.reflector_name == "APOLLO15"
    assert options.min_elevation_deg == 12.5
    assert options.include_reflector_position_partial
    assert output_level(config) is ObservationOutputLevel.FULL
    assert output_level(config, include_design=True) is ObservationOutputLevel.FULL
    assert len(blocks.blocks) == 1
    assert isinstance(blocks.blocks[0], StationRangeBiasParametrization)


def test_serial_equation_source_reuses_processor_and_sets_iteration_progress():
    calls = []

    class Processor:
        def equations(self, dataset, *, options):
            calls.append((dataset, options))
            return [dataset]

    context = RunContext(runtime=None)
    datasets = {"a": object(), "b": object()}
    source = build_equation_source(
        {"showProgress": True},
        context,
        datasets,
        Processor(),
    )

    equations = source(4)

    assert equations == list(datasets.values())
    assert [item[0] for item in calls] == list(datasets.values())
    assert all(item[1].include_reflector_position_partial for item in calls)
    assert all(item[1].progress_description == "linearization 4" for item in calls)
