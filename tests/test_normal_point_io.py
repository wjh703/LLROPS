from pathlib import Path

import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.config.context import RunContext
from llrops.fileio.crd import convert_crd_to_mini
from llrops.fileio.llrops_normal_point_file import read_llrops_npt, write_llrops_npt
from llrops.fileio.normal_point_inputs import read_normal_points
from llrops.fileio.mini import MiniRecord, write_mini_file
from llrops.programs.normal_points_to_llrops import normal_points_to_llrops


def _write_crd(path: Path) -> None:
    path.write_text(
        "H1 CRD 2 2020 1 2 0\n"
        "H2 APOL 70610 1 1 0 APOL\n"
        "H3 APOLLO15 0 0 0 0\n"
        "H4 0 2020 1 2 0 0 0 2020 1 2 1 0 0 0 0 0 0\n"
        "C0 0 532.123 system\n"
        "20 100.123456789 900.123 280.123 50.25 0\n"
        "11 100.123456789 2.500000000123 system 2 300.5 42 12.3456 "
        "0 0 0 0 0 4.567\n",
        encoding="ascii",
    )


def test_crd_reads_directly_to_npt_without_mini_quantization_or_side_effect(tmp_path):
    source = tmp_path / "sample.crd"
    _write_crd(source)

    dataset = read_normal_points(source)
    record = dataset.records[0]

    assert len(dataset.records) == 1
    assert record.station_name == "APOL"
    assert record.station_code == "70610"
    assert record.reflector_name == "Apollo 15"
    assert record.reflector_code == "3"
    assert record.round_trip_time_s == pytest.approx(2.500000000123)
    assert record.uncertainty_two_way_s == pytest.approx(12.3456e-12)
    assert record.pressure_hpa == pytest.approx(900.123)
    assert record.temperature_k == pytest.approx(280.123)
    assert record.humidity_percent == pytest.approx(50.25)
    assert record.wavelength_nm == pytest.approx(532.123)
    assert not (tmp_path / "sample.mini").exists()


def test_mini_reads_directly_to_npt(tmp_path):
    source = tmp_path / "sample.mini"
    epoch = Epoch.from_isot("2020-01-02T00:01:40", scale=TimeScale.UTC)
    write_mini_file(
        [
            MiniRecord(
                format_code=1,
                laser_color_code=1,
                launch_date="20200102",
                launch_time="0001400000000",
                light_time_raw=25_000_000_000_000,
                reflector_id=3,
                station_id="70610",
                number_of_returns=42,
                uncertainty_raw=123,
                signal_noise_ratio_raw=46,
                quality_code=None,
                pressure_raw=90_012,
                temperature_raw=70,
                humidity_percent=50,
                wavelength_raw=5_321,
                version_code=None,
                duration_s=300,
                source_format="mini",
                launch_epoch=epoch,
                seconds_of_day=100.0,
            )
        ],
        source,
    )

    dataset = read_normal_points(source)
    record = dataset.records[0]

    assert record.station_name == "APOL"
    assert record.reflector_name == "Apollo 15"
    assert record.round_trip_time_s == pytest.approx(2.5)


def test_llrops_jsonl_round_trip_preserves_canonical_values(tmp_path):
    source = tmp_path / "sample.crd"
    target = tmp_path / "canonical.llnpt.gz"
    _write_crd(source)
    original = read_normal_points(source)

    assert write_llrops_npt(original, target) == target
    recovered = read_llrops_npt(target)
    dispatched = read_normal_points(target)

    assert recovered.name == original.name
    assert recovered.n_input_records == original.n_input_records
    assert len(recovered.records) == 1
    assert recovered.records[0].transmit_epoch == original.records[0].transmit_epoch
    assert recovered.records[0].round_trip_time_s == original.records[0].round_trip_time_s
    assert recovered.records[0].temperature_k == original.records[0].temperature_k
    assert dispatched.records[0].station_code == "70610"


def test_explicit_crd_to_mini_interchange_tool_remains_available(tmp_path):
    source = tmp_path / "sample.crd"
    target = tmp_path / "sample.mini"
    _write_crd(source)

    converted_path = convert_crd_to_mini(source, target)
    reread = read_normal_points(target)

    assert converted_path == target
    assert target.is_file()
    assert len(reread.records) == 1
    assert reread.records[0].round_trip_time_s == pytest.approx(2.500000000123)


def test_normal_points_to_llrops_program_is_repeatable_inside_input_directory(tmp_path):
    source = tmp_path / "sample.crd"
    target = tmp_path / "canonical.llnpt"
    _write_crd(source)
    context = RunContext(working_dir=tmp_path)
    config = {
        "inputNormalPoints": ["."],
        "datasetName": "campaign",
        "outputFile": "canonical.llnpt",
    }

    assert normal_points_to_llrops(config, context) == str(target)
    assert normal_points_to_llrops(config, context) == str(target)

    recovered = read_llrops_npt(target)
    assert recovered.name == "campaign"
    assert len(recovered.records) == 1
