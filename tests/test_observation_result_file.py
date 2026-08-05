from pathlib import Path

from lunarops.fileio.observation_results import (
    read_observation_results,
    write_observation_results,
)


def test_observation_result_file_round_trip_has_schema_and_units(tmp_path: Path):
    rows = {
        "source-a": [
            {"normal_point_index": 2, "oc_one_way_m": 0.1, "light_time_converged": True},
            {"normal_point_index": 1, "oc_one_way_m": -0.2, "light_time_converged": False},
        ]
    }
    path = tmp_path / "rows.txt.gz"
    write_observation_results(rows, path)
    recovered = read_observation_results(path)

    assert recovered == [
        {
            "source": "source-a",
            "normal_point_index": 2,
            "oc_one_way_m": 0.1,
            "light_time_converged": True,
        },
        {
            "source": "source-a",
            "normal_point_index": 1,
            "oc_one_way_m": -0.2,
            "light_time_converged": False,
        },
    ]
