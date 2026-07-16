from pathlib import Path

from llrops.fileio import oc_table


def test_oc_table_writes_preprojected_rows(tmp_path: Path):
    rows = {
        "source": [
            {"normal_point_index": 2, "oc_one_way_m": 0.1},
            {"normal_point_index": 1, "oc_one_way_m": -0.2},
        ]
    }
    path = tmp_path / "rows.csv"
    oc_table.write_csv_grouped(rows, path)
    text = path.read_text(encoding="utf-8")
    assert "normal_point_index,oc_one_way_m" in text
    assert "2,0.1" in text
    assert "1,-0.2" in text
