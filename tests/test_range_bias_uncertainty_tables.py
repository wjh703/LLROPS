from pathlib import Path

import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.range_bias.table import (
    INPOP21_RANGE_BIAS_TABLE,
    RangeBiasTable,
    load_range_bias_table,
)
from llrops.classes.uncertainty.wrms_table import (
    DEFAULT_WRMS_UNCERTAINTY_TABLE,
    WrmsUncertaintyTable,
    load_wrms_uncertainty_table,
)


def test_builtin_range_bias_table_lookup_and_summary():
    epoch = Epoch.from_isot("2008-01-01T00:00:00", scale=TimeScale.UTC)
    assert INPOP21_RANGE_BIAS_TABLE.two_way_cm(["7045"], epoch) == INPOP21_RANGE_BIAS_TABLE.two_way_cm(["APOLLO"], epoch)
    assert "APOLLO" in INPOP21_RANGE_BIAS_TABLE.coverage_summary()


def test_wrms_table_lookup_and_summary():
    epoch = Epoch.from_isot("2020-01-01T00:00:00", scale=TimeScale.UTC)
    entry = DEFAULT_WRMS_UNCERTAINTY_TABLE.entry(["7045"], epoch)
    assert entry is not None
    assert entry.group == "APO-f"
    assert "APOLLO" in DEFAULT_WRMS_UNCERTAINTY_TABLE.coverage_summary()


def test_declarative_range_bias_table_from_yaml_compact_rows(tmp_path: Path):
    path = tmp_path / "range_bias.yml"
    path.write_text(
        """
biases:
  - APOLLO 2020-01-01/2021-01-01 12.5
  - {station: GRASSE, interval: 2020-01-01/2021-01-01, biasCm: -0.5}
""".strip(),
        encoding="utf-8",
    )
    table = load_range_bias_table(path)
    epoch = Epoch.from_isot("2020-06-01T00:00:00", scale=TimeScale.UTC)
    assert isinstance(table, RangeBiasTable)
    assert table.source == str(path)
    assert table.two_way_cm(["APOLLO"], epoch) == 12.5
    assert table.two_way_cm(["GRASSE"], epoch) == -0.5


def test_declarative_range_bias_rejects_old_entries_key():
    with pytest.raises(ValueError, match="biases"):
        RangeBiasTable.from_mapping(
            {
                "entries": [
                    {"station": "APOLLO", "start": "2020-01-01", "end": "2021-01-01", "biasTwoWayCm": 1.0}
                ],
            }
        )


def test_declarative_range_bias_rejects_table_name_and_aliases():
    for key in ("name", "aliases"):
        with pytest.raises(ValueError, match=key):
            RangeBiasTable.from_mapping(
                {
                    key: {"TEST": "APOLLO"} if key == "aliases" else "custom",
                    "biases": ["APOLLO 2020-01-01/2021-01-01 1.0"],
                }
            )


def test_declarative_wrms_table_from_mapping_compact_rows():
    table = WrmsUncertaintyTable.from_mapping(
        {
            "source": "custom-wrms",
            "uncertainties": [
                "APOLLO 2020-01-01/2021-01-01 0.020 APO-test",
                {"station": "GRASSE", "interval": "2020-01-01/2021-01-01", "wrmsM": 0.030, "group": "GRA-test"},
            ],
        }
    )
    epoch = Epoch.from_isot("2020-06-01T00:00:00", scale=TimeScale.UTC)
    entry = table.entry(["APOLLO"], epoch)
    assert entry is not None
    assert entry.group == "APO-test"
    assert entry.wrms_two_way_m == 0.02
    assert table.source == "custom-wrms"
    grasse = table.entry(["GRASSE"], epoch)
    assert grasse is not None
    assert grasse.group == "GRA-test"


def test_declarative_wrms_table_from_yaml(tmp_path: Path):
    path = tmp_path / "wrms.yml"
    path.write_text(
        """
uncertainties:
  - APOLLO 2020-01-01/2021-01-01 0.020 APO-test
""".strip(),
        encoding="utf-8",
    )
    table = load_wrms_uncertainty_table(path)
    epoch = Epoch.from_isot("2020-06-01T00:00:00", scale=TimeScale.UTC)
    entry = table.entry(["APOLLO"], epoch)
    assert entry is not None
    assert entry.group == "APO-test"


def test_declarative_wrms_rejects_table_name_and_aliases():
    for key in ("name", "aliases"):
        with pytest.raises(ValueError, match=key):
            WrmsUncertaintyTable.from_mapping(
                {
                    key: {"TEST": "APOLLO"} if key == "aliases" else "custom",
                    "uncertainties": ["APOLLO 2020-01-01/2021-01-01 0.020 APO-test"],
                }
            )


def test_declarative_tables_canonicalize_station_aliases_on_load():
    epoch = Epoch.from_isot("2020-06-01T00:00:00", scale=TimeScale.UTC)
    range_table = RangeBiasTable.from_mapping(
        {"biases": [{"station": "APOL", "start": "2020-01-01", "end": "2021-01-01", "biasCm": 7.0}]}
    )
    wrms_table = WrmsUncertaintyTable.from_mapping(
        {
            "uncertainties": [
                {"station": "APOL", "start": "2020-01-01", "end": "2021-01-01", "wrmsM": 0.02}
            ]
        }
    )

    assert range_table.two_way_cm(["APOL", "70610"], epoch) == 7.0
    entry = wrms_table.entry(["APOL", "70610"], epoch)
    assert entry is not None
    assert entry.station == "APOLLO"
