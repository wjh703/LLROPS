from __future__ import annotations

import numpy as np
import pytest

from llrops.base.epoch import Epoch, TimeScale
from llrops.classes.displacement import (
    Iers2010OceanTidalLoading,
    OceanTidalLoadingCatalog,
    StationDisplacementInput,
)
from llrops.classes.observation_factory import ensure_registered
from llrops.config.context import RunContext


_AMPLITUDES = np.array(
    [
        [0.00352, 0.00123, 0.00080, 0.00032, 0.00187, 0.00112, 0.00063, 0.00003, 0.00082, 0.00044, 0.00037],
        [0.00144, 0.00035, 0.00035, 0.00008, 0.00053, 0.00049, 0.00018, 0.00009, 0.00012, 0.00005, 0.00006],
        [0.00086, 0.00023, 0.00023, 0.00006, 0.00029, 0.00028, 0.00010, 0.00007, 0.00004, 0.00002, 0.00001],
    ]
)
_PHASES = np.array(
    [
        [-64.7, -52.0, -96.2, -55.2, -58.8, -151.4, -65.6, -138.1, 8.4, 5.2, 2.1],
        [85.5, 114.5, 56.5, 113.6, 99.4, 19.1, 94.1, -10.4, -167.4, -170.0, -177.7],
        [109.5, 147.0, 92.7, 148.8, 50.5, -55.1, 36.4, -170.4, -15.0, 2.3, 5.2],
    ]
)


def _row(values: np.ndarray) -> str:
    return " ".join(f"{value:.5f}" for value in values)


def _blq_text(*, station_name: str = "APOLLO", model: str = "FES2022b") -> str:
    return "\n".join(
        [
            "$$ Ocean loading displacement",
            "$$ COLUMN ORDER: M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA",
            "$$ CMC: NO (corr.tide centre of mass)",
            f"$$ {model}: M2 S2 N2 K2 K1 O1",
            f"$$ {model}: P1 Q1 MF MM SSA",
            "$$ END HEADER",
            f"  {station_name}",
            f"$$ {station_name} RADI TANG lon/lat: 0.0000 0.0000 0.0000",
            *[_row(row) for row in _AMPLITUDES],
            *[_row(row) for row in _PHASES],
            "$$ END TABLE",
            "",
        ]
    )


def _station_input(*, station_id: str | None = "APOLLO") -> StationDisplacementInput:
    return StationDisplacementInput(
        station_itrf_m=(6_378_137.0, 0.0, 0.0),
        epoch_utc=Epoch.from_isot("2009-06-25T01:10:45", scale=TimeScale.UTC),
        station_id=station_id,
    )


def test_onsala_blq_catalog_parses_metadata_and_station_coefficients(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")

    catalog = OceanTidalLoadingCatalog(coefficient_file)
    coefficients = catalog.coefficients_for("7045")

    assert catalog.station_ids == ("APOLLO",)
    assert catalog.info.station_count == 1
    assert catalog.info.tidal_model == "FES2022b"
    assert catalog.info.center_of_mass_correction is False
    assert coefficients.station_id == "APOLLO"
    assert coefficients.source_station_name == "APOLLO"
    np.testing.assert_allclose(coefficients.amplitudes_m, _AMPLITUDES)
    np.testing.assert_allclose(coefficients.phases_deg, _PHASES)
    assert not coefficients.amplitudes_m.flags.writeable
    assert not coefficients.phases_deg.flags.writeable


def test_onsala_blq_catalog_rejects_malformed_and_duplicate_station_blocks(tmp_path):
    malformed = tmp_path / "malformed.blq"
    malformed.write_text(_blq_text().replace(_row(_AMPLITUDES[0]), "0.1 0.2"), encoding="utf-8")
    with pytest.raises(ValueError, match="11 values"):
        OceanTidalLoadingCatalog(malformed)

    duplicate = tmp_path / "duplicate.blq"
    duplicate.write_text(
        _blq_text().replace("$$ END TABLE", "  APOLLO\n" + "\n".join(
            [*[_row(row) for row in _AMPLITUDES], *[_row(row) for row in _PHASES], "$$ END TABLE"]
        )),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate BLQ station"):
        OceanTidalLoadingCatalog(duplicate)

    bad_order = tmp_path / "bad-order.blq"
    bad_order.write_text(
        _blq_text().replace("M2 S2 N2 K2 K1 O1 P1 Q1 MF MM SSA", "S2 M2 N2 K2 K1 O1 P1 Q1 MF MM SSA", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="column order"):
        OceanTidalLoadingCatalog(bad_order)


def test_ocean_tidal_loading_passes_blq_unchanged_and_converts_usw_to_itrf(tmp_path, monkeypatch):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))

    received = {}

    def hardisp(year, month, day, hour, minute, second, n, sample, amplitudes, phases):
        received["calendar"] = (year, month, day, hour, minute, second)
        received["n"] = n
        received["sample"] = sample
        received["amplitudes"] = np.array(amplitudes, copy=True)
        received["phases"] = np.array(phases, copy=True)
        return np.array([0.003]), np.array([0.002]), np.array([-0.001])

    import llrops.classes.displacement.ocean_tidal_loading as ocean_tidal_loading

    monkeypatch.setattr(ocean_tidal_loading._iers2010, "hardisp", hardisp)
    result = model.evaluate(_station_input())

    assert received["calendar"] == (2009, 6, 25, 1, 10, 45)
    assert received["n"] == 1
    assert received["sample"] == 1.0
    np.testing.assert_allclose(received["amplitudes"], _AMPLITUDES)
    np.testing.assert_allclose(received["phases"], _PHASES)
    np.testing.assert_allclose(result.displacement_up_south_west_m, [0.003, 0.002, -0.001])
    np.testing.assert_allclose(result.displacement_enu_m, [0.001, -0.002, 0.003])
    np.testing.assert_allclose(result.displacement_itrf_m, [0.003, 0.001, -0.002])


def test_ocean_tidal_loading_requires_a_station_id_present_in_the_blq_file(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))

    with pytest.raises(ValueError, match="station_id"):
        model.displacement_itrf_m(_station_input(station_id=None))
    with pytest.raises(KeyError, match="WETTZELL"):
        model.displacement_itrf_m(_station_input(station_id="WETTZELL"))


def test_ocean_tidal_loading_preserves_utc_leap_second_calendar_fields(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    model = Iers2010OceanTidalLoading(OceanTidalLoadingCatalog(coefficient_file))

    inputs = (
        ("2016-12-31T23:59:59", (2016, 12, 31, 23, 59, 59)),
        ("2016-12-31T23:59:60", (2016, 12, 31, 23, 59, 60)),
        ("2017-01-01T00:00:00", (2017, 1, 1, 0, 0, 0)),
    )
    for text, expected_calendar in inputs:
        data = StationDisplacementInput(
            station_itrf_m=(6_378_137.0, 0.0, 0.0),
            epoch_utc=Epoch.from_isot(text, scale=TimeScale.UTC),
            station_id="APOLLO",
        )
        assert model._utc_calendar_second(data.epoch_utc) == expected_calendar
        assert np.all(np.isfinite(model.displacement_itrf_m(data)))


def test_ocean_tidal_loading_factory_checks_model_and_uses_explicit_file(tmp_path):
    coefficient_file = tmp_path / "fes2022b.txt"
    coefficient_file.write_text(_blq_text(), encoding="utf-8")
    ensure_registered()
    context = RunContext(working_dir=tmp_path)

    model = context.create_class(
        "stationDisplacement",
        {
            "type": "iers2010OceanTidalLoading",
            "coefficientFile": "fes2022b.txt",
            "model": "FES2022b",
        },
        cache=False,
    )
    assert isinstance(model, Iers2010OceanTidalLoading)

    with pytest.raises(ValueError, match="model mismatch"):
        context.create_class(
            "stationDisplacement",
            {
                "type": "iers2010OceanTidalLoading",
                "coefficientFile": "fes2022b.txt",
                "model": "FES2014b",
            },
            cache=False,
        )
