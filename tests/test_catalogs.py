import numpy as np
import pytest

from llrops.fileio.catalogs import (
    StationRecord,
    load_reflector_catalog,
    load_station_catalog,
    resolve_catalog_key,
)


def test_resolve_catalog_key_exact_case_compact_and_alias():
    catalog = {
        "Foo-Bar": StationRecord(
            name="Foo Bar Station",
            itrf_xyz_m=(1.0, 2.0, 3.0),
            aliases=("FB-01",),
        )
    }

    assert resolve_catalog_key("Foo-Bar", catalog, "Station") == "Foo-Bar"
    assert resolve_catalog_key("foo-bar", catalog, "Station") == "Foo-Bar"
    assert resolve_catalog_key("foobar", catalog, "Station") == "Foo-Bar"
    assert resolve_catalog_key("fb01", catalog, "Station") == "Foo-Bar"
    with pytest.raises(KeyError):
        resolve_catalog_key("missing", catalog, "Station")


def test_builtin_catalog_loaders_return_deep_copies():
    stations_1 = load_station_catalog("builtin")
    stations_2 = load_station_catalog("builtin")
    station_key = next(iter(stations_1))
    stations_1[station_key].name = "POLLUTED"
    assert stations_2[station_key].name != "POLLUTED"
    assert stations_1[station_key] is not stations_2[station_key]

    reflectors_1 = load_reflector_catalog("builtin")
    reflectors_2 = load_reflector_catalog("builtin")
    reflector_key = next(iter(reflectors_1))
    original = np.asarray(reflectors_2[reflector_key].moon_fixed_xyz_m, dtype=float)
    reflectors_1[reflector_key].moon_fixed_xyz_m = np.array([1.0, 2.0, 3.0])
    assert np.allclose(reflectors_2[reflector_key].moon_fixed_xyz_m, original)
    assert reflectors_1[reflector_key] is not reflectors_2[reflector_key]
