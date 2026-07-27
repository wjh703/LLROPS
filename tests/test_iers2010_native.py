from __future__ import annotations

import hashlib
import importlib.resources
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from llrops import _iers2010


_FCUL_ZD_EXPECTED_M = np.array(
    [
        1.935225924846803114,
        1.932992176591644462,
        0.002233748255158703871,
    ]
)

_HARDISP_ONSALA_AMP = np.array(
    [
        [0.00352, 0.00123, 0.00080, 0.00032, 0.00187, 0.00112, 0.00063, 0.00003, 0.00082, 0.00044, 0.00037],
        [0.00144, 0.00035, 0.00035, 0.00008, 0.00053, 0.00049, 0.00018, 0.00009, 0.00012, 0.00005, 0.00006],
        [0.00086, 0.00023, 0.00023, 0.00006, 0.00029, 0.00028, 0.00010, 0.00007, 0.00004, 0.00002, 0.00001],
    ]
)
_HARDISP_ONSALA_PHASE = np.array(
    [
        [-64.7, -52.0, -96.2, -55.2, -58.8, -151.4, -65.6, -138.1, 8.4, 5.2, 2.1],
        [85.5, 114.5, 56.5, 113.6, 99.4, 19.1, 94.1, -10.4, -167.4, -170.0, -177.7],
        [109.5, 147.0, 92.7, 148.8, 50.5, -55.1, 36.4, -170.4, -15.0, 2.3, 5.2],
    ]
)
_HARDISP_ONSALA_EXPECTED = np.array(
    [
        [0.003094, -0.001538, -0.000895],
        [0.001812, -0.000950, -0.000193],
        [0.000218, -0.000248, 0.000421],
        [-0.001104, 0.000404, 0.000741],
        [-0.001668, 0.000863, 0.000646],
        [-0.001209, 0.001042, 0.000137],
        [0.000235, 0.000926, -0.000667],
        [0.002337, 0.000580, -0.001555],
        [0.004554, 0.000125, -0.002278],
        [0.006271, -0.000291, -0.002615],
        [0.006955, -0.000537, -0.002430],
        [0.006299, -0.000526, -0.001706],
        [0.004305, -0.000244, -0.000559],
        [0.001294, 0.000245, 0.000793],
        [-0.002163, 0.000819, 0.002075],
        [-0.005375, 0.001326, 0.003024],
        [-0.007695, 0.001622, 0.003448],
        [-0.008669, 0.001610, 0.003272],
        [-0.008143, 0.001262, 0.002557],
        [-0.006290, 0.000633, 0.001477],
        [-0.003566, -0.000155, 0.000282],
        [-0.000593, -0.000941, -0.000766],
        [0.001992, -0.001561, -0.001457],
        [0.003689, -0.001889, -0.001680],
    ]
)

_HARDISP_UTC_REGRESSION_CASES = (
    (
        (2016, 12, 31, 23, 59, 59),
        np.array([0.00473209097981453, -0.00054359360365197, -0.00126003834884614]),
    ),
    (
        (2017, 1, 1, 0, 0, 0),
        np.array([0.00473177339881659, -0.00054353591986001, -0.00125983613543212]),
    ),
    (
        (2009, 6, 25, 1, 10, 45),
        np.array([0.00309408246539533, -0.00153829867485911, -0.00089535955339670]),
    ),
)


def test_fcul_a_matches_iers_reference_case():
    mapping = _iers2010.fcul_a(30.67166667, 2075.0, 300.15, 15.0)

    assert mapping == pytest.approx(3.800243667312344087, rel=0.0, abs=1.0e-15)


def test_fculzd_hpa_matches_iers_reference_outputs():
    # The v1.3.0 source header says 2010.344 m, but its three reference
    # outputs are exactly reproduced with 2003.344 m. The upstream source is
    # preserved unchanged and the discrepancy is recorded in the build notes.
    actual = _iers2010.fculzd_hpa(
        30.67166667,
        2003.344,
        798.4188,
        14.322,
        0.532,
    )

    np.testing.assert_allclose(actual, _FCUL_ZD_EXPECTED_M, rtol=0.0, atol=1.0e-15)


def test_ortho_eop_matches_iers_reference_case():
    actual = _iers2010.ortho_eop(47100.0)
    np.testing.assert_allclose(
        actual,
        (-162.8386373279636530, 117.7907525842668974, -23.39092370609808214),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_pmsdnut2_matches_iers_reference_case():
    actual = _iers2010.pmsdnut2(54335.0)
    np.testing.assert_allclose(
        actual,
        (24.83144238273364834, -14.09240692041837661),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_utlibr_matches_iers_reference_cases():
    np.testing.assert_allclose(
        _iers2010.utlibr(44239.1),
        (2.441143834386761746, -14.78971247349449492),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _iers2010.utlibr(55227.4),
        (-2.655705844335680244, 27.39445826599846967),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_fundarg_matches_iers_reference_case():
    actual = _iers2010.fundarg(0.07995893223819302)
    np.testing.assert_allclose(
        actual,
        (2.291187512612069099, 6.212931111003726414,
         3.658025792050572989, 4.554139562402433228,
         -0.5167379217231804489),
        rtol=0.0,
        atol=2.0e-11,
    )


@pytest.mark.parametrize(
    ("xsta", "xsun", "xmon", "date", "expected"),
    [
        (
            (4075578.385, 931852.890, 4801570.154),
            (137859926952.015, 54228127881.4350, 23509422341.6960),
            (-179996231.920342, -312468450.131567, -169288918.592160),
            (2009, 4, 13, 0.0),
            (0.7700420357108125891e-1, 0.6304056321824967613e-1, 0.5516568152597246810e-1),
        ),
        (
            (1112189.660, -4842955.026, 3985352.284),
            (-54537460436.2357, 130244288385.279, 56463429031.5996),
            (300396716.912, 243238281.451, 120548075.939),
            (2012, 7, 13, 0.0),
            (-0.2036831479592075833e-1, 0.5658254776225972449e-1, -0.7597679676871742227e-1),
        ),
        (
            (1112200.5696, -4842957.8511, 3985345.9122),
            (100210282451.6279, 103055630398.3160, 56855096480.4475),
            (369817604.4348, 1897917.5258, 120804980.8284),
            (2015, 7, 15, 0.0),
            (0.00509570869172363845, 0.0828663025983528700, -0.0636634925404189617),
        ),
        (
            (1112152.8166, -4842857.5435, 3985496.1783),
            (8382471154.1312895, 10512408445.356153, -5360583240.3763866),
            (380934092.93550891, 2871428.1904491195, 79015680.553570181),
            (2017, 1, 15, 0.0),
            (-18.217357581922339, -23.505348376537949, 12.097611382175685),
        ),
    ],
)
def test_dehanttideinel_matches_iers_reference_and_source_cases(xsta, xsun, xmon, date, expected):
    # The v1.3.0 header's 2017 expected output is copied from the preceding
    # 2015 case despite a solar vector only 14.5 Gm long. Its source-input
    # calculation is retained as the regression value; see IERS_NATIVE_BUILD.
    actual = _iers2010.dehanttideinel(
        xsta,
        date[0],
        date[1],
        date[2],
        date[3],
        xsun,
        xmon,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_hardisp_matches_official_onsala_case():
    assert not hasattr(_iers2010, "llrops_hardisp")

    actual = np.column_stack(
        _iers2010.hardisp(
            2009,
            6,
            25,
            1,
            10,
            45,
            24,
            3600.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        )
    )
    np.testing.assert_allclose(actual, _HARDISP_ONSALA_EXPECTED, rtol=0.0, atol=6.0e-7)


@pytest.mark.parametrize(("calendar", "expected"), _HARDISP_UTC_REGRESSION_CASES)
def test_hardisp_utc_leap_boundary_and_nonzero_time(calendar, expected):
    actual = np.asarray(
        _iers2010.hardisp(
            *calendar,
            1,
            1.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        ),
        dtype=float,
    ).reshape(3)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-9)


def test_hardisp_utc_leap_second_label_matches_following_midnight():
    leap_second = np.asarray(
        _iers2010.hardisp(
            2016,
            12,
            31,
            23,
            59,
            60,
            1,
            1.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        ),
        dtype=float,
    ).reshape(3)
    following_midnight = np.asarray(
        _iers2010.hardisp(
            2017,
            1,
            1,
            0,
            0,
            0,
            1,
            1.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        ),
        dtype=float,
    ).reshape(3)

    # TDFRPH forms DAYFR with a fixed 86400-second day, so its scalar
    # calendar interface cannot represent this extra UTC label separately.
    np.testing.assert_array_equal(leap_second, following_midnight)


def test_hardisp_regular_series_matches_individual_epochs():
    series = np.column_stack(
        _iers2010.hardisp(
            2009,
            6,
            25,
            1,
            10,
            45,
            4,
            900.0,
            _HARDISP_ONSALA_AMP,
            _HARDISP_ONSALA_PHASE,
        )
    )
    scalar = np.array(
        [
            np.asarray(
                _iers2010.hardisp(
                    2009,
                    6,
                    25,
                    1,
                    minute,
                    45,
                    1,
                    1.0,
                    _HARDISP_ONSALA_AMP,
                    _HARDISP_ONSALA_PHASE,
                ),
                dtype=float,
            ).reshape(3)
            for minute in (10, 25, 40, 55)
        ]
    )
    np.testing.assert_allclose(series, scalar, rtol=0.0, atol=5.0e-9)


def test_installed_iers_sources_match_pinned_hashes():
    expected = {
        "FCUL_A.F": "fdeb39aee3c8d4c2d6eb6a7e743c420372e28da5b3e84942d09580a88847693a",
        "FCUL_ZD_HPA.F": "92731affca053aad15a44be7db58dbf6df689e75cf2e1f3b39cb4d99a4da198b",
        "ORTHO_EOP.F": "dfd1524b583f2a0f11baf2f03282d0f5ba5731026ac1fdaff4aa6e9460995022",
        "CNMTX.F": "8a29c599275110990e6ce93254995d498edbccc523edb2de508455736f45fc93",
        "PMSDNUT2.F": "0818b58bc2a420e1eb3f951d8a74646e5fe7b5371c9beb5e89fa37c12dd0d965",
        "UTLIBR.F": "f523335d552ac14b661121a081ad799382312d819853c674bc0102484b5e2406",
        "FUNDARG.F": "18263cbb1289e222e6ee6e59d52beb343eb77a63ed3212e4f05a4c85d475ae78",
        "DEHANTTIDEINEL.F": "bc6039a1704761881bb785ce44ce084ea82783107ff64c576e69155a4914e2cb",
        "CAL2JD.F": "7634fafdbc761e9e97699102b0d43fdb564916d556497c98505a3205b3aef923",
        "DAT.F": "4692aa5b784070cab731dd05d541f819d6ae01b20ba339a96d85ced0ffb643dc",
        "NORM8.F": "636b6399dc6ab273a7b6104dd9341bf3c36995754aeee696511dbf19c9e909b6",
        "SPROD.F": "817761a92bb5416eb38322ea2d43d41cf8ea435e208a0ce229acf94311e9fa1e",
        "ST1IDIU.F": "d2976b8b76be8dd1d57e57a8d6b48f5764676126515bada3592753a07d3acd1e",
        "ST1ISEM.F": "efdf284bd977826a1f4aea4c79c5dbd0c38fc1c403a2376010048b511a11f2c6",
        "ST1L1.F": "b1dfd0e797a3ce950631ad7dbbf5f576bf6b58dc515688253a3d84ca059bc282",
        "STEP2DIU.F": "898c70d4b8d50e09e0c717c911c4117b3ad1ca4996d369c258b68d00ea3a5674",
        "STEP2LON.F": "f9d3bf0317222986d22e53557020bb13a6fbb90f8e3c9915137da6184d82813a",
        "ZERO_VEC8.F": "5ea9ab87e298d377f6dbe69c46b040906b6575a9132fab3830a4dc02c9139cef",
        "ADMINT.F": "478e3f0c001c09dd4d9e2e9920033d29e76eb60b808238e416e6db9ffaeae6c8",
        "ETUTC.F": "a067c10de2c63269a54b6a14dce2daa49c30bfd52e66086b76492cd438414e1f",
        "EVAL.F": "b4056a09ec5d77674cab8eac0fa32b5d1be8f471fbe33e753e81ab97a7767add",
        "HARDISP_WRAP.F": "e8586808dad1355239f2919fffd5b150ee2f6d4547c26e50b299f8b138e525d5",
        "JULDAT.F": "d1f39f83503178711845532e1a4ba46d0824b896fa961d81c983d694d48dde73",
        "LEAP.F": "31d18153242823606beb9690ab0c685dd5403fe1e9bb5f214e22d21dd5e6a771",
        "MDAY.F": "d4eae8a9a0b866a63f22134fb57bad1c57fc1f3baf98450892ddeee4f7ee8aec",
        "RECURS.F": "8a3c88d69cebd130981887dc9c8c2f9e3d2af26fded3e052385c022126f9d44b",
        "SHELLS.F": "6c69dede79f9adfb16d38dcbd890bbbfc3e4e47b2cbacd5fabfb520deba86701",
        "SPLINE.F": "a13cc2b405079a2b86872341ca50d9b3f0cc219e43d1ce301b461af15df857f0",
        "TDFRPH.F": "24068e0cd1e2e210fab7dd8d4473bb60ff8335bfcb53a2fc12dad7e6d1cccd19",
        "TOYMD.F": "9b69b6a27d544215c516da60351e4ec51d03b04ce9bf71b6608ebf55080bf70a",
    }
    root = importlib.resources.files("llrops").joinpath("_external", "iers2010", "src")

    for filename, expected_hash in expected.items():
        source = root.joinpath(filename)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash

    assert not root.joinpath("HARDISP.F").is_file()
    assert not root.joinpath("LLROPS_HARDISP.F").is_file()
    license_text = importlib.resources.files("llrops").joinpath(
        "_external", "iers2010", "LICENSE"
    ).read_text(encoding="utf-8")
    assert "IERS Conventions Software License" in license_text
    assert "This notice must be reproduced intact" in license_text


def test_native_extension_imports_in_mpi_workers():
    mpi_runner = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpi_runner is None:
        pytest.skip("MPI launcher is not installed")
    pytest.importorskip("mpi4py")
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        pytest.skip("do not start a nested MPI job")

    worker_code = """
from mpi4py import MPI
from llrops import _iers2010

value = _iers2010.fcul_a(30.67166667, 2075.0, 300.15, 15.0)
values = MPI.COMM_WORLD.allgather(value)
assert len(values) == 2
assert all(abs(item - 3.800243667312344087) < 1.0e-15 for item in values)
"""
    env = os.environ.copy()
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    subprocess.run(
        [mpi_runner, "-n", "2", sys.executable, "-c", worker_code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
