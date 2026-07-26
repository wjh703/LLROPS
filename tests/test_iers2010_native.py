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


def test_installed_iers_sources_match_pinned_hashes():
    expected = {
        "FCUL_A.F": "fdeb39aee3c8d4c2d6eb6a7e743c420372e28da5b3e84942d09580a88847693a",
        "FCUL_ZD_HPA.F": "92731affca053aad15a44be7db58dbf6df689e75cf2e1f3b39cb4d99a4da198b",
        "ORTHO_EOP.F": "dfd1524b583f2a0f11baf2f03282d0f5ba5731026ac1fdaff4aa6e9460995022",
        "CNMTX.F": "8a29c599275110990e6ce93254995d498edbccc523edb2de508455736f45fc93",
        "PMSDNUT2.F": "0818b58bc2a420e1eb3f951d8a74646e5fe7b5371c9beb5e89fa37c12dd0d965",
        "UTLIBR.F": "f523335d552ac14b661121a081ad799382312d819853c674bc0102484b5e2406",
        "FUNDARG.F": "18263cbb1289e222e6ee6e59d52beb343eb77a63ed3212e4f05a4c85d475ae78",
    }
    root = importlib.resources.files("llrops").joinpath("_external", "iers2010", "src")

    for filename, expected_hash in expected.items():
        source = root.joinpath(filename)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash


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
