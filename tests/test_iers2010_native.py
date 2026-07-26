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


def test_installed_iers_sources_match_pinned_hashes():
    expected = {
        "FCUL_A.F": "fdeb39aee3c8d4c2d6eb6a7e743c420372e28da5b3e84942d09580a88847693a",
        "FCUL_ZD_HPA.F": "92731affca053aad15a44be7db58dbf6df689e75cf2e1f3b39cb4d99a4da198b",
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
