import numpy as np

from llrops.base.parameter_name import ParameterName
from llrops.fileio.normal_equations import NormalEquations


def test_normal_equations_use_W_and_np_solve_convention(tmp_path):
    names = [ParameterName("x"), ParameterName("y")]
    A = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, -1.0]])
    L = np.array([5.0, 11.0, 1.0])
    sigma = np.array([1.0, 2.0, 0.5])
    P = np.diag(1.0 / sigma**2)

    expected_N = A.T @ P @ A
    expected_W = A.T @ P @ L

    normals = NormalEquations.zeros(names)
    normals.accumulate(A, L, sigma)

    assert np.allclose(normals.N, expected_N)
    assert np.allclose(normals.W, expected_W)

    x, Qxx, sigma0 = normals.solve()
    assert np.allclose(x, np.linalg.solve(expected_N, expected_W))
    assert np.allclose(Qxx, np.linalg.solve(expected_N, np.eye(2)))
    assert sigma0 >= 0.0

    stem = tmp_path / "normals"
    normals.save(stem)
    loaded = NormalEquations.load(stem)
    assert np.allclose(loaded.N, expected_N)
    assert np.allclose(loaded.W, expected_W)
