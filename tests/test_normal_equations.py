import numpy as np

import pytest
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


def test_exactly_determined_system_has_no_posterior_variance_factor():
    names = [ParameterName("x"), ParameterName("y")]
    normals = NormalEquations.zeros(names)
    normals.accumulate(
        np.eye(2),
        np.array([1.0, 2.0]),
        np.ones(2),
    )

    solution, covariance, sigma0 = normals.solve()

    assert np.allclose(solution, [1.0, 2.0])
    assert np.allclose(covariance, np.eye(2))
    assert sigma0 is None


def test_inconsistent_residual_quadratic_form_is_rejected():
    normals = NormalEquations.zeros([ParameterName("x")])
    normals.N[0, 0] = 1.0
    normals.W[0] = 2.0
    normals.lPl = 1.0
    normals.obs_count = 2

    with pytest.raises(np.linalg.LinAlgError, match="negative residual quadratic"):
        normals.solve()


def test_normal_equation_parameter_names_must_be_unique():
    name = ParameterName("x")
    with pytest.raises(ValueError, match="must be unique"):
        NormalEquations.zeros([name, name])
