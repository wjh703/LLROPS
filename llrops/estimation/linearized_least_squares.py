"""Streaming normal-equation utilities shared by adjustment programs.

The public programs have different responsibilities:

* ``LlrAdjustment`` controls nonlinear Gauss--Newton iteration, outlier
  handling, convergence and update absorption.
* ``LlrNormalEquations`` writes fixed-linearization normal-equation files.
* ``NormalsCombineSolve`` loads, adds and solves previously written files.

This module contains the common linearized least-squares core.  It never
materializes the full design matrix; each observation row is converted to a
weighted contribution and added directly to ``N, W, lPl``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import ParametrizationList
from llrops.fileio.normal_equations import NormalEquations


@dataclass(frozen=True)
class PostfitResidual:
    """One linearized post-fit residual, evaluated before update absorption."""

    equation: ObservationEquation
    reduced_observation_m: float
    sigma_m: float
    residual_m: float


@dataclass(frozen=True)
class NormalEquationSolution:
    """Solution of one fixed-linearization normal-equation system."""

    delta: np.ndarray
    covariance: Optional[np.ndarray]
    sigma0_post: Optional[float]
    method: str
    rank_deficient: bool = False


class NormalEquationSingularError(np.linalg.LinAlgError):
    """Raised when a normal-equation matrix cannot be solved strictly."""


@dataclass(frozen=True)
class DenseLinearization:
    """Materialized fixed-linearization system for repeated reweighting."""

    equations: tuple[ObservationEquation, ...]
    parameter_names: tuple[ParameterName, ...]
    design: np.ndarray
    reduced_observations: np.ndarray
    sigmas: np.ndarray
    identities: tuple[object, ...]

    @classmethod
    def build(
        cls,
        equations: Sequence[ObservationEquation],
        parametrization: ParametrizationList,
        parameter_names: Sequence[ParameterName],
    ) -> "DenseLinearization":
        rows = tuple(equations)
        design = np.vstack([parametrization.design_row(eq) for eq in rows])
        reduced = np.asarray(
            [parametrization.reduced_observation(eq) for eq in rows], dtype=float
        )
        sigmas = np.asarray([eq.sigma_m for eq in rows], dtype=float)
        for array in (design, reduced, sigmas):
            array.setflags(write=False)
        return cls(
            equations=rows,
            parameter_names=tuple(parameter_names),
            design=design,
            reduced_observations=reduced,
            sigmas=sigmas,
            identities=tuple(eq.identity for eq in rows),
        )

    def normal_equations(
        self,
        weights: np.ndarray,
        *,
        active: Optional[np.ndarray] = None,
    ) -> NormalEquations:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != len(self.equations):
            raise ValueError("Dense weights do not match the observation count.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Dense weights must be finite and non-negative.")
        mask = weights > 0.0 if active is None else np.asarray(active, dtype=bool)
        if mask.shape != weights.shape:
            raise ValueError("Dense active mask does not match the observation count.")
        A = self.design[mask]
        l = self.reduced_observations[mask]
        w = weights[mask]
        weighted_A = w[:, None] * A
        normal_matrix = A.T @ weighted_A
        normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
        return NormalEquations(
            parameter_names=list(self.parameter_names),
            N=normal_matrix,
            W=A.T @ (w * l),
            lPl=float(np.dot(w, l * l)),
            obs_count=int(np.count_nonzero(mask)),
        )


def build_normal_equations_streaming(
    equations: Iterable[ObservationEquation],
    parametrization: ParametrizationList,
    *,
    parameter_names: Optional[Sequence[ParameterName]] = None,
    weight_for: Optional[Callable[[ObservationEquation], float]] = None,
    **meta,
) -> NormalEquations:
    """Build normal equations by streaming over observation equations.

    Parameters
    ----------
    equations
        Linearized observation equations at one fixed model state.
    parametrization
        Concatenated parameter blocks that provide design rows and reduced
        observations.
    parameter_names
        Optional explicit name list.  Supplying it avoids recomputing names and
        guarantees the same column order across iterations/programs.
    meta
        Metadata stored in the resulting :class:`NormalEquations` object.
    """
    names = list(parameter_names if parameter_names is not None else parametrization.parameter_names())
    normals = NormalEquations.zeros(names, **meta)
    for eq in equations:
        entries = parametrization.design_entries(eq)
        reduced = parametrization.reduced_observation(eq)
        if weight_for is None:
            normals.accumulate_sparse_row(entries, reduced, eq.sigma_m)
        else:
            normals.accumulate_sparse_row(entries, reduced, weight=float(weight_for(eq)))
    return normals


def solve_normal_equations(normals: NormalEquations) -> NormalEquationSolution:
    """Solve ``N x = W`` for one fixed-linearization system.

    The normal-equation route uses :func:`numpy.linalg.solve` directly.  It does
    not fall back to a materialized design-matrix ``lstsq`` solve or to a
    pseudo-inverse; singular systems should be handled by changing the
    parametrization, fixing interval overlap, or reducing the parameter set.
    """
    try:
        delta, Qxx, sigma0 = normals.solve()
    except np.linalg.LinAlgError as exc:
        diagnostics = normal_matrix_rank_diagnostics(normals)
        raise NormalEquationSingularError(diagnostics) from exc
    return NormalEquationSolution(
        delta=np.asarray(delta, dtype=float),
        covariance=Qxx,
        sigma0_post=sigma0,
        method="cholesky",
        rank_deficient=False,
    )


def postfit_residuals_streaming(
    equations: Iterable[ObservationEquation],
    parametrization: ParametrizationList,
    delta: np.ndarray,
) -> Iterator[PostfitResidual]:
    """Yield linearized post-fit residuals ``v = l - a @ delta`` row by row."""
    delta = np.asarray(delta, dtype=float).reshape(-1)
    for eq in equations:
        l = parametrization.reduced_observation(eq)
        yield PostfitResidual(
            equation=eq,
            reduced_observation_m=float(l),
            sigma_m=float(eq.sigma_m),
            residual_m=float(l - parametrization.design_value(eq, delta)),
        )


def weighted_rms_from_residuals(residuals: Iterable[PostfitResidual], *, use_postfit: bool) -> Optional[float]:
    """Weighted RMS helper for prefit or postfit residual streams."""
    sum_w = 0.0
    sum_w_x2 = 0.0
    for res in residuals:
        sigma = float(res.sigma_m)
        w = 1.0 / (sigma * sigma)
        x = res.residual_m if use_postfit else res.reduced_observation_m
        sum_w += w
        sum_w_x2 += w * x * x
    if sum_w == 0.0:
        return None
    return float(np.sqrt(sum_w_x2 / sum_w))


def weighted_rms_pair_from_residuals(residuals: Iterable[PostfitResidual]) -> tuple[Optional[float], Optional[float]]:
    """Return ``(prefit_wrms, postfit_wrms)`` from one residual pass."""
    sum_w = 0.0
    sum_prefit = 0.0
    sum_postfit = 0.0
    for res in residuals:
        sigma = float(res.sigma_m)
        w = 1.0 / (sigma * sigma)
        sum_w += w
        sum_prefit += w * res.reduced_observation_m * res.reduced_observation_m
        sum_postfit += w * res.residual_m * res.residual_m
    if sum_w == 0.0:
        return None, None
    return float(np.sqrt(sum_prefit / sum_w)), float(np.sqrt(sum_postfit / sum_w))


def normal_matrix_condition(normals: NormalEquations) -> Optional[float]:
    """Return the condition number of the weighted design matrix, estimated from N."""
    if normals.N.size == 0:
        return None
    eig = np.linalg.eigvalsh(np.asarray(normals.N, dtype=float))
    positive = eig[eig > 0.0]
    if positive.size == 0:
        return None
    return float(np.sqrt(positive.max() / positive.min()))


def normal_matrix_rank_diagnostics(normals: NormalEquations) -> str:
    """Return a compact diagnostic string for a singular or near-singular N."""
    N = np.asarray(normals.N, dtype=float)
    p = len(normals.parameter_names)
    if N.shape != (p, p):
        return f"normal matrix has shape {N.shape}, expected {(p, p)}."
    if p == 0:
        return "normal matrix has no parameters."
    rank = int(np.linalg.matrix_rank(N))
    diag = np.diag(N)
    zeroish = np.where(np.isclose(diag, 0.0, rtol=0.0, atol=1.0e-30))[0]
    zero_names = [str(normals.parameter_names[i]) for i in zeroish[:10]]
    condition = normal_matrix_condition(normals)
    condition_text = "unknown" if condition is None else f"{condition:.3e}"
    pieces = [
        "normal-equation matrix is singular or numerically singular",
        f"rank={rank}/{p}",
        f"condition≈{condition_text}",
        f"obs_count={normals.obs_count}",
    ]
    if zero_names:
        suffix = "..." if len(zeroish) > len(zero_names) else ""
        pieces.append(f"zero-diagonal parameters={zero_names!r}{suffix}")
    return "; ".join(pieces) + "."
