"""Normal equations as first-class objects and files.

In GROOPS, observation-equation programs write *normal equation files*
(N, W, lPl, observation count, parameter names); separate programs
accumulate, regularize and solve them.  This decoupling is what lets you
process decades of LLR data station-by-station (or epoch-block by
epoch-block, or later: LLR together with other techniques) and combine the
results without re-running the forward model.

The convention is::

    N = A.T @ P @ A
    W = A.T @ P @ L
    P = diag(1 / sigma**2)

File format: ``<stem>.npz`` holding N, W, lPl, obs_count plus a JSON sidecar
``<stem>.parameters.json`` with the structured parameter names and metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from llrops.base.parameter_name import ParameterName, names_to_strings, strings_to_names


@dataclass
class NormalEquations:
    parameter_names: List[ParameterName]
    N: np.ndarray
    W: np.ndarray
    lPl: float = 0.0
    obs_count: int = 0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = list(self.parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Normal-equation parameter names must be ParameterName objects.")
        if len(set(names)) != len(names):
            raise ValueError("Normal-equation parameter names must be unique.")

        normal_matrix = np.asarray(self.N, dtype=float)
        right_hand_side = np.asarray(self.W, dtype=float).reshape(-1)
        parameter_count = len(names)
        if normal_matrix.shape != (parameter_count, parameter_count):
            raise ValueError(
                f"Normal matrix has shape {normal_matrix.shape}, expected "
                f"{(parameter_count, parameter_count)}."
            )
        if right_hand_side.shape != (parameter_count,):
            raise ValueError(
                f"Normal right-hand side has shape {right_hand_side.shape}, "
                f"expected {(parameter_count,)}."
            )
        if not np.all(np.isfinite(normal_matrix)) or not np.all(np.isfinite(right_hand_side)):
            raise ValueError("Normal equations contain non-finite matrix values.")
        if not np.isfinite(self.lPl) or float(self.lPl) < 0.0:
            raise ValueError("Normal-equation lPl must be finite and non-negative.")
        if isinstance(self.obs_count, bool) or int(self.obs_count) != self.obs_count or int(self.obs_count) < 0:
            raise ValueError("Normal-equation observation count must be a non-negative integer.")
        self.parameter_names = names
        self.N = normal_matrix
        self.W = right_hand_side
        self.lPl = float(self.lPl)
        self.obs_count = int(self.obs_count)
        self.meta = dict(self.meta)

    # -- construction --------------------------------------------------------
    @classmethod
    def zeros(cls, parameter_names: Sequence[ParameterName], **meta) -> "NormalEquations":
        p = len(parameter_names)
        return cls(
            parameter_names=list(parameter_names),
            N=np.zeros((p, p), dtype=float),
            W=np.zeros(p, dtype=float),
            lPl=0.0,
            obs_count=0,
            meta=dict(meta),
        )

    def accumulate_sparse_row(self, entries, l: float, sigma: Optional[float] = None, *, weight: Optional[float] = None) -> None:
        """Accumulate one row from sparse ``(column, value)`` design entries.

        ``sigma`` keeps the historical convenience API. Stochastic-model
        solvers pass ``weight`` explicitly so supplied formal uncertainties
        remain immutable while equivalent weights change between iterations.
        """
        if weight is None:
            if sigma is None:
                raise ValueError("Either sigma or weight is required.")
            sigma = float(sigma)
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"Observation sigma must be positive and finite, got {sigma!r}.")
            p_i = 1.0 / (sigma * sigma)
        else:
            if sigma is not None:
                raise ValueError("Specify either sigma or weight, not both.")
            p_i = float(weight)
            if not np.isfinite(p_i) or p_i < 0.0:
                raise ValueError(f"Observation weight must be finite and non-negative, got {p_i!r}.")
        l_i = float(l)

        coalesced: dict[int, float] = {}
        n_params = len(self.parameter_names)
        for raw_index, raw_value in entries:
            index = int(raw_index)
            if index < 0 or index >= n_params:
                raise ValueError(f"Sparse design column {index} is outside [0, {n_params}).")
            value = float(raw_value)
            if value:
                coalesced[index] = coalesced.get(index, 0.0) + value

        if coalesced:
            idx = np.fromiter(coalesced.keys(), dtype=int)
            values = np.fromiter((coalesced[i] for i in idx), dtype=float)
            self.N[np.ix_(idx, idx)] += p_i * np.outer(values, values)
            self.W[idx] += p_i * values * l_i
        self.lPl += p_i * l_i * l_i
        self.obs_count += 1

    def accumulate_row(self, a: np.ndarray, l: float, sigma: float) -> None:
        """Accumulate one weighted observation equation ``l = a x + e``."""
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.size != len(self.parameter_names):
            raise ValueError(
                f"Design row has {a.size} columns, expected {len(self.parameter_names)}."
            )
        self.accumulate_sparse_row(
            ((index, value) for index, value in enumerate(a) if float(value)),
            l,
            sigma,
        )

    def accumulate(self, A: np.ndarray, l: np.ndarray, sigma: np.ndarray) -> None:
        """Accumulate weighted observation equations ``l = A x + e``,
        ``P = diag(1/sigma^2)``.

        Kept for small tests and external callers; production programs should
        prefer :meth:`accumulate_row` through the streaming engine.
        """
        A = np.asarray(A, dtype=float)
        l = np.asarray(l, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)
        if A.ndim != 2:
            raise ValueError("Design matrix A must be two-dimensional.")
        if A.shape[0] != l.size or l.size != sigma.size:
            raise ValueError("A, l and sigma dimensions are inconsistent.")
        for a_i, l_i, sigma_i in zip(A, l, sigma):
            self.accumulate_row(a_i, float(l_i), float(sigma_i))

    # -- combination ----------------------------------------------------------
    def add(self, other: "NormalEquations") -> "NormalEquations":
        """Add another system, aligning/expanding by parameter name."""
        union: List[ParameterName] = list(self.parameter_names)
        index = {name: i for i, name in enumerate(union)}
        for name in other.parameter_names:
            if name not in index:
                index[name] = len(union)
                union.append(name)
        p = len(union)
        N = np.zeros((p, p), dtype=float)
        W = np.zeros(p, dtype=float)

        def _scatter(src: "NormalEquations") -> None:
            idx = np.array([index[name] for name in src.parameter_names], dtype=int)
            N[np.ix_(idx, idx)] += src.N
            W[idx] += src.W

        _scatter(self)
        _scatter(other)
        return NormalEquations(
            parameter_names=union,
            N=N,
            W=W,
            lPl=self.lPl + other.lPl,
            obs_count=self.obs_count + other.obs_count,
            meta={**other.meta, **self.meta},
        )

    # -- solving ---------------------------------------------------------------
    def solve(self) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Solve ``N x = W`` by Cholesky factorization of the SPD normal matrix.

        ``N = A.T @ P @ A`` and ``W = A.T @ P @ L`` where
        ``P = diag(1 / sigma**2)``.  The covariance cofactor matrix is obtained
        by solving ``N Qxx = I`` rather than by explicitly inverting ``N``.
        Singular systems deliberately raise :class:`numpy.linalg.LinAlgError`;
        LLROPS does not add implicit diagonal regularization in this layer.
        """
        N = np.asarray(self.N, dtype=float)
        W = np.asarray(self.W, dtype=float)
        if not np.allclose(N, N.T, rtol=1.0e-12, atol=1.0e-14):
            raise np.linalg.LinAlgError("Normal matrix is not symmetric.")
        symmetric = 0.5 * (N + N.T)
        lower = np.linalg.cholesky(symmetric)
        x = np.linalg.solve(lower.T, np.linalg.solve(lower, W))
        identity = np.eye(N.shape[0])
        Qxx = np.linalg.solve(lower.T, np.linalg.solve(lower, identity))
        quadratic = float(W @ x)
        vPv = self.lPl - quadratic
        tolerance = 1.0e-12 * max(abs(self.lPl), abs(quadratic), 1.0)
        if vPv < -tolerance:
            raise np.linalg.LinAlgError(
                "Normal equations yield a negative residual quadratic form: "
                f"lPl-W.T@x={vPv:.6e}."
            )
        vPv = max(vPv, 0.0)
        degrees_of_freedom = self.obs_count - len(self.parameter_names)
        sigma0 = None if degrees_of_freedom <= 0 else float(
            np.sqrt(vPv / degrees_of_freedom)
        )
        return x, Qxx, sigma0

    # -- IO ---------------------------------------------------------------------
    def save(self, stem) -> Path:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stem.with_suffix(".npz"),
            N=self.N,
            W=self.W,
            lPl=np.asarray(self.lPl),
            obs_count=np.asarray(self.obs_count),
        )
        sidecar = {
            "parameter_names": names_to_strings(self.parameter_names),
            "meta": self.meta,
        }
        stem.with_suffix(".parameters.json").write_text(
            json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
        )
        return stem.with_suffix(".npz")

    @classmethod
    def load(cls, stem) -> "NormalEquations":
        stem = Path(stem)
        if stem.suffix == ".npz":
            stem = stem.with_suffix("")
        data = np.load(stem.with_suffix(".npz"))
        sidecar = json.loads(stem.with_suffix(".parameters.json").read_text(encoding="utf-8"))
        return cls(
            parameter_names=strings_to_names(sidecar["parameter_names"]),
            N=np.asarray(data["N"], dtype=float),
            W=np.asarray(data["W"], dtype=float),
            lPl=float(data["lPl"]),
            obs_count=int(data["obs_count"]),
            meta=dict(sidecar.get("meta") or {}),
        )
