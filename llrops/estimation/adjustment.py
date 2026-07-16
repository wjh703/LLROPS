"""Generic iterated least-squares adjustment over a parametrization list.

``LeastSquaresAdjustment`` is the generalized estimator used for reflector
coordinate fitting and every other parameter block.  It knows nothing about
what is estimated; it only talks to

* a forward-model callable producing observation equations at the current
  linearization point, and
* a :class:`~llrops.classes.parametrization.base.ParametrizationList` that
  fills design columns and absorbs updates.

The estimator is an iteration controller.  The fixed-linearization least-
squares core is shared with ``LlrNormalEquations`` through
:mod:`llrops.estimation.normal_equation_engine`, so the full design matrix is
not materialized: rows are accumulated directly into normal equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional
from collections.abc import Mapping

import numpy as np

from llrops.base.parameter_name import ParameterName, names_to_strings
from llrops.classes.observation.equations import ObservationEquation
from llrops.classes.parametrization.base import ParametrizationList
from llrops.estimation.normal_equation_engine import (
    build_normal_equations_streaming,
    normal_matrix_condition,
    postfit_residuals_streaming,
    solve_normal_equations,
    weighted_rms_pair_from_residuals,
)
from llrops.fileio.normal_equations import NormalEquations

ObsKey = Hashable


@dataclass
class AdjustmentOptions:
    max_iterations: int = 20
    damping: float = 1.0
    update_tolerance_m: float = 1.0e-3      # max parameter update (per block norm)
    wrms_tolerance_m: float = 1.0e-4        # |wrms_k - wrms_{k-1}|
    prefit_gross_threshold_m: Optional[float] = 20.0
    prefit_gross_threshold_by_station_m: Optional[Dict[str, Optional[float]]] = None
    enable_outlier_rejection: bool = True
    outlier_sigma_factor: float = 3.0
    allow_outlier_reentry: bool = True
    require_converged_light_time: bool = True


@dataclass
class AdjustmentIteration:
    iteration: int
    n_used: int
    n_rejected: int
    wrms_before_m: Optional[float]
    wrms_after_m: Optional[float]
    max_update_by_block: Dict[str, float]
    condition_number: Optional[float]
    sigma0_post: Optional[float]
    solve_method: str = "solve"


@dataclass(frozen=True)
class AdjustmentIterationSnapshot:
    """Immutable progress payload emitted after each iteration.

    The snapshot is intentionally small enough for logs/notebooks/MPI status
    reporting, but it still carries the final normal equations and update vector
    for callers that want to inspect a long run without coupling UI code into
    the estimator itself.
    """

    iteration: AdjustmentIteration
    parameter_names: List[ParameterName]
    delta: np.ndarray
    normals: NormalEquations
    rejected_keys: Dict[ObsKey, int]
    active_set_changed: bool


@dataclass
class AdjustmentResult:
    converged: bool
    iterations: List[AdjustmentIteration]
    parameter_names: List[ParameterName]
    solution: Optional[np.ndarray]          # last-iteration delta
    covariance: Optional[np.ndarray]        # (A.T @ P @ A)^-1 of last iteration
    state: Dict[str, object]                # final parameter values per block
    normals: Optional[NormalEquations]      # last-iteration normal equations
    rejected_keys: Dict[ObsKey, int]
    equations: List[ObservationEquation]    # last-iteration equations (all rows)

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "parameter_names": names_to_strings(self.parameter_names),
            "state": self.state,
            "iterations": [vars(it) for it in self.iterations],
            "rejected_observations": {str(k): it for k, it in self.rejected_keys.items()},
        }


class LeastSquaresAdjustment:
    """Iterated Gauss--Newton weighted least squares with outlier screening.

    Parameters
    ----------
    equation_source
        Callable ``(iteration:int) -> List[ObservationEquation]`` evaluating
        the forward model at the current linearization point, after previous
        parametrization updates have been absorbed.
    parametrization
        The concatenated parameter blocks.
    """

    def __init__(
        self,
        *,
        equation_source: Callable[[int], List[ObservationEquation]],
        parametrization: ParametrizationList,
        options: Optional[AdjustmentOptions] = None,
        context=None,
        iteration_callback: Optional[Callable[[AdjustmentIterationSnapshot], None]] = None,
    ) -> None:
        self.equation_source = equation_source
        self.parametrization = parametrization
        self.options = options or AdjustmentOptions()
        self.context = context
        self.iteration_callback = iteration_callback

    def _prefit_gross_threshold_for(self, eq: ObservationEquation) -> Optional[float]:
        """Return the first-pass gross-rejection threshold for one equation.

        ``prefit_gross_threshold_m`` is the global default. Entries in
        ``prefit_gross_threshold_by_station_m`` override it by station key/name/code;
        a station-specific ``None`` disables prefit gross rejection for that station.
        """
        by_station = self.options.prefit_gross_threshold_by_station_m or {}

        raw_candidates = [
            eq.station_key,
            eq.metadata.get("station_catalog_key") if isinstance(eq.metadata, Mapping) else None,
            eq.metadata.get("station_name") if isinstance(eq.metadata, Mapping) else None,
            eq.metadata.get("station_code") if isinstance(eq.metadata, Mapping) else None,
        ]
        candidates: List[str] = []
        for raw in raw_candidates:
            if raw is None:
                continue
            key = str(raw)
            if key and key not in candidates:
                candidates.append(key)
            upper = key.upper()
            if upper and upper not in candidates:
                candidates.append(upper)

        for key in candidates:
            if key in by_station:
                value = by_station[key]
                return None if value is None else float(value)

        value = self.options.prefit_gross_threshold_m
        return None if value is None else float(value)

    # -------------------------------------------------------------------
    def run(self) -> AdjustmentResult:
        opt = self.options
        rejected: Dict[ObsKey, int] = {}
        iterations: List[AdjustmentIteration] = []
        converged = False
        previous_wrms: Optional[float] = None
        last = {"delta": None, "Qxx": None, "normals": None, "equations": []}
        names: List[ParameterName] = []

        for iteration in range(1, opt.max_iterations + 1):
            equations = self.equation_source(iteration)
            if opt.require_converged_light_time:
                equations = [eq for eq in equations if eq.converged]

            if iteration == 1:
                self.parametrization.setup(equations, self.context)
                names = self.parametrization.parameter_names()
                for eq in equations:
                    threshold = self._prefit_gross_threshold_for(eq)
                    if threshold is None:
                        continue
                    if abs(self.parametrization.reduced_observation(eq)) > threshold:
                        rejected[eq.identity] = iteration

            used = [eq for eq in equations if eq.identity not in rejected]
            if len(used) < len(names):
                raise RuntimeError(
                    f"Iteration {iteration}: {len(used)} observations for "
                    f"{len(names)} parameters after outlier rejection."
                )

            normals = build_normal_equations_streaming(
                used,
                self.parametrization,
                parameter_names=names,
                iteration=iteration,
            )
            solution = solve_normal_equations(normals)
            delta = opt.damping * np.asarray(solution.delta, dtype=float)
            Qxx = solution.covariance
            sigma0 = solution.sigma0_post

            wrms_before, wrms_after = weighted_rms_pair_from_residuals(
                postfit_residuals_streaming(used, self.parametrization, delta)
            )
            condition = normal_matrix_condition(normals)

            # ---- post-fit outlier screening on the same linearized system ----
            # The posterior residual of the Gauss--Newton step at x_k is
            #     v_k = l_k - a_k delta_k.
            # Outlier/re-entry decisions must use this quantity before the
            # update is absorbed into catalogs/bias-state; otherwise geometry
            # blocks and bias-like blocks would be tested at mixed
            # linearization points.
            active_set_changed = False
            if opt.enable_outlier_rejection:
                test_pool = equations if opt.allow_outlier_reentry else used
                new_rejected: Dict[ObsKey, int] = {}
                for residual in postfit_residuals_streaming(test_pool, self.parametrization, delta):
                    threshold = opt.outlier_sigma_factor * residual.sigma_m
                    if abs(residual.residual_m) > threshold:
                        new_rejected[residual.equation.identity] = iteration

                old_rejected_keys = set(rejected)
                if opt.allow_outlier_reentry:
                    active_set_changed = set(new_rejected) != old_rejected_keys
                    rejected = new_rejected
                else:
                    rejected.update(new_rejected)
                    active_set_changed = len(set(rejected) - old_rejected_keys) > 0

            max_updates = self.parametrization.apply_update(delta)

            iteration_record = AdjustmentIteration(
                iteration=iteration,
                n_used=len(used),
                n_rejected=len(rejected),
                wrms_before_m=wrms_before,
                wrms_after_m=wrms_after,
                max_update_by_block=max_updates,
                condition_number=condition,
                sigma0_post=sigma0,
                solve_method=solution.method,
            )
            iterations.append(iteration_record)
            if self.iteration_callback is not None:
                self.iteration_callback(
                    AdjustmentIterationSnapshot(
                        iteration=iteration_record,
                        parameter_names=list(names),
                        delta=np.array(delta, dtype=float, copy=True),
                        normals=normals,
                        rejected_keys=dict(rejected),
                        active_set_changed=bool(active_set_changed),
                    )
                )
            last.update(delta=delta, Qxx=Qxx, normals=normals, equations=equations)

            max_update = max(max_updates.values()) if max_updates else 0.0
            wrms_step = (
                abs(wrms_after - previous_wrms)
                if (wrms_after is not None and previous_wrms is not None)
                else None
            )
            previous_wrms = wrms_after
            if max_update <= opt.update_tolerance_m and not active_set_changed and (
                wrms_step is None or wrms_step <= opt.wrms_tolerance_m
            ):
                converged = True
                break

        return AdjustmentResult(
            converged=converged,
            iterations=iterations,
            parameter_names=names,
            solution=last["delta"],
            covariance=last["Qxx"],
            state=self.parametrization.state(),
            normals=last["normals"],
            rejected_keys=rejected,
            equations=last["equations"],
        )
