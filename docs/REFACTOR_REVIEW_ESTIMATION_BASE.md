# Follow-up review: estimation, parametrization, base, range bias and uncertainty

> Historical review note for v28-v29. The generic `LeastSquaresAdjustment`,
> `AdjustmentOptions`, and iteration-snapshot API described below were later
> removed. Current nonlinear estimation is owned by `adjustment_solver.py`, with
> strict options, parsing, preprocessing, and reporting in the adjacent
> `adjustment_*` modules documented in `ARCHITECTURE.md`.

This note records the v28 direction after the v27 ERFA refactor.  The guiding
principle is strict, explicit modelling: no hidden regularization, no silent
parameter rescue, and domain tables should carry their own units, coverage and
lookup policy.

## estimation/

### Strict normal-equation solve

LLROPS now keeps the normal-equation layer strict.  The old optional diagonal
regularization knob has been removed from `AdjustmentOptions`,
`NormalEquations.solve()`, `solve_normal_equations()`, `LlrAdjustment`,
`NormalsCombineSolve`, and the example YAML.  Singular or numerically singular
normal matrices are errors, not silently regularized solutions.

`solve_normal_equations()` catches the low-level `numpy.linalg.LinAlgError` only
to attach a compact diagnostic message: matrix rank, approximate condition
number, observation count, and any parameters with zero diagonal normal-matrix
entries.  It then raises `NormalEquationSingularError`, a subclass of
`LinAlgError`, so callers can still handle it as a linear-algebra failure.

### Outlier policy

The current gross/post-fit rejection logic is intentionally left in
`LeastSquaresAdjustment`.  A future `RobustLeastSquaresAdjustment` can implement
Huber/Tukey-style reweighting without changing the strict normal-equation core.

### Iteration snapshots

`LeastSquaresAdjustment` now accepts an optional `iteration_callback`.  After
each iteration it emits an `AdjustmentIterationSnapshot` containing the iteration
summary, parameter names, solved update vector, final normal equations, rejected
keys and whether the active set changed.  This keeps notebook/MPI/progress
reporting outside the estimator while still making long nonlinear adjustments
observable.

## classes/parametrization/

The block-id and interval-overlap ideas are deferred.  v28 instead adds a shared
contract-test helper for parametrization blocks.  The tests check stable
parameter-name counts, design-column length, strict update-vector length, JSON
serializable state, and stable name order after setup.

`ParametrizationList` also validates structured parameter types against the
registered LLROPS parameter schema.  The initial registry covers the existing
`position.x`, `position.y`, `position.z`, and `rangeBias` types, plus reserved
future EOP/orbit/tide examples.

## llrops/base/

`base.constants` has been reduced to foundational constants only: speed of
light, its square, and seconds per day.  Model-specific constants moved next to
their owning models:

- `classes.relativistic.constants` for GM values, relativistic scale factors,
  and external-potential body lists;
- `classes.frames.constants` for WGS84 ellipsoid constants;
- `classes.displacement.constants` for lunar displacement defaults.

LLROPS now keeps time handling scalar-only at module boundaries: `Epoch` remains the authoritative physical contract.

`base.array_validation` now also validates parameter vectors and catalog coordinate
triples.  Station and reflector catalog records normalize position/velocity
arrays through the shared validation path.

## classes/range_bias and normal-point uncertainty

The INPOP21a range-bias table uses an explicit object with station-indexed lookup
instead of repeated global scans. Callers pass `RangeBiasTable` and
`TableRangeBiasModel` objects explicitly.

The range-bias table exposes coverage summaries and validation hooks. This follows
the useful part of the GROOPS style: configuration/data tables are first-class
model objects, not loose module-level lists.

Observation uncertainty is intentionally not a configurable model. Each
`NptRecord` carries source-owned two-way timing uncertainty, with the one-way
range conversion defined next to the canonical record fields.
