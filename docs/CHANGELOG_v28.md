# Changelog v28

## Estimation

- Removed the remaining diagonal normal-matrix regularization entry points.
  `NormalEquations.solve()` is now a strict `numpy.linalg.solve` path.
- Added `NormalEquationSingularError` with rank/condition/zero-diagonal diagnostics.
- Added optional `LeastSquaresAdjustment(iteration_callback=...)` and
  `AdjustmentIterationSnapshot` for progress/event consumers.
- Kept the existing gross/post-fit outlier policy unchanged; robust reweighting is
  left for a future `RobustLeastSquaresAdjustment`.

## Parametrization

- Added schema validation for structured `ParameterName.type` values.
- Added shared parametrization contract tests covering column counts, update length,
  JSON-serializable state, and stable parameter names.

## Base

- Reduced `base.constants` to foundational constants only.
- Moved relativistic/GM constants, WGS84 constants, and lunar displacement defaults
  to model-owned modules.
- Kept time handling on the scalar `Epoch` contract; batch facade code has been removed.
- Routed catalog vector and parameter-vector checks through `base.array_validation`.

## Range bias and uncertainty

- Added `RangeBiasTable` and `WrmsUncertaintyTable` station-indexed table objects.
- Added a first table-object layer before the v29 public API cleanup.
- Added coverage summaries and table lookup tests.
