# v25 — GROOPS-inspired restructuring (breaking)

* Package renamed: llr_processor_refactored -> llrops, layered as
  config / base / files / classes / estimation / programs (see ARCHITECTURE.md).
* Single entry point `python -m llrops run <config.yml>` with variables,
  loops and --set overrides replaces the six run_*.py argparse CLIs
* LLRPipeline renamed LlrRangeModel (alias kept); model assembly moved to
  `llrops.classes.observation_factory.build_observation_processor`, driven by registered class
  categories: ephemerides, earthRotation, troposphere, relativity,
  stationDisplacement, reflectorDisplacement, and rangeBias.
* New GROOPS-style estimation stack: ObservationEquation with named partial
  blocks, Parametrization classes (reflectorPosition, stationRangeBias),
  structured ParameterName, NormalEquations files with name-aligned
  combination, generic LeastSquaresAdjustment, and programs
  LlrAdjustment / LlrNormalEquations / NormalsCombineSolve.
* All v24 numerical modules moved into the package structure (imports rewritten):
  light-time, tides, delays, frames, ephemeris backend, MINI/CRD IO,
  and range-bias tables. Reflector fitting is no longer a separate
  estimator; it is expressed through the generalized adjustment stack.

## Update (this revision)
* Package `llrops.files` renamed to `llrops.fileio` (clearer, avoids the
  ambiguous "files" package name); all imports/docs updated.
* MPI ported into the new architecture (`llrops/parallel/mpi.py`): generic
  master-worker runtime with a task-handler registry; `--mpi` CLI flag
  (`mpirun/srun python -m llrops run cfg.yml --mpi`); rank 0 runs the program
  sequence, workers build and cache their own range model from a broadcast
  model spec and compute chunks of already-parsed NptRecords. Wired into
  LlrResiduals, LlrAdjustment and LlrNormalEquations through the shared
  oc_rows task; per-iteration rows carry the current linearization point as
  catalogState.
  Per-program option: `mpi: {chunksize: 8}`. v24 MPI scripts remain in
  external archived v24 checkouts for cross-validation.

## Update (backend/state and station-displacement cleanup)
* Fixed catalog-key resolution by importing `re`; builtin station/reflector
  catalogs now return deep copies so mutable fit state cannot pollute module
  globals across programs or fresh `RunContext`s.
* CLI `--set name=value` now parses booleans, null/none, ints, floats, quoted
  strings, JSON/YAML lists and mappings; full-string `{var}` substitution keeps
  those native types.
* MPI `cmd_run()` now shuts workers down even when config loading or override
  parsing fails on rank 0.
* Earth station tide flags are honored through explicit tide components.
  but ocean-pole tide now fails early if enabled without a coefficient file.
* `stationDisplacement` can now be composed explicitly with
  `{type: sum, components: [SolidEarthTide, PoleTide,
  {type: OceanPoleTide, coefficientFile: ...}]}`.
  wrapper is preserved for compatibility.
* `RunContext.create_class(..., cache=True)` is now used by the range-model
  builder for CALCEPH, EOP, delay models and immutable displacement providers.
  `LlrRangeModel` has an explicit `LlrModelState`/`LlrRangeBackends` split;
  MPI workers reuse cached class/backends while constructing a fresh mutable
  model state for each task.

## Update (generalized adjustment consolidation)
* Removed the dedicated LlrReflectorFit program and estimation/reflector_fit.py.
  Reflector coordinate fitting is now LlrAdjustment with reflectorPosition and
  optional stationRangeBias parametrizations.
* LlrAdjustment and LlrNormalEquations now share a streaming normal-equation
  engine that accumulates rows directly into N, W and lPl instead of
  materializing the full design matrix.
* NormalsCombineSolve remains a fixed-linearization load/add/solve program.
