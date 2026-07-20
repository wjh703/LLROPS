# llrops — GROOPS-inspired architecture for LLR processing

This document explains the v24 → llrops restructuring: the layer model, how it
maps to GROOPS concepts, where every v24 module went, and how each of the four
planned extensions plugs in.

## 1. Why restructure

The v24 layout coupled concerns that must evolve independently:

| v24 location | Coupled concerns |
|---|---|
| `pipeline.py` (1117 lines) | model assembly, catalog resolution, per-NP forward model, reflector partials, CSV/JSON writers, options for all of the above |
| six `run_*.py` scripts | duplicated argparse wiring, duplicated model assembly, copy-pasted MPI variants |

Every planned extension (non-tidal displacement, new lunar tide models, more
estimated parameters, lunar orbit/attitude integration) would have had to be
threaded through one monolithic pipeline constructor, one oversized option object,
the estimator and all six scripts simultaneously.

## 2. Layer model (GROOPS mapping)

```
llrops/
├── config/        GROOPS "config": class registry, config loader, run context
│   ├── registry.py    polymorphic class factory: create("troposphere", {...})
│   ├── loader.py      YAML/JSON scenario files, {var} substitution, loops
│   └── context.py     shared heavyweight objects (ephemeris, EOP, catalogs)
├── base/          GROOPS "base": constants, unified Epoch/time conversion, ParameterName
├── fileio/         GROOPS "files" layer (renamed fileio to avoid the ambiguous Python package name): every on-disk format in one place
│   ├── mini.py, crd.py            source adapters: MINI→NPT and CRD→NPT
│   ├── npt.py, llrops_npt.py      canonical memory model + versioned JSONL
│   ├── catalogs.py                StationRecord/ReflectorRecord + loaders
│   ├── inputs.py                  side-effect-free input discovery and dispatch
│   ├── oc_table.py                O-C result tables
│   └── normal_equations.py        N, W, lPl, names — save/load/add/solve
├── classes/       GROOPS "classes": config-selectable model implementations
│   ├── ephemerides/   typed ephemeris interface and CALCEPH implementation
│   │   ├── base.py            BodyState, Ephemeris (TDB Epoch queries)
│   │   ├── libration.py       optional longitude-libration corrections
│   │   └── calceph.py         CalcephEphemeris
│   ├── frames/        typed Earth orientation and composable frame transforms
│   │   ├── earth_orientation.py  IERS C04 source
│   │   ├── terrestrial.py        ITRF ↔ GCRS
│   │   ├── lunar.py              PA ↔ LCRS
│   │   ├── relativistic.py       GCRS/LCRS ↔ BCRS
│   │   └── system.py             ReferenceFrameSystem facade
│   ├── delays/       troposphere + gravitational delay models
│   ├── displacement/  typed station & reflector displacement models
│   │   ├── base.py             immutable inputs, protocols, zero/composite models
│   │   ├── solid_earth.py      IERS 2010 solid-Earth tide
│   │   ├── pole_tide.py        IERS 2010 solid-Earth pole tide
│   │   ├── ocean_pole_tide.py  grid reader + IERS 2010 ocean pole-tide loading
│   │   └── lunar.py            Moon-fixed lunar solid tide
│   ├── range_bias/, uncertainty/  (declarative table models)
│   ├── observation/   typed observation workflow
│   │   ├── light_time.py   request/solver/solution for two-way propagation
│   │   ├── model.py        pure theoretical LLR observable + reflector partial
│   │   ├── resolver.py     catalog resolution boundary
│   │   ├── corrections.py range-bias and uncertainty strategies
│   │   ├── reduction.py   deterministic corrections + stochastic reduction
│   │   ├── assembly.py    typed result construction and diagnostics
│   │   ├── processor.py   dataset orchestration only
│   │   ├── results.py     immutable result + standard/full table projections
│   │   └── equations.py   immutable equation with NAMED PARTIAL BLOCKS
│   ├── parametrization/  base + reflectorPosition + stationRangeBias
│   └── builders.py    the ONLY place mapping config type names → classes
├── estimation/
│   ├── adjustment.py              generic Gauss–Newton over a ParametrizationList
│   ├── adjustment_preprocessing.py  outlier/uncertainty/initial-scale preprocessing
│   └── normal_equation_engine.py  shared streaming normal-equation core
├── programs/      GROOPS "programs": one task each, driven by config
│   ├── CrdToMini, NormalPointsToLlrops, LlrResiduals
│   ├── LlrAdjustment             (generalized iterative adjustment; reflector fit is a parametrization case)
│   ├── LlrNormalEquations        (build + store fixed-linearization normals, don't solve)
│   └── NormalsCombineSolve       (align by parameter name, add, solve once)
├── parallel/      single-process serial + MPI master-worker backend
│   ├── cache.py     worker-cache lifecycle and deduplicated resource cleanup
│   └── observation_spec.py  picklable model specs and catalog-state transfer
└── cli.py         python -m llrops run config.yml [--mpi] --set var=value
```

GROOPS concept → llrops equivalent:

| GROOPS | llrops |
|---|---|
| XML scenario file with global elements, loops | YAML/JSON run config: `variables`, `globals`, `programs`, `loop` |
| Class categories (`troposphere`, `tides`, `ephemerides`, ...) selected by type | `config/registry.py` categories; `builders.py` registrations |
| `parametrization*` classes producing named parameters | `classes/parametrization/` + `base/parameter_name.py` (`object:type:temporal:interval`) |
| Observation equations (l, A) per arc/epoch | `classes/observation/equations.py` — l, sigma, named partial blocks |
| Normal-equation files; accumulate/combine/solve programs | `fileio/normal_equations.py`; `LlrNormalEquations`, `NormalsCombineSolve` |
| Programs run in sequence sharing config globals | `programs/` + `RunContext` object cache |

## 2.1 Unified time contract

`Epoch` is the only scalar time value passed between LLROPS modules. It stores
`jd1`, `jd2`, and an explicit `TimeScale` (`utc`, `tt`, or `tdb`). File readers
construct UTC epochs from their native civil fields:

```python
Epoch.from_calendar(year, month, day, hour, minute, second)   # CRD H4
Epoch.from_date_seconds(yyyymmdd, seconds_of_day)             # MINI / CRD records
Epoch.from_isot(text)                                         # config/catalog input
```

UTC<->TT is handled entirely by ERFA. TT<->TDB is
performed only by `TimeScaleConverter` using the configured
`Ephemeris.tdb_minus_tt_sec`; no generic time library is allowed to perform that
scale conversion. TDB epochs are serialized as two-part Julian dates.
Human-readable ISOT output converts TDB to TT/UTC through the ephemeris first.

`LightTimeSolution` stores only three event epochs (`transmit`, `bounce`,
`receive`), all in TDB. UTC copies are computed only at boundaries that actually
need civil time, Earth orientation, ITRF station motion, or output formatting.
See [`TIME.md`](TIME.md) for construction, conversion, arithmetic, and serialization rules.
See [`NORMAL_POINTS.md`](NORMAL_POINTS.md) for source adapters and the canonical
LLROPS normal-point file contract.

## 2.2 Data objects and behavior classes

Use `dataclass` only for objects whose primary purpose is to carry data: typed
inputs and outputs, configuration records, table entries, value objects, and
immutable iteration snapshots. These classes may validate or normalize their
fields in `__post_init__`, but they must not own resources or coordinate a
workflow.

Use a regular `class` for services, solvers, reference-frame transforms,
physical models, correction strategies, resource owners, and zero/composite
implementations. Their constructor states dependencies and validation
explicitly; generated equality and value-style representations are not part of
their contract. If initialization grows beyond field normalization, move it to
an explicit `__init__` instead of expanding `__post_init__`.

## 3. The two key contracts

### 3.1 Observation workflow = typed stages, not row-dict plumbing

The observation layer has one-way dependency flow:

```text
NptRecord
  -> ObservationResolver -> ResolvedObservation
  -> LlrObservationModel -> LlrPrediction
  -> LlrObservationReducer -> ObservationReduction
  -> LlrObservationResultBuilder -> LlrObservationResult
  -> LlrObservationProcessor (orchestration)
  -> result.to_equation() / result.to_row(level)
```

`LightTimeSolver.solve(LightTimeRequest)` owns only the two-way propagation
problem. `LlrObservationModel` adds the theoretical observable and optional
reflector PA partial. Catalog resolution, bias correction, uncertainty selection, result assembly, progress reporting, and table projection live in separate objects.
The estimator never reconstructs equations from table dictionaries.

```python
ObservationEquation(
    observed_minus_computed_m=result.observed_minus_computed_m,
    sigma_m=result.sigma_one_way_m,
    partials={
        "reflector_position_pa": [d/dx, d/dy, d/dz],
        "station_range_bias": [1.0],
    },
    identity=result.normal_point_index,
    station_key=result.station_key,
    reflector_key=result.reflector_key,
    epoch=result.epoch,  # UTC Epoch
)
```

`LlrObservationResult` is pickle-safe for MPI transport. CSV/JSON writers
project it to `standard` or `full` rows only at the file-I/O boundary.

### 3.2 Parametrization = columns + update absorption

A `Parametrization` block declares `ParameterName`s, fills its design columns
from partial blocks, optionally reduces `l` by its current value (bias-type
parameters), and absorbs solved corrections back into model state (catalog
positions, bias tables, force-model coefficients, integrator initial
conditions). `LeastSquaresAdjustment` and the normal-equation engine are fully
generic over the block list — they never learn what a "reflector" is.

`LlrAdjustment`, `LlrNormalEquations` and `NormalsCombineSolve` share the same
normal-equation representation.  `LlrAdjustment` is the nonlinear iteration
controller: each iteration reruns the forward model, streams typed equations directly into
normal equations, solves, applies the update and checks outliers/convergence.
`LlrNormalEquations` writes one fixed-linearization normal-equation file.
`NormalsCombineSolve` only loads, aligns, adds and solves those files once.

## 4. v24 → llrops migration table

| v24 | llrops | change |
|---|---|---|
| `constants.py`, `time_scales.py` | `base/{constants,epoch}.py`, `classes/{time,relativistic/constants,frames/constants,displacement/constants}.py` | unified scalar `Epoch`, ephemeris-owned `TimeScaleConverter`, and model-owned constants |
| `mini_io.py`, `crd_convert.py`, `sample_catalogs.py` | `fileio/mini.py`, `fileio/crd.py`, `fileio/sample_catalogs.py` | moved |
| `frame_transform.py`, `iers_config.py` | `classes/frames/{earth_orientation,terrestrial,lunar,relativistic,system}.py` | split into typed data source and composable transforms |
| `ephemeris.py` | `classes/ephemerides/{base,libration,calceph}.py` | split into interface, immutable query/result objects, correction model and CALCEPH implementation |
| `iers_delay_models.py` | `classes/delays/{base,shapiro,troposphere}.py` | moved; registered as `troposphere`/`relativity` types |
| `tidal_displacement.py` | `classes/displacement/{base,solid_earth,pole_tide,ocean_pole_tide,lunar}.py` | split into typed inputs, composable interfaces, independent physics models and explicit backend injection |
| `range_bias.py`, `uncertainty_model.py` | `classes/range_bias/table.py`, `classes/uncertainty/wrms_table.py` | moved into explicit station-indexed table models |
| `light_time.py` | `classes/observation/light_time.py` | request/solver/result API; long keyword-list interface removed |
| `pipeline.py` StationRecord/ReflectorRecord/resolve | `fileio/catalogs.py` | moved + config loaders added |
| `pipeline.py` observation workflow | `classes/observation/{resolver,model,corrections,reduction,assembly,processor,results}.py` | split into typed, independently testable stages; assembled by `builders.build_observation_processor` |
| `pipeline.py` writers | `fileio/oc_table.py` | serializes typed `LlrObservationResult` objects |
| `reflector_fit.py` | removed | reflector fitting is now `LlrAdjustment` + `reflectorPosition` (+ optional `stationRangeBias`) |
| — | `estimation/adjustment.py` | NEW generic iterative estimator |
| — | `estimation/normal_equation_engine.py` | NEW shared streaming normal-equation core |
| `run_llr_np_oc.py` | program `LlrResiduals` | config-driven |
| `run_llr_reflector_fit.py` | removed | use program `LlrAdjustment` with reflector parametrization |
| six argparse CLIs | `python -m llrops run config.yml` | one entry point |

Breaking changes: package renamed (`llr_processor_refactored` → `llrops`),
CLI replaced by configs, and observation assembly is centralized in
`builders.build_observation_processor`. Physics formulas are preserved, while public APIs and dependency injection are intentionally redesigned during development.

## 5. Extension guides

### 5.1 Non-tidal station displacement (atmospheric/hydrological loading, ...)

Category already exists: `stationDisplacement`. Steps:

1. Implement `StationDisplacement.displacement_itrf_m(data)` in
   `classes/displacement/non_tidal.py`.  `data` is a frozen
   `StationDisplacementInput` containing a read-only ITRF vector and scalar UTC
   epoch.  Return one finite `np.ndarray(3)` in ITRF meters.
2. Register it in `builders.py`:
   `register_factory("stationDisplacement", "atmosphericLoading", ...)`.
3. Combine independent components with the registered `sum` model:

   ```yaml
   stationDisplacement:
     type: sum
     components:
       - type: iers2010SolidEarthTide
       - type: iers2010PoleTide
       - type: iers2010OceanPoleTide
         coefficientFile: /path/to/opoleloadcoefcmcor.txt.gz
       - type: atmosphericLoading
         file: /path/to/loading.product
   ```


The light-time layer only consumes the `ReferenceFrameSystem` facade; new displacement components do not require changes to frame or ephemeris internals.

### 5.2 New lunar tidal model

Category `reflectorDisplacement`, interface
`displacement_lcrs_m(data)`, where `data` is a frozen
`ReflectorDisplacementInput`.  Models that need ephemerides receive the typed `Ephemeris` object
in their constructor; dependencies are fixed during construction and must not
be replaced at runtime. Implement in
`classes/displacement/`, register a new type name, and select it in the config.
If the model has estimable parameters (e.g. h2/l2), also expose partials — see
5.3.

### 5.3 Estimating more parameters

Three-step recipe, all local:

1. **Partial block** — compute the one-way range derivative in
   `LlrObservationModel` (or a dedicated partial provider), attach it to
   `LlrObservationResult.partials`, and let `result.to_equation()` carry the
   named block to the estimator. Table serialization does not participate.
2. **Parametrization** — subclass `Parametrization` in
   `classes/parametrization/`: declare `ParameterName`s (use the
   `temporal`/`interval` fields for time-resolved parameters such as daily
   EOP), place the block into columns, implement `apply_update` (write into
   the catalog / Earth-orientation source / model coefficients so the next iteration
   relinearizes), decorate with `@register("parametrization", "myType")`.
3. **Config** — add `{type: myType, ...}` to the `parametrization:` list of
   `LlrAdjustment` / `LlrNormalEquations`.

Estimator, normal equations, IO and CLI are untouched. Because parameters are
*named*, per-station or per-epoch normal equations built in separate program
calls combine in `NormalsCombineSolve`.  For nonlinear
parameters, `NormalsCombineSolve` is a single fixed-linearization solve; use
`LlrAdjustment` when updates must be applied and the model relinearized.

### 5.4 Lunar orbit / attitude integration

1. New category `lunarDynamics` (or reuse `ephemerides`): implement an
   `IntegratedLunarEphemeris` implementing `Ephemeris`
   (`body_state_bcrs`, `pa2lcrs_matrix`, `tdb_minus_tt_sec`, `lb_minus_ll`)
   but backed by numerical integration of the orbit/attitude equations,
   with force-model classes (point masses, Earth/Moon harmonics, tides)
   registered under a `forces` category — the direct GROOPS analogue.
2. Add a program `LunarOrbitIntegration` that integrates state + variational
   equations over the data span and writes the trajectory + state transition
   matrices to `fileio/` (GROOPS "orbit file" pattern), so the expensive
   integration is shared by subsequent programs.
3. Estimating initial state / dynamical parameters: the variational-equation
   output is exactly a partial block (`"orbit_state"`, shape (n_dyn,)) —
   interpolate the state transition matrix at each bounce epoch t2, project
   onto the range direction, and add a `LunarOrbitStateParametrization`
   whose `apply_update` feeds corrections back to the integrator's initial
   conditions before the next Gauss–Newton iteration.

## 6. Operational notes

* `RunContext` caches class instances by config hash: `CalcephEphemeris` and
  `C04EarthOrientation` objects are opened once per run and shared by frame, delay and
  displacement models. Raw native handles are never copied into model state.
* Loops + `--set` replace the SLURM/salloc shell wrappers: submit
  `python -m llrops run cfg.yml --set station=apollo` per array task.
* Two execution modes (see `llrops/parallel/`): single-process serial, and MPI
  master-worker (`llrops/parallel/mpi.py`): `mpirun/srun python -m llrops run
  cfg.yml --mpi`. Worker ranks enter the receive loop before program/config
  modules are imported. Rank 0 alone parses inputs, registers programs and
  writes outputs. Each observation spec is broadcast once: catalogs and the
  rank-0-parsed EOP columns are cached by `specId`, while later chunk tasks carry
  only that ID and their NptRecords. After the broadcast, rank 0 explicitly asks
  every worker to construct its processor and process-local CALCEPH handle, then
  waits for all workers to report ready. Observation timing and task dispatch
  begin only after this initialization barrier. CALCEPH native handles remain
  process-local because C-library handles cannot be serialized or shared safely
  between MPI processes. Supported by
  `LlrResiduals`, `LlrAdjustment` and `LlrNormalEquations`; the current
  linearization point is snapshotted into each task as `catalogState`.
* Validation strategy: validate `LlrAdjustment` against trusted scientific
  datasets and compare convergence, residual statistics and solved parameter
  updates. Reflector fitting is no longer a separate validated v24 path.
