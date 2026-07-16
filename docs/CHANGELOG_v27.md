# llrops v27 ERFA / vectorization refactor

## Correctness and dependency changes

- Removed hidden process-global EOP installation from terrestrial frame construction.
  `ReferenceFrameSystem` now keeps its own explicit `EarthOrientation` source and
  `TerrestrialFrameTransform` never mutates global state.
- Replaced the Astropy-dependent runtime time/frame path with ERFA + explicit IERS C04
  interpolation:
  - UTC <-> TT uses ERFA when available, with a bundled leap-second fallback.
  - TT <-> TDB remains owned by `classes.time.TimeScaleConverter` and the configured
    ephemeris target-16 table.
  - ITRS <-> GCRS uses ERFA `c2t06a` with interpolated xp, yp and UT1-UTC from the
    configured EOP file.
- Removed Astropy object export from `Epoch`. Runtime epochs are two-part JDs with an
  explicit `TimeScale`.
- CALCEPH target-16 failures are no longer silently collapsed into `None` unless the
  error looks like a missing quantity; other CALCEPH errors are raised with context.

## API and structure

- Moved `TimeScaleConverter` from `llrops.base.epoch` to `llrops.classes.time` to keep
  `base` as pure foundational value types. A lazy import shim is retained for older
  imports from `llrops.base.epoch`.
- Added `llrops.base.validation` with shared `vector3`, `readonly_vector3`, `matrix3x3`
  and `readonly_matrix3x3`; frame, ephemeris, displacement and light-time modules now
  use the same validators.
- Added batch hooks for ephemeris, time conversion, Earth orientation and frame transforms
  (`*_many`) so later arc-level vectorization can reuse the same contracts.
- `LightTimeSolution` now carries the converged station/reflector BCRS positions, so
  reflector-position partials no longer rebuild the solver geometry in the observation
  model.

## Diagnostics and output schema

- Troposphere elevation clamping is now explicit in `ObservationReduction` and output
  rows via:
  - `tropo_elevation_up_used_deg`
  - `tropo_elevation_down_used_deg`
  - `tropo_up_clamped`
  - `tropo_down_clamped`
  - `tropo_clamped`
- Output fields are defined once in `STANDARD_OUTPUT_SCHEMA`; row generation and field
  validation are derived from the same schema, removing the separate fragile field list.
- `fileio.catalogs.first_resolvable_key` is public; the private name remains as a
  compatibility alias only.

## Parametrization / estimation cleanup included in this pass

- `ParametrizationList` now caches its parameter layout after setup and fills design rows
  into preallocated arrays instead of rebuilding name lists and concatenating blocks on
  every row.
- Reflector and station range-bias parametrizations now maintain key -> column-index maps.
- `ParameterName` is a slotted frozen value object with stricter canonical parsing.

## Config updates

- The example YAML files are updated to v27 and document that ERFA + explicit IERS C04 EOP
  is now the supported time/frame path.
