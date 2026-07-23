# Python Module Naming Optimization

## Objective

Align LLROPS Python module names with Python conventions and GROOPS domain
terminology. This migration is intentionally breaking: no compatibility
modules, import aliases, or deprecated paths will be retained.

## Naming Rules

- Use `snake_case` module names.
- Name modules after one domain concept or explicit responsibility.
- Avoid vague suffixes such as `engine`, `system`, `builders`, and `containers`.
- Keep established scientific acronyms such as CRD, EOP, IERS, MPI, VCE,
  WRMS, and CALCEPH.
- Follow the GROOPS pattern of one configurable program per source file.
- Keep test filenames aligned with the production modules they verify.

## Canonical Modules

- Estimation: `linearized_least_squares.py`, `adjustment_results.py`, and
  `helmert_vce.py`.
- Program infrastructure: `programs/registry.py`.
- Normal-point I/O: `builtin_catalogs.py`, `normal_point_inputs.py`,
  `normal_points.py`, `llrops_normal_point_file.py`, and
  `observation_result_writer.py`.
- Observation construction: `classes/observation_factory.py`,
  `classes/observation/result_builder.py`, and
  `classes/observation/frozen_mapping.py`.
- Time and frames: `classes/time_scale_converter.py` and
  `classes/frames/reference_frame_system.py`.
- Displacement: `lunar_solid_tide.py`, `solid_earth_tide.py`, and
  `terrestrial_geometry.py`.
- Ephemerides: `classes/ephemerides/longitude_libration.py`.
- Shared infrastructure: `base/array_validation.py`,
  `parallel/worker_cache.py`, and `resource_lifecycle.py`.

## Program Modules

Each configurable program has one source file:

- `llr_adjustment.py`
- `llr_normal_equations.py`
- `normals_combine_solve.py`
- `llr_residuals.py`
- `crd_to_mini.py`
- `normal_points_to_llrops.py`

## Domain Models

Range-bias strategies live in `classes/range_bias/models.py`; uncertainty
strategies live in `classes/uncertainty/models.py`. The observation package
does not re-export these domain models.

## Test Modules

Tests use production-aligned names for observation equations and results,
result writing, CLI/MPI startup, resource lifecycle, program registration,
ephemeris/reference-frame integration, and linearized least-squares solving.

## Verification

1. Update imports, lazy exports, dynamic program registration, and documentation.
2. Confirm that no old module path remains in the repository.
3. Compile every Python module.
4. Run the complete pytest suite.
5. Verify serial and MPI program discovery.
