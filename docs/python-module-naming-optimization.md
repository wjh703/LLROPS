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

## Direct Renames

- `estimation/normal_equation_engine.py` -> `estimation/linearized_least_squares.py`
- `programs/base.py` -> `programs/registry.py`
- `fileio/sample_catalogs.py` -> `fileio/builtin_catalogs.py`
- `fileio/inputs.py` -> `fileio/normal_point_inputs.py`
- `fileio/npt.py` -> `fileio/normal_points.py`
- `fileio/llrops_npt.py` -> `fileio/llrops_normal_point_file.py`
- `fileio/oc_table.py` -> `fileio/observation_result_writer.py`
- `classes/builders.py` -> `classes/observation_factory.py`
- `classes/time.py` -> `classes/time_scale_converter.py`
- `classes/frames/system.py` -> `classes/frames/reference_frame_system.py`
- `classes/displacement/lunar.py` -> `classes/displacement/lunar_solid_tide.py`
- `classes/displacement/solid_earth.py` -> `classes/displacement/solid_earth_tide.py`
- `classes/displacement/geometry.py` -> `classes/displacement/terrestrial_geometry.py`
- `classes/observation/assembly.py` -> `classes/observation/result_builder.py`
- `classes/observation/containers.py` -> `classes/observation/frozen_mapping.py`
- `classes/ephemerides/libration.py` -> `classes/ephemerides/longitude_libration.py`
- `estimation/adjustment_reporting.py` -> `estimation/adjustment_results.py`
- `estimation/vce.py` -> `estimation/helmert_vce.py`
- `base/validation.py` -> `base/array_validation.py`
- `parallel/cache.py` -> `parallel/worker_cache.py`
- `lifecycle.py` -> `resource_lifecycle.py`

## Structural Splits

- Keep `LlrAdjustment` in `programs/llr_adjustment.py`; extract
  `llr_normal_equations.py` and `normals_combine_solve.py`.
- Keep `LlrResiduals` in `programs/llr_residuals.py`; extract
  `crd_to_mini.py` and `normal_points_to_llrops.py`.
- Replace `observation/corrections.py` with
  `range_bias/models.py` and `uncertainty/models.py`.

## Test Renames and Splits

Rename or split the vague test modules for observation types, result writing,
CLI/MPI lifecycle, resource lifecycle, program registry, ephemeris/frame
components, and normal-equation solving. Other test names remain unchanged.

## Verification

1. Update imports, lazy exports, dynamic program registration, and documentation.
2. Confirm that no old module path remains in the repository.
3. Compile every Python module.
4. Run the complete pytest suite.
5. Verify serial and MPI program discovery.
