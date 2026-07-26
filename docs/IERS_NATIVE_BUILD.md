# IERS native build and source policy

Status: build foundation implemented; production models still use their
existing Python or third-party implementations.

Last verified (UTC): 2026-07-26

## Source pin

All official routines in `llrops._iers2010` must come from the IERS
Conventions (2010) v1.3.0 packaged archive. Live files from individual chapter
pages must not be mixed into the extension, even when their contents appear to
match the package.

The archive URL, retrieval date, archive hash, archive size, selected-file
hashes, and purpose of every imported upstream file are recorded in
`external/iers2010/upstream/v1.3.0/SOURCE.toml`. `SHA256SUMS` provides a directly
executable integrity check. The original files are unchanged. Each Fortran
file contains the complete IERS Conventions Software License and is included
in both source and binary distributions.

When another routine is added:

1. Extract it from the archive identified by `archive_sha256` in `SOURCE.toml`.
2. Keep its upstream path below `external/iers2010/upstream/v1.3.0`.
3. Do not edit it, including its source header or license.
4. Add its SHA-256 and purpose to `SOURCE.toml` and `SHA256SUMS`.
5. Put project-specific signatures and adaptations in
   `external/iers2010/bindings` under a project-specific name.

## Build decision

The project uses `meson-python` as its PEP 517 backend and NumPy f2py with an
explicit `.pyf` signature. One private extension, `llrops._iers2010`, contains
all selected IERS routines.

This path was selected because it supports fixed-form Fortran 77, builds
editable installs and standard wheels/sdists through the same configuration,
and lets the signature file expose only reviewed entry points. Meson also
handles mixed C/Fortran linking without relying on the removed
`numpy.distutils` build layer.

Rejected alternatives:

- `setuptools` plus `numpy.distutils`: `numpy.distutils` was removed for
  Python 3.12 and is not a supported base for Python 3.14/NumPy 2.5.
- f2py automatic interface discovery: it would expose internal dependencies
  and make the Python API depend on upstream helper layout.
- one extension per physical model: this duplicates f2py runtime glue,
  complicates packaging, and creates unnecessary public module boundaries.
- a hand-written C or Cython ABI layer: it adds another maintained interface
  without solving a requirement that the explicit f2py signature does not
  already cover.

`fortranobject.c` and `fortranobject.h` are copied from the build environment's
NumPy installation into the Meson build directory. This avoids absolute-source
path failures when the development virtual environment is located inside the
repository.

## Native entry points

The extension is private. Production code must call a validating Python facade
before a native routine becomes an active backend. The initial build proof
exposes only these official Chapter 9 routines:

| Entry point | Inputs | Output | Time/frame/sign convention |
|---|---|---|---|
| `fcul_a` | north geodetic latitude in degrees; height above mean sea level in metres; temperature in kelvin; elevation in degrees | dimensionless mapping factor | no epoch or coordinate frame; positive path scale |
| `fculzd_hpa` | north geodetic latitude in degrees; ellipsoidal height in metres; total pressure and water-vapour pressure in hPa; wavelength in micrometres | total, hydrostatic, and non-hydrostatic zenith delays in metres | no epoch or coordinate frame; positive excess path |

The current troposphere class does not import this extension. Switching that
class, adding input validation, and centralising atmospheric unit conversion
belong to the next integration change.

## Build and verification

A development environment can be prepared and installed with:

```bash
uv sync --extra build --extra test --extra mpi --extra physics
uv pip install --no-build-isolation --editable .
python -m pytest
```

Do not omit `--no-build-isolation` for an editable install. An isolated
`meson-python` editable build records the isolated environment's temporary
Ninja path in its loader; that path no longer exists when the first import
triggers a rebuild. The `build` extra places Meson and Ninja in the persistent
development environment, and the non-isolated command records those paths.

Standard distribution builds use isolation:

```bash
python -m build --sdist --wheel
```

The verified Linux toolchain is:

| Component | Version |
|---|---|
| Python | 3.14.6 |
| NumPy/f2py | 2.5.1 |
| meson-python | 0.20.0 |
| Meson | 1.11.2 |
| Ninja | 1.13.0 |
| GNU Fortran | 9.4.0 |
| mpi4py / Open MPI | 4.1.2 / 4.0.3 |

The test suite checks both official FCUL vectors, installed-source hashes, and
imports/calls from two MPI workers. Wheel inspection additionally checks that
the extension, `.F` files, complete embedded license notices, source manifest,
and checksums are present.

## Upstream FCUL_ZD test input

The v1.3.0 `FCUL_ZD_HPA.F` header states an ellipsoidal height of
`2010.344 m`, but that input produces a total delay of
`1.9352297248760677 m`. The three published expected outputs are reproduced
exactly when the input height is `2003.344 m`:

```text
ZTD = 1.935225924846803114 m
ZHD = 1.932992176591644462 m
ZWD = 0.002233748255158703871 m
```

This is treated as an upstream test-header input discrepancy. The official
source remains unchanged; the corrected input and the reason for it are
explicit in the regression test.

## Pre-replacement baseline

Before the native extension was added, the Python-only suite collected 155
tests and passed in 5.40 seconds of pytest-reported time (8.23 seconds wall,
73,772 KiB maximum RSS) on the toolchain above. No production factory or model
imports `llrops._iers2010`, so this build-foundation change has no calculated
range or residual contribution.

The full `configs/llrops_oc_residuals.yml` MPI output baseline is recorded
below after processing the unchanged production path:

```text
Command: mpirun -n 16 python -m llrops run configs/llrops_oc_residuals.yml --mpi
MPI workers: 15
Observations: 35055
Status: ok=35055
Elapsed: 395.75 s
Maximum RSS: 199336 KiB
CSV bytes: 5774890
CSV SHA-256: 3caeaf5f177e6d6b98b88373156a4778a38b7f22e73ee25f67f3cd52ed660fec
Computed RTT sum: 88850.04897072203 s
One-way O-C sum: 1432.7743634047229 m
One-way O-C RMS: 1.1623352002192717 m
```

Input hashes for that run were:

```text
9e564f759b8e56550110462d2ceac82d44def348be8a39ffd3d27f03fc2c5b13  configs/llrops_oc_residuals.yml
7b0bd873f144dad407343893a478b9e53989e63cdb214f934d50d47e0f73e692  data/kernels/inpop21a_TDB_m100_p100_tt.dat
c1d8f9e771a2cf5f3137f67d65b0987ea845220adab6496281304bf50430b5cc  data/auxiliary/eopc04.1962-now.txt
3b1d099df46af0c4a7b5d3fd58db609e1b59579fc4329d29f1150b8cf1375e64  data/auxiliary/opoleloadcoefcmcor.txt
1b1efeaf7209d9d04fbbfed6e81c4d2a5f6013deef981c1ae83c9a7e6d488e0e  data/polac/TOTALOBS6924.DAT
```
