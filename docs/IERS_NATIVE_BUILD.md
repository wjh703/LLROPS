# IERS native build and source policy

Status: the production optical troposphere and high-frequency EOP models use
the official Fortran routines.

Last verified (UTC): 2026-07-26

## Source pin

All official routines in `llrops._iers2010` come from the IERS Conventions
(2010) v1.3.0 packaged archive. Live files from individual chapter pages must
not be mixed into the extension, even when their contents appear to match the
package. The imported routines live directly in `external/iers2010/src`.

The archive was retrieved on 2026-07-26 from
`https://iers-conventions.obspm.fr/packaged_versions/iersconventions_v1_3_0.tar.gz`.
Its size is 63,646,900 bytes and its SHA-256 is
`5f6215b74d22cf53c5f8c40804db091f5ea2cafdaa5e131b8a9ca87c0fb43ea1`.
The selected source hashes are:

| File | SHA-256 |
|---|---|
| `FCUL_A.F` | `fdeb39aee3c8d4c2d6eb6a7e743c420372e28da5b3e84942d09580a88847693a` |
| `FCUL_ZD_HPA.F` | `92731affca053aad15a44be7db58dbf6df689e75cf2e1f3b39cb4d99a4da198b` |
| `ORTHO_EOP.F` | `dfd1524b583f2a0f11baf2f03282d0f5ba5731026ac1fdaff4aa6e9460995022` |
| `CNMTX.F` | `8a29c599275110990e6ce93254995d498edbccc523edb2de508455736f45fc93` |
| `PMSDNUT2.F` | `0818b58bc2a420e1eb3f951d8a74646e5fe7b5371c9beb5e89fa37c12dd0d965` |
| `UTLIBR.F` | `f523335d552ac14b661121a081ad799382312d819853c674bc0102484b5e2406` |
| `FUNDARG.F` | `18263cbb1289e222e6ee6e59d52beb343eb77a63ed3212e4f05a4c85d475ae78` |

The official files are unchanged. Each contains the complete IERS Conventions Software
License and is included in both source and binary distributions.

When another routine is added:

1. Extract it from the pinned archive identified above.
2. Put the unchanged Fortran source directly in `external/iers2010/src`.
3. Do not edit it, including its source header or license.
4. Add its SHA-256 and purpose to the source table above.
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

The extension is private. The production troposphere and terrestrial-frame
models call it through their existing interfaces. The extension exposes
selected official Chapter 5, Chapter 8, and Chapter 9 routines:

| Entry point | Inputs | Output | Time/frame/sign convention |
|---|---|---|---|
| `fcul_a` | north geodetic latitude in degrees; height above mean sea level in metres; temperature in kelvin; elevation in degrees | dimensionless mapping factor | no epoch or coordinate frame; positive path scale |
| `fculzd_hpa` | north geodetic latitude in degrees; ellipsoidal height in metres; total pressure and water-vapour pressure in hPa; wavelength in micrometres | total, hydrostatic, and non-hydrostatic zenith delays in metres | no epoch or coordinate frame; positive excess path |
| `ortho_eop` | explicit UTC `Epoch` plus `UT1-UTC`; Fortran receives UT1 MJD (`TIME`) | NumPy array `[delta_xp, delta_yp, delta_ut1]`, in microarcseconds/microseconds | ocean-tide phase follows UT1/GMST |
| `pmsdnut2` | TT `Epoch` (TT is the accepted approximation to TDB); Fortran receives MJD (`RMJD`) | NumPy array `[delta_xp, delta_yp]`, in microarcseconds | official source converts MJD to TDB centuries for FUNDARG |
| `utlibr` | TT `Epoch` (TT is the accepted approximation to TDB); Fortran receives MJD (`RMJD`) | scalar `delta_ut1` in microseconds; scalar `delta_lod` in microseconds/day | official source converts MJD to TDB centuries for FUNDARG |
| `fundarg` | Julian centuries since J2000 (`T`) | `L`, `LP`, `F`, `D`, `OM` in radians | official FUNDARG convention |

`CNMTX.F` is compiled as the private helper used by `ORTHO_EOP.F`. The Chapter
5 and Chapter 8 `FUNDARG.F` files are byte-identical; the Chapter 5 copy is the
single canonical source. The official array outputs are unpacked at the Python
facade boundary; no project-specific Fortran adapter is required.

There is no Python transcription or fallback for these selected routines. The
Python facades retain only application-level work: requiring an explicit UTC
observation epoch, applying C04 `UT1-UTC` to the ocean-tide input, converting
UTC to TT for the libration routines, and converting relative humidity
to water-vapour pressure, converting radians to degrees, applying the minimum
elevation policy, and converting official micro-units to SI units before
calling or returning Fortran results.

## C04 convention and double-counting policy

The production configuration uses the IERS 20 C04 file
`eopc04.1962-now.txt`. Its upstream readme identifies this product as a daily
combined Earth-orientation series sampled at 0h UTC. It contains observed
`UT1-UTC`, LOD, and celestial-pole offsets, among other fields. The same readme
documents Vondrak combined smoothing for UT1 and LOD from 1984 onward; this is
not a declaration that the tidal terms have been removed.

IERS Conventions (2010), Chapter 8, Section 8.1, states that `RG_ZONT2` gives
the zonal-tide corrections and that subtracting those corrections from observed
`UT1-UTC`, LOD, and rotation rate produces a tide-free series. The conventions
also recommend exchanging ordinary UT1 and LOD rather than an ambiguous
regularized variant. LLROPS therefore treats C04 as the observed series and
does not call or apply `RG_ZONT2`. Subtracting its output from observed C04
would intentionally produce a tide-free series, which is not the convention
used by the current frame transform; adding its output to C04 would double
count the zonal-tide contribution.

The C04 celestial-pole-offset columns are observed offsets relative to the
product's declared precession-nutation reference model. `FCNNUT` is a separate
free-core-nutation prediction model, so it is not added on top of those observed
`dX/dY` values. A future input explicitly documented as tide-free or model-only
must be handled by a separate, named policy rather than silently changing the
C04 path.

References:

- <https://hpiers.obspm.fr/eoppc/eop/eopc04/readme>
- <https://iers-conventions.obspm.fr/chapter8.php>, Section 8.1 and `RG_ZONT2.F`
- <https://iers-conventions.obspm.fr/chapter5.php>, `FCNNUT.F`

## Atmospheric input policy

The Python boundary accepts scalar, finite inputs and rejects latitude or
elevation outside `[-90, 90] deg`, non-positive pressure, temperature, or
wavelength, and negative water-vapour pressure. Relative humidity must be in
`[0, 100] percent`. The minimum-elevation setting must be in `[0, 90] deg` and
is applied before `FCUL_A`; it is not part of the official routine.

Relative humidity is converted outside `FCUL_ZD_HPA` with the existing
Magnus-type saturation-vapour-pressure convention:

```text
T_c = T_k - 273.15
e_s = 6.1121 * exp(17.502 * T_c / (240.97 + T_c)) hPa
WVP = RH / 100 * e_s
```

The station catalogue derives WGS84 geodetic latitude and ellipsoidal height
from ITRF coordinates. The ellipsoidal height is therefore correct for
`FCUL_ZD_HPA`. `FCUL_A`, however, documents height above mean sea level. Until
an orthometric-height or geoid source is configured, the same ellipsoidal
height is passed to `FCUL_A` as an explicit approximation.

The approximation was evaluated by changing only the `FCUL_A` height by
`+/-100 m`. At latitude `45 deg`, height `100 m`, temperature `293.15 K`, and
elevation `30 deg`, the one-way mapped-delay change is `0.0264 mm`. A sampled
grid used latitudes in 10-degree steps, heights `[-500, 0, 500, 2000, 5000] m`,
temperatures `[230, 273.15, 330] K`, elevations `[3, 5, 10, 30, 90] deg`,
pressure `1100 hPa`, water-vapour pressure `50 hPa`, and wavelength `0.3 um`.
Its largest change was `14.84 mm` at the 3-degree boundary. A future
geoid-backed orthometric height remains necessary for low-elevation precision
work.

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

The test suite checks official FCUL and high-frequency EOP vectors, installed-source hashes, and
imports/calls from two MPI workers. Wheel inspection additionally checks that
the extension, `.F` files, and complete embedded license notices are present.

The `FUNDARG` reference values are documented to 15-19 decimal places; the
native regression allows `2e-11` radians to account for the official test
input's decimal representation and compiler floating-point evaluation.

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

## Replacement comparison

Before removing the Python transcription, 100,000 deterministic samples were
compared over latitude `[-90, 90] deg`, height `[-500, 5000] m`, pressure
`[500, 1100] hPa`, water-vapour pressure `[0, 50] hPa`, wavelength
`[0.3, 1.1] um`, temperature `[230, 330] K`, and elevation `[3, 90] deg`.
`FCUL_A` was bit-identical. The maximum absolute differences were
`3.646e-10 m` for zenith total delay and `4.802e-9 m` for mapped slant delay.
The differences come from default-real literal rounding in the unchanged
official `FCUL_ZD_HPA.F` source.

On an AMD EPYC 7313, median scalar-call timings for the former Python and
Fortran implementations were respectively `2.099 us` versus `0.187 us` for
`FCUL_A`, and `2.068 us` versus `0.255 us` for `FCUL_ZD_HPA`. The complete
slant-delay calculation, including common input conversions, improved from
`5.531 us` to `1.614 us` per call.

## Pre-replacement baseline

Before the native extension was added, the Python-only suite collected 155
tests and passed in 5.40 seconds of pytest-reported time (8.23 seconds wall,
73,772 KiB maximum RSS) on the toolchain above. The output below was captured
before the production troposphere model switched to `llrops._iers2010`.

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
