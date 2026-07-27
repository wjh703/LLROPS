# IERS native build and source policy

Status: the production optical troposphere, high-frequency EOP, and
solid-Earth tide models use the official Fortran routines.

Last verified (UTC): 2026-07-27

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
| `DEHANTTIDEINEL.F` | `bc6039a1704761881bb785ce44ce084ea82783107ff64c576e69155a4914e2cb` |
| `ADMINT.F` | `478e3f0c001c09dd4d9e2e9920033d29e76eb60b808238e416e6db9ffaeae6c8` |
| `ETUTC.F` | `a067c10de2c63269a54b6a14dce2daa49c30bfd52e66086b76492cd438414e1f` |
| `EVAL.F` | `b4056a09ec5d77674cab8eac0fa32b5d1be8f471fbe33e753e81ab97a7767add` |
| `HARDISP_WRAP.F` (derived) | `e8586808dad1355239f2919fffd5b150ee2f6d4547c26e50b299f8b138e525d5` |
| `JULDAT.F` | `d1f39f83503178711845532e1a4ba46d0824b896fa961d81c983d694d48dde73` |
| `LEAP.F` | `31d18153242823606beb9690ab0c685dd5403fe1e9bb5f214e22d21dd5e6a771` |
| `MDAY.F` | `d4eae8a9a0b866a63f22134fb57bad1c57fc1f3baf98450892ddeee4f7ee8aec` |
| `RECURS.F` | `8a3c88d69cebd130981887dc9c8c2f9e3d2af26fded3e052385c022126f9d44b` |
| `SHELLS.F` | `6c69dede79f9adfb16d38dcbd890bbbfc3e4e47b2cbacd5fabfb520deba86701` |
| `SPLINE.F` | `a13cc2b405079a2b86872341ca50d9b3f0cc219e43d1ce301b461af15df857f0` |
| `TDFRPH.F` | `24068e0cd1e2e210fab7dd8d4473bb60ff8335bfcb53a2fc12dad7e6d1cccd19` |
| `TOYMD.F` | `9b69b6a27d544215c516da60351e4ec51d03b04ce9bf71b6608ebf55080bf70a` |
| `CAL2JD.F` (derived) | `7634fafdbc761e9e97699102b0d43fdb564916d556497c98505a3205b3aef923` |
| `DAT.F` (derived) | `4692aa5b784070cab731dd05d541f819d6ae01b20ba339a96d85ced0ffb643dc` |
| `NORM8.F` | `636b6399dc6ab273a7b6104dd9341bf3c36995754aeee696511dbf19c9e909b6` |
| `SPROD.F` | `817761a92bb5416eb38322ea2d43d41cf8ea435e208a0ce229acf94311e9fa1e` |
| `ST1IDIU.F` | `d2976b8b76be8dd1d57e57a8d6b48f5764676126515bada3592753a07d3acd1e` |
| `ST1ISEM.F` | `efdf284bd977826a1f4aea4c79c5dbd0c38fc1c403a2376010048b511a11f2c6` |
| `ST1L1.F` | `b1dfd0e797a3ce950631ad7dbbf5f576bf6b58dc515688253a3d84ca059bc282` |
| `STEP2DIU.F` | `898c70d4b8d50e09e0c717c911c4117b3ad1ca4996d369c258b68d00ea3a5674` |
| `STEP2LON.F` | `f9d3bf0317222986d22e53557020bb13a6fbb90f8e3c9915137da6184d82813a` |
| `ZERO_VEC8.F` | `5ea9ab87e298d377f6dbe69c46b040906b6575a9132fab3830a4dc02c9139cef` |

The official IERS files are unchanged and retain their complete IERS
Conventions Software License. The HARDISP computational package is complete,
but the standalone `HARDISP.F` program is represented by the documented
derived `HARDISP_WRAP.F` callable routine. The two bundled SOFA
support routines retain their complete SOFA license but are derived works:
`CAL2JD.F` renames `iau_CAL2JD` to `CAL2JD`, and
`DAT.F` renames `iau_DAT` to `DAT` and its one internal call to `CAL2JD`.
These names match the calls made by the unchanged `DEHANTTIDEINEL.F`, removing
the need for a separate compatibility source. Their original v1.3.0 archive
hashes are `686af399ea3a493e6c0ca659d2d64f367280f5bb331584ded0800ae8d45964d3`
and `64a5a69c38d41f6d9b64204f353f5ce46750d59a1067ca507cb9b6e71e934462`.
All source files are included in both source and binary distributions.

When another routine is added:

1. Extract it from the pinned archive identified above.
2. Put the unchanged Fortran source directly in `external/iers2010/src`.
3. Do not edit it, including its source header or license, except for a
   documented license-compliant derived-work adaptation when an upstream
   linkage defect cannot otherwise be resolved.
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
selected official Chapter 5, Chapter 7, Chapter 8, and Chapter 9 routines:

| Entry point | Inputs | Output | Time/frame/sign convention |
|---|---|---|---|
| `fcul_a` | north geodetic latitude in degrees; height above mean sea level in metres; temperature in kelvin; elevation in degrees | dimensionless mapping factor | no epoch or coordinate frame; positive path scale |
| `fculzd_hpa` | north geodetic latitude in degrees; ellipsoidal height in metres; total pressure and water-vapour pressure in hPa; wavelength in micrometres | total, hydrostatic, and non-hydrostatic zenith delays in metres | no epoch or coordinate frame; positive excess path |
| `ortho_eop` | explicit UTC `Epoch` plus `UT1-UTC`; Fortran receives UT1 MJD (`TIME`) | NumPy array `[delta_xp, delta_yp, delta_ut1]`, in microarcseconds/microseconds | ocean-tide phase follows UT1/GMST |
| `pmsdnut2` | TT `Epoch` (TT is the accepted approximation to TDB); Fortran receives MJD (`RMJD`) | NumPy array `[delta_xp, delta_yp]`, in microarcseconds | official source converts MJD to TDB centuries for FUNDARG |
| `utlibr` | TT `Epoch` (TT is the accepted approximation to TDB); Fortran receives MJD (`RMJD`) | scalar `delta_ut1` in microseconds; scalar `delta_lod` in microseconds/day | official source converts MJD to TDB centuries for FUNDARG |
| `fundarg` | Julian centuries since J2000 (`T`) | `L`, `LP`, `F`, `D`, `OM` in radians | official FUNDARG convention |
| `dehanttideinel` | station, Sun, and Moon geocentric ITRF/ECEF vectors in metres; UTC calendar date and fractional hour | NumPy array `[dX, dY, dZ]` in metres | official Chapter 7 UTC input and permanent-tide convention |
| `hardisp` | UTC calendar date and whole seconds; output count and sample interval in seconds; BLQ amplitudes and phases with shape `(3, 11)` | three arrays `[dU, dS, dW]` in metres | BLQ order is vertical, west, south; output is radial/up, south, west |

`CNMTX.F` is compiled as the private helper used by `ORTHO_EOP.F`. The Chapter
5 and Chapter 8 `FUNDARG.F` files are byte-identical; the Chapter 5 copy is the
single canonical source. The official array outputs are unpacked at the Python
facade boundary; no project-specific Fortran adapter is required. The v1.3.0
Chapter 7 source calls `CAL2JD` and `DAT`, while its bundled SOFA support
routines define `iau_CAL2JD` and `iau_DAT`. The two SOFA files are renamed in
place, with their required derived-work notices, so the extension links the
official `DEHANTTIDEINEL` interface directly.

`HARDISP_WRAP.F` defines the project-derived callable `HARDISP_WRAP` routine
instead of the upstream standalone `PROGRAM`. The `.pyf` signature maps this
Fortran symbol directly to the Python entry point `hardisp`. It accepts already
parsed BLQ arrays, sets the official date common block, calls the unchanged
`ADMINT` and `RECURS` routines, and returns the same `dU`, `dS`, and `dW`
series without files, standard input, or standard output. Its header documents
the adaptation and retains the complete IERS license. It is not distributed
by or endorsed by the IERS Conventions Center. BLQ parsing, station lookup,
and ENU to ITRF conversion are implemented by `Iers2010OceanTidalLoading`.

### HARDISP series decision

The native `hardisp` entry point retains the upstream regular-series interface:
one UTC calendar start, `N` samples, and one fixed sample interval. A 32-sample
series at 900-second spacing was about 11.3 times faster than 32 individual native
calls in the development environment, and a regression test verifies that the
series agrees with scalar evaluations to 5 nanometres.

LLROPS deliberately uses `N=1` in its production station-displacement API.
LLR transmit epochs are irregular, and receive epochs are created and revised
inside the iterative light-time solve. In `TOTALOBS6924.DAT`, only 242 of
35,055 transmit epochs (0.690%) belong to equal-step sequences of at least
three points after HARDISP whole-second rounding. Sorting or regridding the
event stream would therefore complicate the solver and MPI task scheduling for
negligible usable batching coverage. The native series entry remains available
for future offline products that provide a genuine regular UTC grid.

### HARDISP UTC leap seconds

The upstream standalone program explicitly describes its calendar input as
UTC. LLROPS therefore passes ordinary UTC calendar fields directly; it does
not substitute TT or UT1. Its `TDFRPH` dependency forms `DAYFR` using a fixed
86400-second day, however, so scalar input `2016-12-31 23:59:60` is folded into
the same phase as `2017-01-01 00:00:00`. The direct native binding preserves
that upstream behavior for verification. The production
`Iers2010OceanTidalLoading` facade rejects an exact leap-second label instead
of silently returning a duplicate displacement. Native regressions cover both
sides of the 2016 boundary and the official non-zero-time test epoch.

### HARDISP date validity

The official `ETUTC.F` header states that its leap-second model is valid from
1700.0 through the current last leap-second epoch, 2017.0 in this pinned source.
Because the routine otherwise returns extrapolated or stale offsets outside that
interval, `Iers2010OceanTidalLoading` rejects UTC epochs before
`1700-01-01T00:00:00` or after `2017-01-01T00:00:00` before calling Fortran.

### HARDISP component-sign validation

The production Up/South/West, ENU, and ITRF component signs are also checked
against an independently generated displacement series. Orekit 13.1.7 with
`orekit-data` revision `315cce51` parsed the APOLLO coefficients from the
FES2022b BLQ file, evaluated its independent Java `OceanLoading` implementation
with IERS 2010 UT1 arguments, and projected the result into both the local and
terrestrial frames. Four UTC epochs spanning sign changes in all three local
components agree with LLROPS to 5 micrometres. The fixed reference values and
BLQ SHA-256 provenance are in `tests/test_ocean_tidal_loading.py`; Orekit is not
a build or runtime dependency.

### Ocean pole-tide provenance and validation

Ocean pole-tide loading remains the Python grid/interpolation model. The pinned
IERS Chapter 7 coefficient archive is `opoleloadcoefcmcor.txt.gz`, SHA-256
`3f256265439ae9c9081107950b4be76ff649d71c1f946eddedba55728ae60234`; its
expanded deployed text file has SHA-256
`3b1d099df46af0c4a7b5d3fd58db609e1b59579fc4329d29f1150b8cf1375e64`. The
official `opoleloadcmcor.test` file has SHA-256
`65a65ae32c9e1a8a88200cd009ac3b52ff82058d879e5b79917394a9537a36b2`.
Representative official vectors are tested at the published latitude and
longitude, including negative-to-0..360 longitude wrapping, ENU component
signs, and ITRF projection. The grid tests also cover axis boundaries and
bilinear interpolation. The official test uses `G = 6.673e-11`, while the
production model retains its documented `G = 6.67428e-11`; therefore the
representative vector assertions use a `3e-7 m` absolute tolerance rather than
changing the physical constant.

### Solid-Earth pole-tide validation

The solid-Earth pole tide remains the Python model because Chapter 7 publishes
the equations but no complete replacement routine. Its independent regression
uses a spherical station at 30 degrees geocentric latitude and -75 degrees
longitude on 2020-01-01 UTC. It asserts the 2018 secular-pole values, the
`m1 = xp - xp0` and `m2 = -(yp - yp0)` signs, the ENU displacement, and the
ENU-to-ITRF vector. This validates the geocentric latitude and frame rotation
without treating the model as an official Fortran integration.

There is no Python transcription or fallback for these selected routines. The
Python facades retain only application-level work: requiring an explicit UTC
observation epoch, applying C04 `UT1-UTC` to the `ortho_eop` input, passing
UTC whole seconds to `hardisp`, converting UTC to TT for the libration
routines, and converting relative humidity to water-vapour pressure,
converting radians to degrees, applying the minimum elevation policy, and
converting official micro-units to SI units before calling or returning Fortran
results.

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

The C04 and terrestrial-frame acceptance path is covered by
`tests/test_earth_orientation_iers2010.py`: C04 layouts retain `dX/dY`, UT1
interpolation crosses the 2016 leap boundary without a one-second jump, the
ERFA matrix includes the observed CIP offsets and native high-frequency EOP,
and the resulting matrix is orthogonal. Inverse position conversion uses the
transpose of the same `W * R * Q` matrix at the same epoch.

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

## APG scope decision

The official `APG.F` routine is not part of the production optical LLR model.
The Chapter 9 index places it in the tropospheric material for radio
techniques. Its interface accepts station latitude and longitude plus line-of-
sight azimuth and elevation, and returns an empirical asymmetric delay and
north/east gradients. It does not accept optical wavelength, pressure, or water
vapour pressure, which are required by the optical `FCUL_ZD_HPA` and `FCUL_A`
models. The supplied test case is the Kashima 11 VLBI station, and the source
references the Chen-Herring radio space-geodetic gradient model.

LLROPS therefore keeps the optical atmospheric input and residual path limited
to the wavelength-dependent zenith delay and `FCUL_A` mapping. No APG source,
Fortran binding, azimuth/longitude fields, or implicit gradient correction is
added. If a radio/VLBI observation path is introduced later, APG should be
implemented as a separately selected model with its own gradient estimation and
low-elevation policy.

References:

- <https://iers-conventions.obspm.fr/chapter9.php>
- <https://iers-conventions.obspm.fr/content/chapter9/software/APG.F>
- <https://doi.org/10.1029/97JB01739>

## Build and verification

A development environment can be prepared and installed with:

```bash
uv sync --extra build --extra test --extra mpi
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
`scripts/verify_distribution.py` performs the same payload check for both
sdists and wheels in clean-checkout CI.

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

## Upstream DEHANTTIDEINEL 2017 test input

The v1.3.0 `DEHANTTIDEINEL.F` header's fourth test case, dated 2017-01-15,
lists a Sun vector whose length is only `14,474,543,400.36 m`. That is about
one tenth of the expected geocentric Sun distance and produces the following
result when passed unchanged to the official source:

```text
DXTIDE = [-18.217357581922339, -23.505348376537949, 12.097611382175685] m
```

The published expected output instead repeats the preceding 2015 test case's
centimetre-scale vector. The first three official test cases reproduce their
published vectors, confirming the f2py boundary and complete dependency link.
The fourth is retained as a regression of the source input and documented as
an upstream test-header discrepancy. The DEHANT tide algorithm and its listed
inputs are unchanged; the only local source adaptations are the documented
support-routine symbol renames above.

## DEHANTTIDEINEL and pysolid comparison

`pysolid 0.3.4` wraps Dennis Milbert's independent `solid.for` implementation.
The comparison must use consistent epochs and celestial geometry. For each UTC
epoch, the native call receives Sun and Moon positions computed from the
configured INPOP21a CALCEPH kernel, converted from BCRS to GCRS and then through
the production GCRS-to-ITRF transformation. `pysolid` computes its own
low-precision Sun and Moon positions for that same epoch. Its ENU displacement
is rotated into ITRF before comparison.

Eight WGS84 stations were generated with NumPy seed `20260726`: longitude was
sampled uniformly, latitude uniformly in `sin(latitude)` within 80 degrees of
the equator, and ellipsoidal height between -300 m and 4000 m. The resulting
ITRF displacement-vector differences are:

| UTC epoch | Latitude | Longitude | Height | Native minus pysolid norm |
|---|---:|---:|---:|---:|
| 2009-04-13 00:00 | 24.287029 deg | -63.589793 deg | -178.263 m | `0.219665 mm` |
| 2012-07-13 06:14 | 46.615893 deg | 62.805271 deg | 818.685 m | `0.095969 mm` |
| 2015-07-15 12:31 | 9.706040 deg | 90.442026 deg | 2439.386 m | `0.119653 mm` |
| 2017-01-15 00:00 | 22.246531 deg | 32.642419 deg | 1206.685 m | `0.195429 mm` |
| 2017-01-15 18:47 | 26.164800 deg | 45.810002 deg | 1607.483 m | `0.175227 mm` |
| 2020-06-21 05:43 | -9.137010 deg | -123.595304 deg | 2699.411 m | `0.341612 mm` |
| 2024-03-20 12:00 | 27.292041 deg | -74.777061 deg | 2976.987 m | `0.043587 mm` |
| 2025-12-31 23:59 | 4.460520 deg | 85.424868 deg | -117.222 m | `0.163799 mm` |

The vector-norm difference ranges from `0.043587 mm` to `0.341612 mm`, with a
median of `0.169513 mm`; the largest absolute ITRF component difference is
`0.271390 mm`. This is consistent with the different celestial ephemerides and
independent tide implementations. In particular, both 2017 production-geometry
cases remain below 0.2 mm. The tens-of-metres result above is only a regression
of the anomalous Sun vector copied into the official source header and is not a
production-model discrepancy.

The production DEHANT boundary is explicit: the station, Sun, and Moon are
geocentric ITRF metres; Sun and Moon use the configured ephemeris followed by
the same corrected-EOP GCRS-to-ITRF transform as the observation model; and the
date is UTC with the documented fractional-hour argument. The bundled `DAT.F`
leap-second table is pinned by the source hash above and is exercised by the
native date regressions. DEHANT's permanent-tide convention is retained and
its optional Step 3 term is not enabled.

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
