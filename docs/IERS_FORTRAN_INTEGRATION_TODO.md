# IERS official Fortran integration TODO

Status: active; Section 10 atmospheric-loading design intentionally deferred.

Last reviewed: 2026-07-25

## 1. Goal

Use the original IERS Conventions Fortran routines wherever an official,
applicable implementation exists. Keep Python responsible for model assembly,
time and unit conversion, data access, validation, and coordinate conversion.

This work has two distinct parts:

- Replace existing Python or third-party implementations with official IERS
  routines.
- Add missing models for which IERS supplies an implementation.

Atmospheric propagation delay and atmospheric loading are different effects:

- Atmospheric propagation delay changes the laser path. IERS supplies the
  optical `FCUL` routines for this calculation.
- Atmospheric tidal and non-tidal loading displace the station. IERS does not
  publish a complete, directly usable Fortran implementation for this effect.

## 2. Source and integration policy

- [x] Pin one IERS Conventions package version for the entire native library.
      Prefer the latest packaged release, IERS Conventions v1.3.0, rather than
      downloading individual live files at different dates.
- [x] Record the package version, original URL, retrieval date, and SHA-256 for
      every vendored file.
- [x] Preserve every original source header and the complete IERS software
      license.
- [x] Keep official files unchanged. Put project-specific adaptations in
      separately named wrapper routines.
- [x] Compile only one canonical copy of duplicate dependencies such as
      `FUNDARG.F`, after confirming the Chapter 5 and Chapter 8 copies match.
- [x] Document every native entry point with its time scale, coordinate frame,
      input units, output units, and sign convention.
- [x] Do not silently fall back to the old implementation after the production
      backend has been changed. Missing native support must fail explicitly.

Official package index:

- <https://iers-conventions.obspm.fr/conventions_versions.php>

## 3. Functional inventory

| Function | Current implementation | Official IERS implementation | Action | Complexity |
|---|---|---|---|---|
| Ocean-tide high-frequency EOP | NumPy implementation | `ORTHO_EOP`, `CNMTX` | Replace | Low |
| Polar-motion libration | NumPy implementation | `PMSDNUT2`, `FUNDARG` | Replace | Low |
| UT1 libration | NumPy implementation | `UTLIBR`, `FUNDARG` | Replace and expose `delta LOD` | Low |
| Solid-Earth tide | Direct `DEHANTTIDEINEL` call | `DEHANTTIDEINEL` package | Complete | Medium |
| Optical atmospheric delay | Python transcription | `FCUL_ZD_HPA`, `FCUL_A` | Replace | Low |
| Ocean tidal loading | Not implemented | `HARDISP` package | Add | High |
| Solid-Earth pole tide | Python implementation | No complete official routine | Keep and validate | Low |
| Ocean pole-tide loading | Python grid interpolation | Official grid and test data only | Keep and validate | Medium |
| Atmospheric loading | Not implemented | No complete official routine | Separate model decision | High |
| ITRS/GCRS matrices | ERFA | SOFA/ERFA reference implementation | Keep | None |

Relevant current modules:

- `llrops/classes/frames/high_frequency_eop.py`
- `llrops/classes/frames/terrestrial.py`
- `llrops/classes/delays/troposphere.py`
- `llrops/classes/displacement/solid_earth_tide.py`
- `llrops/classes/displacement/pole_tide.py`
- `llrops/classes/displacement/ocean_pole_tide.py`
- `llrops/classes/observation_factory.py`

## 4. Phase 0: native build foundation

- [x] Add one private native extension, provisionally named
      `llrops._iers2010`, instead of creating a separate extension for every
      physical model.
- [x] Use `meson-python` plus `f2py` and an explicit `.pyf` signature file as
      the preferred build path. Record the decision and rejected alternatives.
- [x] Confirm support for the project's Python and NumPy versions, fixed-form
      Fortran 77, editable installs, source distributions, and Linux wheels.
- [x] Expose only the required top-level routines. Do not expose internal
      helpers as public Python API.
- [x] Add thin Python facades that validate shapes and finite values and perform
      all unit conversions in one place.
- [x] Consider batch entry points where repeated Python/native crossings are
      measurable, while retaining scalar entry points for official test cases.
      Retain HARDISP's native regular-series entry point; keep the other
      observation-path calls scalar because epochs and geometry are resolved
      independently inside the light-time iteration.
- [x] Verify that the extension can be imported and used safely by every MPI
      worker process.
- [x] Capture baseline runtime and end-to-end LLR outputs before replacing any
      implementation.

## 5. Phase 1: high-frequency EOP

### 5.1 Source routines

- [x] Vendor Chapter 8 `ORTHO_EOP.F` and `CNMTX.F`.
- [x] Vendor Chapter 5 `PMSDNUT2.F`, `UTLIBR.F`, and one `FUNDARG.F`.
- [x] Expose the official native array outputs directly; Python unpacks the
      documented array order at the facade boundary.

Official source pages:

- <https://iers-conventions.obspm.fr/chapter5.php>
- <https://iers-conventions.obspm.fr/chapter8.php>

### 5.2 Interface and combination

- [x] Resolve and document the intended MJD time scale for each routine: use
      UT1 for `ORTHO_EOP/CNMTX` tidal phase, and TT (accepted for TDB) for
      `PMSDNUT2/UTLIBR/FUNDARG`. Do not pass an unlabelled `Epoch.mjd` value
      through the model boundary.
- [x] Convert official outputs at the Python boundary:
      microarcseconds to arcseconds or radians, and microseconds to seconds.
- [x] Return the following components separately for diagnostics:
      `ocean_delta_xp`, `ocean_delta_yp`, `ocean_delta_ut1`,
      `libration_delta_xp`, `libration_delta_yp`,
      `libration_delta_ut1`, and `libration_delta_lod`.
- [x] Form the corrections applied to C04 as:

  ```text
  xp  = xp_C04  + ocean_delta_xp  + libration_delta_xp
  yp  = yp_C04  + ocean_delta_yp  + libration_delta_yp
  UT1 = UT1_C04 + ocean_delta_ut1 + libration_delta_ut1
  ```

- [x] Extend the typed correction result to retain `delta LOD`, even if the
      position-only terrestrial rotation currently does not consume it.
- [x] Keep C04 interpolation in Python. Continue interpolating `UT1-TAI`
      internally across leap seconds before reconstructing `UT1-UTC`.

### 5.3 Double-counting audit

- [x] Determine from the selected C04 product documentation whether its UT1
      and LOD fields are regularized and exactly when `RG_ZONT2` is required.
      The selected IERS 20 C04 readme describes a combined observed series with
      Vondrak smoothing from 1984, but does not state that zonal tides were
      removed. Chapter 8 defines `RG_ZONT2` as a subtraction from observed
      UT1/LOD to form a tide-free series.
- [x] Do not enable `RG_ZONT2` by default. Applying it to an already restored
      UT1 series would double count zonal-tide effects. LLROPS keeps the
      observed C04 UT1/LOD convention and records this decision in
      `IERS_NATIVE_BUILD.md`.
- [x] Do not apply `FCNNUT` on top of observed C04 `dX/dY`. C04 celestial pole
      offsets are observed residuals relative to the product's reference
      precession-nutation model and already include the physical FCN signal.

### 5.4 Verification

- [x] Reproduce the official `ORTHO_EOP` test at MJD 47100:
      `delta xp = -162.8386373279636530 microarcseconds`,
      `delta yp = 117.7907525842668974 microarcseconds`, and
      `delta UT1 = -23.39092370609808214 microseconds`.
- [x] Reproduce all official `PMSDNUT2` and `UTLIBR` test cases.
- [x] Differential-test the native and former NumPy implementations over
      historical, leap-second-adjacent, and modern epochs.
- [x] Verify the complete C04 plus high-frequency EOP plus ERFA matrix against
      an independent reference case.

## 6. Phase 2: optical atmospheric propagation delay

### 6.1 Core replacement

- [x] Import the official Chapter 9 `FCUL_ZD_HPA.F` and `FCUL_A.F` sources.
- [x] Replace the Python implementations of `fculzd_hpa()` and `fcul_a()` with
      calls to the Fortran routines without changing the public troposphere
      configuration name.
- [x] Continue evaluating uplink and downlink independently at their respective
      elevation angles.

Official Chapter 9 software index:

- <https://iers-conventions.obspm.fr/chapter9.php>

### 6.2 Input semantics

- [x] Keep relative-humidity to water-vapor-pressure conversion outside
      `FCUL_ZD_HPA`. The official routine accepts water vapor pressure in hPa,
      not relative humidity.
- [x] Select and document the saturation vapor pressure convention used to
      convert CRD temperature and relative humidity to water vapor pressure.
- [x] Distinguish the two height definitions:
      `FCUL_ZD_HPA` requires ellipsoidal height, while `FCUL_A` documents height
      above mean sea level.
- [x] Add a source for orthometric station height or explicitly document and
      quantify any temporary ellipsoidal-height approximation in `FCUL_A`.
- [x] Keep the minimum-elevation policy outside the official routine. The
      current 3-degree clamp is application policy, not part of `FCUL_A`.
- [x] Validate pressure in hPa, wavelength in micrometers, temperature in
      Kelvin, latitude convention, and finite positive atmospheric inputs.

### 6.3 Verification

- [x] Reproduce the official `FCUL_A` test value
      `3.800243667312344087` for the McDonald Observatory input case.
- [x] Reproduce the official `FCUL_ZD_HPA` test values:
      `ZTD = 1.935225924846803114 m`,
      `ZHD = 1.932992176591644462 m`, and
      `ZWD = 0.002233748255158703871 m`.
      The v1.3.0 header lists `2010.344 m` as the input height, but the
      published outputs exactly match `2003.344 m`; preserve the source and
      use the corrected test input documented in `IERS_NATIVE_BUILD.md`.
- [x] Differential-test the native and former Python formulas over station
      latitude, height, temperature, pressure, humidity, wavelength, and
      elevation ranges used by the observations.
- [x] Verify the final two-way delay in the light-time solution, not only the
      two low-level routines.

### 6.4 APG applicability decision

- [x] Evaluate official `APG.F` as a separate asymmetric-delay model.
- [x] Confirm that `APG.F` is outside the current optical LLR propagation path.
      The routine is documented in the Chapter 9 radio-techniques material and
      its test case is a Kashima VLBI station. It models empirical north/east
      atmospheric gradients from latitude, longitude, azimuth, and elevation;
      it is not part of the wavelength-dependent optical `FCUL` model.
- [x] Keep `APG` separate from the basic `FCUL` replacement so it cannot
      silently change optical results or expand the optical atmospheric input.
- [ ] If a future LLROPS radio/VLBI observation model is added, implement and
      validate `APG` there as an explicitly selected model with its own gradient
      parameters and low-elevation estimation policy.

Official source:

- <https://iers-conventions.obspm.fr/content/chapter9/software/APG.F>

## 7. Phase 3: solid-Earth tide

### 7.1 Source routines

- [x] Vendor the complete Chapter 7 `DEHANTTIDEINEL` package:
      `DEHANTTIDEINEL.F`, `CAL2JD.F`, `DAT.F`, `NORM8.F`, `SPROD.F`,
      `ST1IDIU.F`, `ST1ISEM.F`, `ST1L1.F`, `STEP2DIU.F`, `STEP2LON.F`, and
      `ZERO_VEC8.F`.
- [x] Expose the official `DEHANTTIDEINEL` entry point directly to the private
      f2py extension. Keep the other nine Chapter 7 files byte-identical; in
      the bundled SOFA `CAL2JD.F` and `DAT.F`, rename `iau_CAL2JD`/`iau_DAT`
      to the unprefixed names called by `DEHANTTIDEINEL`, including `DAT`'s
      internal `CAL2JD` call. The files carry explicit derived-work notices.

Official source directory:

- <https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel/>

### 7.2 Runtime inputs

- [x] Supply station position in geocentric ITRF metres.
- [x] Obtain geocentric Sun and Moon vectors from the configured CALCEPH
      ephemeris and transform them to ECEF/ITRF at the requested UTC epoch.
- [x] Confirm that the ephemeris-to-ECEF conversion uses the same corrected EOP
      pipeline as the observation model without introducing a displacement
      recursion.
- [x] Pass UTC calendar date and fractional hour exactly as documented by
      `DEHANTTIDEINEL`.
- [x] Put the bundled `DAT` leap-second table under an explicit update and test
      policy.
- [x] Confirm the permanent-tide convention of the station catalogue and the
      disabled Step 3 term in the official routine.

### 7.3 Replacement and verification

- [x] Reproduce the first three published `DEHANTTIDEINEL` vectors. The fourth
      (2017) header input does not produce its copied expected output; preserve
      and regression-test the source calculation while documenting the upstream
      inconsistency.
- [x] Compare the direct native result against `pysolid` using INPOP21a/CALCEPH
      Sun and Moon positions transformed through the production BCRS-to-GCRS
      and GCRS-to-ITRF path, plus fixed-seed random WGS84 stations. Across eight
      cases the displacement-vector norm differences are `0.044--0.342 mm`;
      the two 2017 cases differ by `0.195 mm` and `0.175 mm`.
- [x] Benchmark a true single-epoch call. The native model evaluates one
      `DEHANTTIDEINEL` epoch without generating a time series; the complete
      station-displacement call is about 3.7x faster than the former `pysolid`
      path in the recorded benchmark.
- [x] Add an end-to-end regression for transmit and receive station positions
      inside the iterative light-time solution. The regression verifies both
      displacement vectors are non-zero, both UTC legs are evaluated, and the
      native contribution changes the converged round-trip observable.
- [x] Remove `samplingIntervalSeconds`, the `pysolid` runtime import, and the
      obsolete `physics` optional dependency after accepting the native
      implementation.

## 8. Phase 4: ocean tidal loading

### 8.1 Native model

- [x] Vendor the complete Chapter 7 `HARDISP` package:
      derived `HARDISP_WRAP.F`, `ADMINT.F`, `ETUTC.F`, `EVAL.F`, `JULDAT.F`,
      `LEAP.F`, `MDAY.F`, `RECURS.F`, `SHELLS.F`, `SPLINE.F`, `TDFRPH.F`,
      and `TOYMD.F`.
- [x] Do not invoke the original standalone program through files or a
      subprocess per observation. `HARDISP_WRAP.F` defines the separately
      named derived Fortran routine `HARDISP_WRAP`, exported by f2py as
      `hardisp`, around the unchanged computational dependencies.
- [x] Retain a scalar production API (`n=1`) for arbitrary light-time events.
      The native `HARDISP` series interface remains available for offline,
      regularly sampled UTC grids and is tested against scalar calls. In the
      35,055-record TOTALOBS6924 input, only 242 transmit epochs (0.690%) form
      an equal-step run of three or more samples after HARDISP whole-second
      rounding; receive epochs are additionally created during iteration.
      Therefore sorting/regridding production events would add complexity with
      negligible usable coverage.

Official source directory:

- <https://iers-conventions.obspm.fr/content/chapter7/software/hardisp/>

### 8.2 BLQ data and coordinates

- [x] Add a strict BLQ parser for amplitudes and phases of the 11 principal
      ocean tides.
- [x] Require an explicit BLQ source file and canonical station identifier in
      configuration. Do not silently substitute a nearby station.
- [x] Cache parsed station coefficients for the lifetime of the configured
      displacement model.
- [x] Verify BLQ phase convention and the Up/West/South component ordering.
- [x] Add an explicit, tested conversion from HARDISP output to project ENU and
      then ITRF coordinates.
- [x] Add `iers2010OceanTidalLoading` to the station-displacement factory and
      composable station-displacement configuration.

### 8.3 Verification

- [x] Reproduce the official HARDISP Onsala program test case and supporting
      report through the native callable interface.
- [x] Test native UTC epochs on both sides of the 2016 leap second and at a
      non-zero time of day. The upstream scalar calendar interface folds the
      exact `23:59:60` label into the following midnight; retain that behavior
      in the direct native test and reject that ambiguous label in the
      production Python facade.
- [x] Test component signs with an independently generated BLQ displacement
      series before enabling the model in production scenarios.
- [x] Add missing-BLQ, duplicate-station, malformed-file, and out-of-range-date
      failure tests. The production facade rejects dates outside the pinned
      `ETUTC.F` validity interval before entering Fortran.

## 9. Models that should remain outside the native replacement

### 9.1 Solid-Earth pole tide

- [x] Keep the current Python implementation because IERS publishes the model
      equations but no equivalent complete Fortran routine in the Chapter 7
      software collection.
- [x] Validate the secular-pole calculation, `m1/m2` signs, geocentric latitude,
      and ENU-to-ITRF conversion against independent reference values in the
      station-displacement component tests.

### 9.2 Ocean pole-tide loading

- [x] Keep the current grid reader and interpolation implementation. The
      implementation remains Python because Chapter 7 supplies coefficient and
      test data, not a complete replacement routine.
- [x] Pin the official `opoleloadcoefcmcor.txt.gz` coefficient file and record
      its checksum and provenance in `docs/IERS_NATIVE_BUILD.md`.
- [x] Reproduce representative `opoleloadcmcor.test` vectors, including the
      official longitude/latitude location, longitude wrapping, and component
      signs. Boundary and interpolation behavior remains covered by the grid
      tests.

Official Chapter 7 material:

- <https://iers-conventions.obspm.fr/chapter7.php>

### 9.3 C04 interpolation and terrestrial matrices

- [x] Keep C04 parsing and leap-safe interpolation in Python.
- [x] Keep ERFA for IAU 2006/2000A `X`, `Y`, `s`, ERA, polar motion, and C2T
      matrix construction.
- [x] Continue applying C04 `dX/dY` to the ERFA modelled CIP `X/Y` in radians.
- [x] Keep the explicit matrix order `C2T = W * R * Q` and use its transpose
      only for inverse position transformations at the same epoch.

## 10. Atmospheric loading follow-up

This section is intentionally deferred and is not part of this PR.

## 11. Cross-cutting acceptance criteria

- [x] Every native routine passes the test case embedded in its official source.
- [x] Units and coordinate frames are asserted at every Python/native boundary.
- [x] Old/new differential tests cover historical dates, modern dates,
      leap-second boundaries, multiple stations, and low elevations where
      applicable. The evidence is recorded in `docs/IERS_NATIVE_BUILD.md` and
      the native, frame, displacement, and observation-pipeline test modules.
- [x] End-to-end LLR tests report the change in calculated round-trip range and
      residual, with each physical contribution available separately. The
      observation-pipeline regression checks the RTT-to-O-C sign relation and
      exposes transmit/receive station displacement fields independently.
- [x] Benchmarks include native-call cost, per-observation cost, and complete
      processing throughput. Scalar FCUL timings and the 35,055-observation
      MPI run are recorded in `docs/IERS_NATIVE_BUILD.md`.
- [x] CI builds and tests the native extension from a clean source checkout
      (`.github/workflows/ci.yml`).
- [x] Source distributions and wheels contain all required Fortran sources,
      data, license notices, and compiled artifacts as appropriate. The
      `scripts/verify_distribution.py` archive check is run by CI.
- [x] Documentation states the exact IERS version and model/data conventions
      used in every released LLROPS version.
- [x] The old implementation is removed only after official vectors,
      differential tests, end-to-end tests, and performance checks pass for
      every routine selected for native replacement. The explicitly deferred
      Section 10 atmospheric loading and future APG work remain deferred.

## 12. Recommended implementation order

1. Pin the IERS source package and establish the native build layer.
2. Replace `FCUL_ZD_HPA` and `FCUL_A` as a small end-to-end build proof.
3. Replace `ORTHO_EOP`, `PMSDNUT2`, and `UTLIBR`.
4. Complete the direct single-epoch `DEHANTTIDEINEL` replacement.
5. Add BLQ ingestion and ocean tidal loading through `HARDISP`.
6. Validate the Python pole-tide and ocean pole-tide implementations.
7. Design atmospheric tidal and non-tidal loading as a separate project.
