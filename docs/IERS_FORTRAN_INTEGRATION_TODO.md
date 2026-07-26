# IERS official Fortran integration TODO

Status: planned

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
| Solid-Earth tide | `pysolid` high-level API | `DEHANTTIDEINEL` package | Replace | Medium |
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
- [x] Evaluate `meson-python` plus `f2py` and an explicit `.pyf` signature file
      as the preferred build path. Record the decision and rejected
      alternatives.
- [x] Confirm support for the project's Python and NumPy versions, fixed-form
      Fortran 77, editable installs, source distributions, and Linux wheels.
- [x] Expose only the required top-level routines. Do not expose internal
      helpers as public Python API.
- [x] Add thin Python facades that validate shapes and finite values and perform
      all unit conversions in one place.
- [ ] Consider batch entry points where repeated Python/native crossings are
      measurable, while retaining scalar entry points for official test cases.
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

### 6.4 Optional atmospheric gradient

- [ ] Evaluate adding official `APG.F` as a separate asymmetric-delay model.
- [ ] Extend the atmospheric input with station longitude and line-of-sight
      azimuth before enabling `APG`.
- [ ] Keep this optional model separate from the basic `FCUL` replacement so it
      does not silently change existing results.

Official source:

- <https://iers-conventions.obspm.fr/content/chapter9/software/APG.F>

## 7. Phase 3: solid-Earth tide

### 7.1 Source routines

- [ ] Vendor the complete Chapter 7 `DEHANTTIDEINEL` package:
      `DEHANTTIDEINEL.F`, `CAL2JD.F`, `DAT.F`, `NORM8.F`, `SPROD.F`,
      `ST1IDIU.F`, `ST1ISEM.F`, `ST1L1.F`, `STEP2DIU.F`, `STEP2LON.F`, and
      `ZERO_VEC8.F`.
- [ ] Keep the official source unchanged and expose one project wrapper for a
      single station and epoch.

Official source directory:

- <https://iers-conventions.obspm.fr/content/chapter7/software/dehanttideinel/>

### 7.2 Runtime inputs

- [ ] Supply station position in geocentric ITRF metres.
- [ ] Obtain geocentric Sun and Moon vectors from the configured CALCEPH
      ephemeris and transform them to ECEF/ITRF at the requested UTC epoch.
- [ ] Confirm that the ephemeris-to-ECEF conversion uses the same corrected EOP
      pipeline as the observation model without introducing a displacement
      recursion.
- [ ] Pass UTC calendar date and fractional hour exactly as documented by
      `DEHANTTIDEINEL`.
- [ ] Put the bundled `DAT` leap-second table under an explicit update and test
      policy.
- [ ] Confirm the permanent-tide convention of the station catalogue and the
      disabled Step 3 term in the official routine.

### 7.3 Replacement and verification

- [ ] Reproduce every official `DEHANTTIDEINEL` vector test case.
- [ ] Compare the direct native result against the existing `pysolid` result at
      representative stations and epochs, explaining expected ephemeris or
      epoch-sampling differences.
- [ ] Benchmark a true single-epoch call. It must not generate or allocate a
      full-day, one-minute time series.
- [ ] Add an end-to-end regression for transmit and receive station positions
      inside the iterative light-time solution.
- [ ] Remove `samplingIntervalSeconds`, the `pysolid` runtime import, and the
      `physics` optional dependency after the native implementation is accepted.

## 8. Phase 4: ocean tidal loading

### 8.1 Native model

- [ ] Vendor the complete Chapter 7 `HARDISP` package:
      `HARDISP.F`, `ADMINT.F`, `ETUTC.F`, `EVAL.F`, `JULDAT.F`, `LEAP.F`,
      `MDAY.F`, `RECURS.F`, `SHELLS.F`, `SPLINE.F`, `TDFRPH.F`, and `TOYMD.F`.
- [ ] Do not invoke the original standalone program through files or a
      subprocess per observation. Add a separately named callable wrapper
      around its computational routines.
- [ ] Decide whether the production API evaluates one epoch or a sorted batch
      of epochs. Prefer batch evaluation when processing many normal points for
      the same station.

Official source directory:

- <https://iers-conventions.obspm.fr/content/chapter7/software/hardisp/>

### 8.2 BLQ data and coordinates

- [ ] Add a strict BLQ parser for amplitudes and phases of the 11 principal
      ocean tides.
- [ ] Require an explicit BLQ source file and canonical station identifier in
      configuration. Do not silently substitute a nearby station.
- [ ] Cache parsed station coefficients and any admittance expansion reused
      across epochs.
- [ ] Verify BLQ phase convention and the Up/West/South component ordering.
- [ ] Add an explicit, tested conversion from HARDISP output to project ENU and
      then ITRF coordinates.
- [ ] Add `iers2010OceanTidalLoading` to the station-displacement factory and
      composable station-displacement configuration.

### 8.3 Verification

- [ ] Reproduce the official HARDISP program test case and supporting report.
- [ ] Test epochs on both sides of leap seconds and at non-zero times of day.
- [ ] Test component signs with an independently generated BLQ displacement
      series before enabling the model in production scenarios.
- [ ] Add missing-BLQ, duplicate-station, malformed-file, and out-of-range-date
      failure tests.

## 9. Models that should remain outside the native replacement

### 9.1 Solid-Earth pole tide

- [ ] Keep the current Python implementation because IERS publishes the model
      equations but no equivalent complete Fortran routine in the Chapter 7
      software collection.
- [ ] Validate the secular-pole calculation, `m1/m2` signs, geocentric latitude,
      and ENU-to-ITRF conversion against independent reference values.

### 9.2 Ocean pole-tide loading

- [ ] Keep the current grid reader and interpolation implementation.
- [ ] Pin the official `opoleloadcoefcmcor.txt.gz` coefficient file and record
      its checksum and provenance.
- [ ] Reproduce the official `opoleloadcmcor.test` results, including longitude
      wrapping, latitude boundaries, interpolation, and component signs.

Official Chapter 7 material:

- <https://iers-conventions.obspm.fr/chapter7.php>

### 9.3 C04 interpolation and terrestrial matrices

- [ ] Keep C04 parsing and leap-safe interpolation in Python.
- [ ] Keep ERFA for IAU 2006/2000A `X`, `Y`, `s`, ERA, polar motion, and C2T
      matrix construction.
- [ ] Continue applying C04 `dX/dY` to the ERFA modelled CIP `X/Y` in radians.
- [ ] Keep the explicit matrix order `C2T = W * R * Q` and use its transpose
      only for inverse position transformations at the same epoch.

## 10. Atmospheric loading follow-up

- [ ] Treat atmospheric propagation delay, atmospheric tidal loading, and
      non-tidal atmospheric loading as three separately configurable models.
- [ ] Do not use Chapter 9 `FCUL` propagation routines for station loading.
- [ ] Define the required atmospheric pressure product, spatial and temporal
      coverage, reference-frame convention, ocean response treatment, and Green
      functions before selecting an implementation.
- [ ] Decide whether periodic S1/S2 atmospheric tides and non-tidal pressure
      loading will use gridded displacements, station time series, or an
      in-process convolution model.
- [ ] Add provenance, caching, interpolation, missing-data, and reproducibility
      requirements for the selected loading product.

This phase cannot currently be classified as an official-IERS-Fortran
replacement because the IERS Chapter 7 software distribution does not provide a
complete routine for it.

## 11. Cross-cutting acceptance criteria

- [ ] Every native routine passes the test case embedded in its official source.
- [ ] Units and coordinate frames are asserted at every Python/native boundary.
- [ ] Old/new differential tests cover historical dates, modern dates,
      leap-second boundaries, multiple stations, and low elevations where
      applicable.
- [ ] End-to-end LLR tests report the change in calculated round-trip range and
      residual, with each physical contribution available separately.
- [ ] Benchmarks include native-call cost, per-observation cost, and complete
      processing throughput.
- [ ] CI builds and tests the native extension from a clean source checkout.
- [ ] Source distributions and wheels contain all required Fortran sources,
      data, license notices, and compiled artifacts as appropriate.
- [ ] Documentation states the exact IERS version and model/data conventions
      used in every released LLROPS version.
- [ ] The old implementation is removed only after official vectors,
      differential tests, end-to-end tests, and performance checks pass.

## 12. Recommended implementation order

1. Pin the IERS source package and establish the native build layer.
2. Replace `FCUL_ZD_HPA` and `FCUL_A` as a small end-to-end build proof.
3. Replace `ORTHO_EOP`, `PMSDNUT2`, and `UTLIBR`.
4. Replace `pysolid` with direct single-epoch `DEHANTTIDEINEL` calls.
5. Add BLQ ingestion and ocean tidal loading through `HARDISP`.
6. Validate the Python pole-tide and ocean pole-tide implementations.
7. Design atmospheric tidal and non-tidal loading as a separate project.
