# Development and IERS backend

## Build

The project uses `meson-python`, NumPy f2py, and one private extension named
`llrops._iers2010`. Official IERS Conventions v1.3.0 Fortran sources are kept
unchanged under `external/iers2010/src`; project-specific signatures and
wrappers live under `external/iers2010/bindings`.

The pinned archive is `iersconventions_v1_3_0.tar.gz`, retrieved from the IERS
Conventions Center. Its size is 63,646,900 bytes and its SHA-256 is
`5f6215b74d22cf53c5f8c40804db091f5ea2cafdaa5e131b8a9ca87c0fb43ea1`.
Do not mix live chapter files with this packaged source set. The vendored
source inventory and call graph are documented in `external/iers2010/README.md`.

```bash
uv sync --extra build --extra test --extra mpi
uv pip install --no-build-isolation --editable .
python -m pytest
uv run -m build
```

The non-isolated editable install is intentional: it keeps Meson/Ninja paths in
the persistent development environment. Standard distribution builds use
isolation.

## Native scope

The extension exposes selected routines for optical atmospheric delay,
high-frequency Earth orientation, solid-Earth tide, and ocean tidal loading:
`FCUL_A`, `FCUL_ZD_HPA`, `ORTHO_EOP`, `PMSDNUT2`, `UTLIBR`, `FUNDARG`,
`DEHANTTIDEINEL`, and `HARDISP`.

The Python facade owns validation, unit conversion, C04 interpolation, BLQ
parsing, coordinate conversion, and model composition. Solid-Earth pole tide,
ocean pole-tide grid interpolation, C04 interpolation, and ERFA frame matrices
remain Python implementations by design because no complete official replacement
routine is available or because ERFA is the reference implementation.

## Conventions that must not drift

- C04 is treated as the observed Earth-orientation series. `RG_ZONT2` is not
  applied, and observed `dX/dY` is not augmented with `FCNNUT`.
- Native boundaries document time scale, frame, units, and signs explicitly.
- `FCUL_ZD_HPA` receives pressure and water-vapour pressure in hPa; official
  micro-units are converted to SI at the facade.
- `HARDISP` production calls use irregular scalar light-time epochs (`N=1`);
  its regular-series API remains available for genuine regular grids.
- Atmospheric loading is intentionally outside the current native scope.

## Verification

Tests cover official source vectors, installed-source hashes, leap-second and
date-validity boundaries, native/Python differential values, signs and frames,
end-to-end light-time effects, MPI worker imports, and source/wheel contents.
Detailed benchmark logs belong in release notes or CI artifacts rather than the
user-facing reference.

When adding a routine, pin its source and checksum, preserve the upstream
header/license, add a thin validated facade, add official and differential
tests, and update this scope list. A native replacement is production-ready
only after official vectors, end-to-end regressions, and performance checks
pass.
