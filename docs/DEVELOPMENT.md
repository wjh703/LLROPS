# Development and IERS backend

## Build

The project uses setuptools and Cython to build the private extension
`lunarops._iers2010_core`. The stable Python facade is
`lunarops._iers2010`. No Fortran compiler or f2py step is required.

```bash
uv sync --extra build --extra test --extra mpi
uv pip install --no-build-isolation --editable .
python -m pytest
uv run -m build
python scripts/verify_distribution.py dist/*.tar.gz dist/*.whl
```

The non-isolated editable install keeps Cython, NumPy, and setuptools available
in the persistent development environment. Standard distribution builds use
the isolated requirements declared in `pyproject.toml`. `setup.py` only
describes the Cython extension; use the PEP 517 commands above instead of
running `python setup.py install` directly.

## Backend boundary

`lunarops/_iers2010.py` validates inputs and delegates UTC/TAI/TT conversion,
leap seconds, and IAU 2003 fundamental arguments to ERFA. The compiled Cython
core implements FCUL, orthotide EOP, libration corrections, DEHANT solid-Earth
tides, and HARDISP ocean-loading synthesis.

The application layer continues to own C04 interpolation, BLQ parsing,
coordinate conversion, units, and model composition. Solid-Earth pole tide,
ocean pole-tide grid interpolation, and ERFA frame matrices remain Python
implementations by design.

## Conventions that must not drift

- C04 is the observed Earth-orientation series. `RG_ZONT2` is not applied, and
  observed `dX/dY` is not augmented with `FCNNUT`.
- Native boundaries document time scale, frame, units, and signs explicitly.
- `FCUL_ZD_HPA` receives pressure and water-vapour pressure in hPa.
- Production HARDISP calls use irregular scalar light-time epochs (`n=1`).
- A regular HARDISP series cannot cross a UTC offset transition.
- Atmospheric loading is outside the current backend scope.

## Source and licensing

The algorithms and tables are derived from IERS Conventions v1.3.0. The pinned
upstream archive had size 63,646,900 bytes and SHA-256
`5f6215b74d22cf53c5f8c40804db091f5ea2cafdaa5e131b8a9ca87c0fb43ea1`.
The repository retains the complete IERS license, not the former source tree.
Derived routine names use the `lunarops_` prefix. Wheels include the `.pyx`,
`.pxi`, and license so the notice and modified source remain available.

## Verification

Tests cover published source vectors, the frozen pre-removal differential grid,
leap and validity boundaries, signs and frames, end-to-end light-time effects,
MPI imports, and distribution contents. `scripts/verify_distribution.py`
requires the Cython sources and license and rejects Fortran/f2py files.

The numerical comparison and accepted differences are recorded in
[IERS_CYTHON_MIGRATION.md](IERS_CYTHON_MIGRATION.md). A model change is ready
only after its source vector, differential, end-to-end, and packaging tests pass.
