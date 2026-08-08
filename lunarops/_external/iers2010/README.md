# IERS 2010 provenance and license

LunarOps implements selected IERS Conventions (2010) algorithms in
`lunarops/_iers2010_core.pyx`, with mechanically transcribed coefficient tables
in `lunarops/_iers2010_tables.pxi`. Calendar conversion, time scales, leap
seconds, and standard fundamental arguments are provided by ERFA through the
Python facade `lunarops/_iers2010.py`.

The former Fortran source and f2py binding tree was deleted after differential
validation. This directory now retains only:

- `LICENSE`: the complete IERS Conventions Software License;
- this provenance note.

The implementation is derived from IERS Conventions software v1.3.0. Its pinned
upstream archive had size 63,646,900 bytes and SHA-256
`5f6215b74d22cf53c5f8c40804db091f5ea2cafdaa5e131b8a9ca87c0fb43ea1`.
All modified routines have project-specific `lunarops_` names. The derived
Cython source, coefficient table, and intact license are included in source and
binary distributions.

Implementation ownership, supported epochs, validation results, and maintenance
rules are documented in `docs/IERS_CYTHON_MIGRATION.md`.
