# LLROPS v34 EOP parser hardening

## Fixed

- Extended `read_iers_c04()` to parse IERS `finals.all` / `finals2000A`
  style EOP rows with `I`/`P` flags and formal-error columns.
- Added support for C04 variants with an hour column before MJD.
- Improved the no-samples error message by printing the first non-comment rows
  seen by the parser.

The duplicate-MJD policy from v33 is unchanged: default is still `error`, and
the sample configs explicitly use `duplicateMjdPolicy: last`.
