# LLROPS v33 EOP duplicate-MJD handling

## Fixed

- Added explicit `earthRotation.duplicateMjdPolicy` for IERS C04/EOP files with
  repeated MJD rows.  The default policy remains `error`.
- Updated sample YAML files to use `duplicateMjdPolicy: last` for concatenated
  `eopc04.1962-now.txt` style files.
- Improved the duplicate-MJD error message so it reports the duplicated MJD
  values and tells the user which explicit policies are available.

## Available policies

- `error`: reject duplicate MJD values.
- `first`: keep the first row for each duplicated MJD.
- `last`: keep the last row for each duplicated MJD.
- `mean`: average xp, yp, and UT1-UTC over duplicated rows.
