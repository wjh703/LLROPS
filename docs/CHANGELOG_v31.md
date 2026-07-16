# LLROPS v31 compact table schema cleanup

## Removed

- Removed top-level `aliases` from forward range-bias and WRMS uncertainty table
  configs. Station-code normalization is now an internal catalog rule only.
- Removed top-level table `name` from forward range-bias and WRMS uncertainty
  table configs and dataclasses.

## Changed

- Declarative range-bias tables now accept only `biases`, optional `source`, or
  `file` when referenced through class config.
- Declarative WRMS uncertainty tables now accept only `uncertainties`, optional
  `source`, or `file` when referenced through class config.
- Observation diagnostics use `source` or a generic table label instead of a
  table name.
