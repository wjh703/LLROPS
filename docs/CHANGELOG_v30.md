# v30 compact declarative tables and explicit config cleanup

- Range-bias table configs now use a compact `biases` list.  Each row can be a
  single string such as `APOLLO 2020-01-01/2021-01-01 1.25`, or a one-line
  mapping such as `{station: APOLLO, interval: 2020-01-01/2021-01-01, biasCm: 1.25}`.
- Removed the old range-bias table config keys `entries` and `segments`.
- WRMS uncertainty table configs now use `uncertainties` rows with the same
  compact style, for example `APOLLO 2020-01-01/2021-01-01 0.020 APO-test`.
- Removed hidden fallbacks for observation table models.  Observation processing
  now requires explicit `rangeBias` and `uncertaintyModel` config entries.
- Removed remaining implicit-choice wording from config examples and docs so
  model choices are expressed directly in YAML.
