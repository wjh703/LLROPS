# Changelog v29

## Strict public API cleanup

- Removed compatibility shims retained in v27/v28:
  - `llrops.base.epoch.TimeScaleConverter` and `llrops.base.TimeScaleConverter` lazy shims;
  - legacy Earth-orientation global-state helpers and aliases;
  - `_first_resolvable_key` catalog alias;
  - lower/camel-case duplicate reflector catalog keys.
- `TimeScaleConverter` is now imported only from `llrops.classes.time_scale_converter`.
- ERFA Earth-orientation loading is explicit: `load_iers_c04(file)` has no
  `install_global` option.

## Declarative range-bias model

- Range-bias model selection now uses the `rangeBias` class config category.
- Built-in INPOP21 range-bias corrections are selected with:

```yaml
globals:
  rangeBias:
    type: inpop21
```

- Custom range-bias tables can be supplied as inline YAML/JSON or as external
  files via `rangeBias: {type: table, file: ...}`.
- Removed module-level function wrappers such as `range_bias_two_way_cm()`;
  callers now use `RangeBiasTable` or `TableRangeBiasModel` explicitly.

## Tests

- Added declarative YAML/mapping tests for range-bias tables.
- Updated strict import tests after removing compatibility aliases.
- Current suite: `47 passed`.
