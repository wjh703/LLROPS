# stationRangeBias parametrization and forward range-bias tables

LLROPS separates two concepts that are intentionally configured in different places:

1. `globals.rangeBias` is a deterministic forward-model correction applied to
   the computed two-way observable.
2. `parametrization: stationRangeBias` estimates additive one-way station-bias
   parameters in the least-squares system.

## Forward-model range bias

Built-in INPOP21 station range-bias corrections are selected explicitly in YAML:

```yaml
globals:
  rangeBias:
    type: inpop21
```

Disable deterministic station range-bias correction with:

```yaml
globals:
  rangeBias:
    type: none
```

A custom declarative table can be supplied inline. Each row has one canonical
schema:

```yaml
globals:
  rangeBias:
    type: table
    biases:
      - station: APOLLO
        start: 2020-01-01
        end: 2021-01-01
        biasCm: 1.25
      - station: GRASSE
        start: 2009-11-01
        end: 2014-01-01
        biasCm: -0.99
        source: local-calibration
```

External YAML/JSON files use the same minimal `biases` schema:

```yaml
globals:
  rangeBias:
    type: table
    file: tables/range_bias.yml
```

All range-bias values are two-way light-distance corrections in centimetres.
Interval starts are inclusive and interval ends are exclusive.

## Estimated stationRangeBias parameters

```yaml
parametrization:
  - type: stationRangeBias
    per: station
```

This estimates one constant one-way range bias per observed station.

```yaml
parametrization:
  - type: stationRangeBias
    per: station+interval
    intervals:
      - station: APOLLO
        start: 2006-04-07
        end_exclusive: 2010-11-01
      - station: APOLLO
        start: 2010-11-01
        end_exclusive: 2012-04-07
        name: apollo-era-2
```

This estimates one one-way range-bias parameter for each declared
`station × interval` block.  The interval list is separate from the deterministic
forward-model `rangeBias` table.


## Table schema notes

Forward range-bias tables accept only `type`, `source`, and `biases`; each bias
accepts only `station`, `start`, `end`, `biasCm`, and optional `source`.
`stationRangeBias` intervals accept only `station`, `start`, `end_exclusive`,
and optional `name`. Station identity normalization is centralized in
`llrops.base.station_identity`; tables and parametrizations do not define local
aliases.
