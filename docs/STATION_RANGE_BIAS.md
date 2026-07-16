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

A custom declarative table can be supplied inline.  The preferred compact form is
one row per bias:

```yaml
globals:
  rangeBias:
    type: table
    biases:
      - APOLLO 2020-01-01/2021-01-01 1.25
      - GRASSE 2009-11-01/2014-01-01 -0.99
```

The same table can be written with one-line mappings when a per-row source or
stable machine-readable key is useful:

```yaml
globals:
  rangeBias:
    type: table
    biases:
      - {station: APOLLO, interval: 2020-01-01/2021-01-01, biasCm: 1.25}
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
      APOLLO:
        - 2006-04-07/2010-11-01
        - {start: 2010-11-01, end: 2012-04-07}
```

This estimates one one-way range-bias parameter for each declared
`station × interval` block.  The interval list is separate from the deterministic
forward-model `rangeBias` table.


## Table schema notes

Forward range-bias tables deliberately do not accept top-level `name` or
`aliases` fields.  Station-code normalization is handled by the built-in station
catalog rules; the table itself only declares rows of physical corrections.
