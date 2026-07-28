# Inputs and data contracts

## Normal points

The observation and estimation layers consume one canonical record type:

```text
MINI file  -> NptDataset
CRD/FRD    -> NptDataset
LLROPS JSONL -> NptDataset
```

`read_normal_points(path)` dispatches files and directories without converting
CRD through MINI. Canonical files are versioned JSON Lines (`.llnpt`,
`.llnpt.gz`, `.llrops.jsonl`, or `.llrops.jsonl.gz`) and can be produced with
`NormalPointsToLlrops`.

Required physical fields include station/reflector identity, UTC transmit
epoch, two-way light time, two-way uncertainty, pressure, temperature,
humidity, and wavelength. Canonical field names include units. Readers reject
unknown schema versions, malformed/non-finite values, invalid humidity, and
inconsistent record counts.

`NptRecord.uncertainty_two_way_s` is the only observation uncertainty input:
MINI supplies its uncertainty field, CRD supplies record 11 bin RMS, and
canonical files store it directly. Estimation uses
`0.5 * c * uncertainty_two_way_s` as the one-way range sigma.

CRD record 20 meteorology is matched to record 11 by circular seconds-of-day
distance. CRD event 2 is the ground transmit epoch; event 1 is used only to
derive the approximate transmit epoch required by `NptRecord`.

## Time

`Epoch(jd1, jd2, scale)` is immutable and is the only runtime scalar time type.
Do not pass raw floats, strings, `datetime`, or foreign time objects between
modules.

```python
Epoch.from_date_seconds("20080417", 12345.6789012)  # MINI/CRD UTC
Epoch.from_calendar(2008, 4, 17, 3, 25, 45.5)       # civil UTC/TT
Epoch.from_jd(jd1, jd2, scale=TimeScale.TDB)         # ephemeris-native
```

ERFA owns UTC/TAI/TT conversion. The configured ephemeris owns TT/TDB
conversion. Light-time solutions store transmit, bounce, and receive events in
TDB; UTC is computed only at model or output boundaries that need it.

Intervals in configuration use `[start, end_exclusive)`. `null` means no upper
bound and must not be replaced by the current date.

## Catalogs and paths

The built-in station and reflector catalogs can be selected with
`stationCatalog: builtin` and `reflectorCatalog: builtin`. Input and output
paths may contain config variables such as `{dataDir}` and are resolved by the
run context.
