# Time representation and conversion

LLROPS passes one scalar time type between modules:

```python
Epoch(jd1, jd2, scale)
```

`Epoch` is immutable and always carries an explicit `TimeScale.UTC`,
`TimeScale.TT`, or `TimeScale.TDB`. Two-part Julian dates preserve precision;
modules must not pass raw floats, strings, `datetime`, or foreign time-library objects
as runtime epochs.

## Input construction

Civil-time parsing is centralized in `Epoch`:

```python
# MINI: YYYYMMDD plus UTC seconds of day
utc = Epoch.from_date_seconds("20080417", 12_345.6789012)

# CRD H4 calendar fields
utc = Epoch.from_calendar(2008, 4, 17, 3, 25, 45.5)

# Configuration/catalog text
utc = Epoch.from_isot("2008-04-17T03:25:45.5")

# Ephemeris-native TDB data
raw_tdb = Epoch.from_jd(jd1, jd2, scale=TimeScale.TDB)
```

`from_isot`, `from_datetime`, `from_calendar`, and `from_date_seconds` accept
UTC or TT civil representations. TDB must be constructed from its two-part JD
or obtained through `TimeScaleConverter`; no hidden library conversion is allowed.


## Conversion ownership

```text
UTC <---- ERFA ----> TAI <---- ERFA ----> TT <---- configured ephemeris ----> TDB
```

PyERFA is a required dependency. `Epoch` delegates UTC calendar parsing and
formatting to `dtf2d`/`d2dtf`, UTC/TAI conversion to `utctai`/`taiutc`, and
TAI/TT conversion to `taitt`/`tttai`. This preserves ERFA's quasi-JD convention
for leap-second labels and its 1960-1972 UTC drift model. LLROPS treats ERFA
`dubious year` warnings as errors rather than silently extrapolating the leap-
second table. Earth rotation and terrestrial/celestial frame transforms use
explicit IERS C04 EOP data and ERFA rotation matrices; LLROPS does not install
or mutate an Astropy process-global EOP table.

TT/TDB conversion is never delegated to a generic library. `TimeScaleConverter` reads
`Ephemeris.tdb_minus_tt_sec()` and optionally includes the topocentric
`v_E dot X / c^2` term:

```python
converter = TimeScaleConverter(ephemeris)
tt = converter.utc2tt(utc)
tdb = converter.tt2tdb(tt, station_gcrs_m=station_gcrs_m)
recovered_utc = converter.convert(tdb, TimeScale.UTC,
                                  station_gcrs_m=station_gcrs_m)
```

## Arithmetic and serialization

`Epoch.shifted(seconds)` and `seconds_until(other)` use elapsed SI seconds.
UTC arithmetic is evaluated on the TAI timeline so leap-second days are handled
correctly. TT and TDB arithmetic uses their uniform Julian-date scales.

Canonical serialization keeps all three fields:

```python
payload = epoch.to_dict()   # {"jd1": ..., "jd2": ..., "scale": "tdb"}
epoch = Epoch.from_dict(payload)
```

TDB is not directly formatted as ISOT. Human-readable output must explicitly
select UTC or TT, which forces the ephemeris conversion first:

```python
text = tdb.isot(converter, scale=TimeScale.UTC)
```

## Light-time events

`LightTimeSolution` stores exactly three event epochs:

- `transmit_epoch`
- `bounce_epoch`
- `receive_epoch`

All three are TDB `Epoch` values. Duplicate UTC copies are not retained. UTC is
computed only where Earth orientation, terrestrial station motion, civil-date
tables, or presentation requires it. Observation output keeps the input UTC
ISOT once and serializes each solved event as `jd1`, `jd2`, and `scale`.
