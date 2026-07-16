# LLROPS v35 MPI bootstrap and shared-resource broadcast

## Startup path

- `cmd_run()` now constructs `MpiRuntime` and branches worker ranks into
  `worker_loop()` before importing the program registry, `RunContext`, or the
  config loader.
- Only rank 0 imports and registers programs.
- The entire rank-0 startup path remains inside the MPI lifecycle guard, so a
  registration or config error still sends `STOP` to workers.
- MPI headings and program headings use `flush=True`.

## Observation-spec broadcast

- Rank 0 creates one immutable observation spec per model configuration.
- The spec is broadcast once and cached on every worker by `specId`.
- Per-chunk task payloads now carry only `specId`; station/reflector catalogs,
  model configs, and other immutable spec data are no longer pickled for every
  chunk.

## EOP initialization

- Rank 0 parses and deduplicates the IERS C04 text file once.
- `mjd`, `xp`, `yp`, and `UT1-UTC` arrays are broadcast inside the observation
  spec.
- Workers construct `C04EarthOrientation` directly from the arrays without
  reopening or reparsing the text file.

## CALCEPH boundary

- CALCEPH handles are intentionally not broadcast. A `CalcephBin` instance wraps
  process-local native state and is neither pickle-safe nor valid in another
  process.
- Each worker opens one CALCEPH handle per observation spec and reuses it for the
  worker lifetime.
- If the ephemeris kernel is on a slow shared filesystem, the appropriate next
  optimization is explicit per-node staging to local SSD/tmpfs, followed by
  broadcasting the staged path—not broadcasting an open handle or one full
  kernel copy per worker.
