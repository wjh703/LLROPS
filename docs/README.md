# LLROPS documentation

LLROPS is a configuration-driven Lunar Laser Ranging processor with a
GROOPS-inspired split between files, classes, parametrizations, programs, and
estimation.

## Start here

| Task | Document |
|---|---|
| Run a program or choose an output | [PROGRAMS.md](PROGRAMS.md) |
| Prepare MINI, CRD, or canonical LLROPS inputs | [INPUTS.md](INPUTS.md) |
| Configure reflector and station-bias adjustment | [ADJUSTMENT.md](ADJUSTMENT.md) |
| Understand module boundaries and extension points | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Review the GROOPS-aligned file and program roadmap | [GROOPS_FILE_PROGRAM_DESIGN.md](GROOPS_FILE_PROGRAM_DESIGN.md) |
| Build and validate the IERS native extension | [DEVELOPMENT.md](DEVELOPMENT.md) |

## Minimal command line

```bash
python -m llrops list-programs
python -m llrops list-classes
python -m llrops run configs/llrops_oc_residuals.yml
python -m llrops run configs/llrops_oc_residuals.yml --mpi
```

`--set name=value` overrides a value in the config `variables` section. Paths
are resolved relative to the config working directory unless they are absolute.

## Supported programs

`CrdToMini`, `NormalPointsToLlrops`, `LlrResiduals`, `LlrAdjustment`,
`LlrNormalEquations`, and `NormalsCombineSolve` are registered program tasks.
Each task is selected by a `program` entry in a YAML run configuration.

## Current conventions

- Runtime epochs are explicit two-part `Epoch` values with `UTC`, `TT`, or
  `TDB` scale.
- Normal-point uncertainty comes from the input record; it is not replaced by
  a station lookup table.
- Range-bias corrections in `globals.rangeBias` are deterministic forward
  corrections. Estimated `stationRangeBias` parameters are separate.
- The production Earth-orientation path uses explicit IERS C04 data, ERFA,
  and the private `llrops._iers2010` extension.
