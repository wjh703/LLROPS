# Programs

Run one or more registered tasks from a YAML file:

```bash
python -m llrops run config.yml
python -m llrops run config.yml --mpi
python -m llrops run config.yml --set dataDir=/data/llr
```

## Program reference

| Program | Required input | Main output |
|---|---|---|
| `CrdToMini` | `inputCrd`, `outputDir` | MINI files |
| `NormalPointsToLlrops` | `inputNormalPoints`, `outputFile` | versioned LLROPS JSONL |
| `LlrResiduals` | `inputNormalPoints` | standard/full O-C CSV or JSON |
| `LlrAdjustment` | `inputNormalPoints`, `parametrization` | adjustment JSON and optional normals |
| `LlrNormalEquations` | `inputNormalPoints`, `outputNormals` | fixed-linearization normals |
| `NormalsCombineSolve` | `inputNormals` | combined normals and optional solution JSON |

All programs share `variables`, `globals`, and the run context. Supported input
paths may be individual files or directories containing MINI, CRD/FRD, or
canonical LLROPS files.

## Residual run

```yaml
programs:
  - program: LlrResiduals
    inputNormalPoints: [data/polac/TOTALOBS6924.DAT]
    outputLevel: standard
    outputCsv: output/oc.csv
    mpi:
      chunksize: 32
```

`standard` is the compact O-C table; `full` adds model and correction
diagnostics. Input uncertainty is always record-owned.

## Adjustment run

Use `configs/llrops_reflector_bias_adjustment_detailed.yml` as the complete
reference. The essential sections are `globals`, `parametrization`,
`adjustment`, `initialization`, `robustEstimation`, `vce`, `mpi`, and output
paths. `LlrReflectorFit` and the old local `parallel` block are removed.

## Normal-equation workflow

Use `LlrNormalEquations` to create one fixed-linearization file per data arc,
then `NormalsCombineSolve` to align parameter names, add the systems, and solve
once. Use `LlrAdjustment` when parameter updates must be absorbed and the
observation model relinearized.
