# LLROPS v32 MPI residual throughput fix

## Fixed

- `LlrResiduals` under MPI now returns projected table rows from worker ranks
  instead of shipping full `LlrObservationResult` objects back to rank 0.
  This restores the lightweight transport shape used by the older O-C MPI path.
- MPI workers now cache one `LlrObservationProcessor` per observation spec for
  the lifetime of the worker instead of rebuilding the processor graph for every
  small chunk.
- The serial fallback inside the MPI runtime now closes cached worker-side
  objects before returning.

## Unchanged

- `LlrAdjustment` still uses typed `LlrObservationResult` objects because it
  needs partials and equation metadata.
