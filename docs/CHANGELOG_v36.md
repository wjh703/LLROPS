# LLROPS v36 explicit MPI worker initialization

## Observation worker lifecycle

MPI observation processing now has an explicit initialization barrier:

1. rank 0 broadcasts the immutable observation spec;
2. rank 0 sends an initialization command to every worker;
3. every worker constructs and caches its observation processor, including its
   process-local CALCEPH handle;
4. every worker returns a ready response;
5. rank 0 waits for all ready responses;
6. only then does rank 0 start the observation timer and dispatch data chunks.

The first observation task can no longer initialize a processor lazily. A task
received without a completed initialization phase raises a lifecycle error.

## Diagnostics

- Rank 0 reports the total initialization time and the slowest worker time.
- Worker initialization failures include the worker rank and remote traceback.
- Rank 0 collects all ready/error responses before raising, avoiding workers
  being left mid-initialization.
- The formal progress line is printed immediately at `0/total` after all workers
  are ready.
- Observation throughput excludes processor construction and CALCEPH opening.

## Tests

Unit coverage now verifies:

- one-time spec broadcast;
- all-worker initialize/ready synchronization;
- initialization failure propagation;
- single-rank initialization;
- worker-side processor construction before the ready response.
