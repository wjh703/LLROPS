"""MPI backend for distributed LLR forward-model tasks.

Serial execution is intentionally single-process.  Multi-rank execution is
provided by :mod:`llrops.parallel.mpi`; launch with
``mpirun/srun python -m llrops run cfg.yml --mpi`` and configure task size
per program with ``mpi: {chunksize: 8}``.  Supported by ``LlrResiduals``,
``LlrAdjustment`` and ``LlrNormalEquations``.
"""
